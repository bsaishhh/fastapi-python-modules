from __future__ import annotations

import hashlib
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from modules.resume_ats.analysis.llm_analyzer import LLMAnalyzer
from modules.resume_ats.contracts import StructuredResume
from modules.resume_ats.data.jd_loader import load_jd
from modules.resume_ats.db.repository import ResumeRepository
from modules.resume_ats.domain_classifier.classifier import DomainClassifier
from modules.resume_ats.entities.extractor import EntityExtractor
from modules.resume_ats.extraction.pipeline import ExtractionPipeline
from modules.resume_ats.extraction.text_adapter import resume_input_to_text_and_proxy
from modules.resume_ats.scoring.orchestrator import ScoringOrchestrator
from modules.resume_ats.scoring.skill_synonyms import text_matches_skill


class ExtractService:
    def __init__(self) -> None:
        self.pipeline = ExtractionPipeline()
        self.repo = ResumeRepository()

    async def extract(
        self,
        file_bytes: bytes,
        filename: str = "resume.pdf",
        bearer_token: str | None = None,
    ) -> str:
        return await self.pipeline.run(file_bytes, filename=filename, bearer_token=bearer_token)

    async def extract_and_save(
        self,
        file_bytes: bytes,
        filename: str,
        role: str | None = None,
        bearer_token: str | None = None,
    ) -> dict[str, Any]:
        extracted_text = await self.pipeline.run(
            file_bytes,
            filename=filename,
            bearer_token=bearer_token,
        )
        content_hash = compute_content_hash(file_bytes)
        resume_id = await self.repo.save_resume_upload(
            filename,
            content_hash,
            role,
            extracted_text=extracted_text,
        )
        extraction_id = await self.repo.save_extraction(resume_id, extracted_text)
        return {
            "resume_id": resume_id,
            "extraction_id": extraction_id,
            "extracted_text": extracted_text,
            "extraction_version": settings.extraction_version,
        }


class ResumeAnalysisService:
    def __init__(self) -> None:
        self.extractor = ExtractionPipeline()
        self.entity_extractor = EntityExtractor()
        self.scorer = ScoringOrchestrator()
        self.llm = LLMAnalyzer()
        self.domain_classifier = DomainClassifier()
        self.repo = ResumeRepository()

    async def analyze_from_file(
        self,
        file_bytes: bytes,
        filename: str,
        role: str,
        bearer_token: str | None = None,
        custom_jd: str | None = None,
    ) -> dict[str, Any]:
        extract_service = ExtractService()
        extracted = await extract_service.extract_and_save(
            file_bytes,
            filename,
            role,
            bearer_token=bearer_token,
        )
        return await self.analyze_from_structured(
            role=role,
            structured_resume=extracted["extracted_text"],
            resume_id=extracted["resume_id"],
            extraction_id=extracted["extraction_id"],
            custom_jd=custom_jd,
        )

    async def score_only(self, structured_resume: str | dict[str, Any], role: str) -> dict[str, Any]:
        """Deterministic ATS scores without LLM or MongoDB writes."""
        _, structured_proxy = resume_input_to_text_and_proxy(structured_resume)
        entities = self.entity_extractor.extract(structured_proxy)
        jd = load_jd(role)
        return self.scorer.score(structured_proxy, entities, jd)

    async def analyze_from_structured(
        self,
        role: str,
        structured_resume: str | dict[str, Any],
        resume_id: str | None = None,
        extraction_id: str | None = None,
        custom_jd: str | None = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        resume_text, structured_proxy = resume_input_to_text_and_proxy(structured_resume)
        entities = self.entity_extractor.extract(structured_proxy)
        
        # Use custom JD if provided, otherwise load system JD
        if custom_jd:
            jd = _parse_custom_jd(custom_jd, role)
        else:
            jd = load_jd(role)

        # Layer 1: deterministic ATS tournament scorers
        scores = self.scorer.score(structured_proxy, entities, jd)

        # Layer 2: LLM qualitative feedback only
        analysis = self.llm.analyze(role, resume_text, entities, jd)
        analysis_dict = analysis.model_dump()
        llm_score = int(analysis_dict.get("overall_score", scores.get("overall_score", 0)))
        
        # Apply mandatory skill penalty - if critical required skills are missing, cap score at 70
        critical_missing = scores.get("missing_keyword_severity", {}).get("critical", [])
        if critical_missing:
            llm_score = min(70, llm_score)
        
        analysis_dict["llm_overall_score"] = llm_score
        # Use scoring system's critical missing keywords, not LLM's (LLM may return empty)
        analysis_dict["critical_missing_keywords"] = critical_missing

        # Layer 3: Global role-fitting evaluation across all JDs
        # Determine top best-fit roles (family, sub_roles, fit score, reasoning)
        global_ranking = self.domain_classifier.suggest_best_role(entities, resume=structured_proxy)
        top_10 = global_ranking.get("top_10_best_fit_roles", [])
        career_track_summary = global_ranking.get("career_track_summary", "")

        # Limit to top 2 roles only for DB storage
        top_2 = []
        for r in top_10[:2]:
            top_2.append({
                "role_key": r["role_key"],
                "role_family": r["role_family"],
                "sub_roles": r["sub_roles"],
                "fit_score": r["fit_score"],
                "confidence_level": r["confidence_level"],
                "matching_evidence": r["matching_evidence"],
                "missing_evidence": r["missing_evidence"],
                "reasoning": r["reasoning"],
            })

        # Supplied role assessment
        supplied_assessment = self.domain_classifier.classify(entities, role, resume=structured_proxy)
        supplied_fit_score = int(supplied_assessment.get("domain_score", 0))

        # Decide recommended primary/secondary roles based on career fit
        recommended_primary = top_10[0]["role_key"] if top_10 else role.upper()
        recommended_secondaries = [r["role_key"] for r in top_10[1:2]] if len(top_10) > 1 else []

        # If the supplied role is not among top 2, explicitly note it's not optimal
        supplied_role_assessment = (
            "Supplied role is not the optimal match; candidate aligns better with other domains." 
            if role.upper() not in [r["role_key"] for r in top_10[:2]] and top_10 and top_10[0]["fit_score"] > supplied_fit_score + 5
            else "Supplied role is a reasonable match." 
        )

        # Attach global fit info into analysis_dict for DB storage (not returned in response)
        analysis_dict.update({
            "supplied_role_fit_score": supplied_fit_score,
            "supplied_role_assessment": supplied_role_assessment,
            "top_2_best_fit_roles": top_2,
            "top_10_best_fit_roles": top_10,  # Keep full list for DB
            "recommended_primary_role": recommended_primary,
            "recommended_secondary_roles": recommended_secondaries,
            "career_track_summary": career_track_summary,
        })

        if resume_id and extraction_id:
            analysis_id = await self.repo.save_domain_analysis(
                resume_id=resume_id,
                extraction_id=extraction_id,
                role=role,
                master_jd=jd,
                scores=scores,
                analysis=analysis_dict,
            )
        else:
            content_hash = hashlib.sha256(resume_text.encode()).hexdigest()
            resume_id = await self.repo.save_resume_upload(
                "structured_input.txt",
                content_hash,
                role,
                extracted_text=resume_text,
            )
            extraction_id = await self.repo.save_extraction(resume_id, resume_text)
            analysis_id = await self.repo.save_domain_analysis(
                resume_id=resume_id,
                extraction_id=extraction_id,
                role=role,
                master_jd=jd,
                scores=scores,
                analysis=analysis_dict,
            )

        processing_time_ms = int((time.perf_counter() - started_at) * 1000)
        payload = _build_analysis_response(
            role=role,
            scores=scores,
            analysis_dict=analysis_dict,
            processing_time_ms=processing_time_ms,
        )
        payload.update({
            "resume_id": resume_id,
            "extraction_id": extraction_id,
            "analysis_id": analysis_id,
        })
        return payload

    async def domain_detect(
        self,
        file_bytes: bytes,
        role: str | None = None,
        department: str | None = None,
        bearer_token: str | None = None,
    ) -> dict[str, Any]:
        extracted_text = await self.extractor.run(file_bytes, bearer_token=bearer_token)
        _, structured_proxy = resume_input_to_text_and_proxy(extracted_text)
        entities = self.entity_extractor.extract(structured_proxy)

        if department:
            return self.domain_classifier.detect_from_department(department)
        if role:
            return self.domain_classifier.classify(entities, role, resume=structured_proxy)
        return self.domain_classifier.suggest_best_role(entities, resume=structured_proxy)


def compute_content_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def generate_upload_key(filename: str) -> str:
    return f"uploads/{uuid.uuid4()}/{filename}"


def _build_analysis_response(
    role: str,
    scores: dict[str, Any],
    analysis_dict: dict[str, Any],
    processing_time_ms: int,
) -> dict[str, Any]:
    score_breakdown = _build_score_breakdown(scores)
    deterministic_overall_score = int(scores.get("overall_score", 0))
    llm_overall_score = int(analysis_dict.get("llm_overall_score", analysis_dict.get("overall_score", deterministic_overall_score)))
    
    # Clean markdown artifacts from text fields
    strengths = [_remove_markdown_artifacts(s) for s in (analysis_dict.get("strengths", []) or [])]
    areas_of_improvement = [_remove_markdown_artifacts(s) for s in (analysis_dict.get("areas_of_improvement", []) or [])]
    critical_missing_keywords = list(analysis_dict.get("critical_missing_keywords", []) or [])
    action_plan = [_clean_action_plan_item(item) for item in (analysis_dict.get("action_plan", []) or [])]

    # Use LLM score for score level
    score_level = _build_score_level(llm_overall_score)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    payload = {
        "success": True,
        "cached": False,
        "data": {
            "llm_overall_score": llm_overall_score,
            "scoreLevel": score_level,
            "scoreBreakdown": score_breakdown,
            "strengths": strengths,
            "areasOfImprovement": areas_of_improvement,
            "criticalMissingKeywords": critical_missing_keywords,
            "actionPlan": action_plan,
            "screenshot": None,
            "role": role.lower(),
            "processingTimeMs": processing_time_ms,
        },
        "timestamp": timestamp,
        "requestId": f"ats_opt_{int(time.time() * 1000)}",
        "role": role.upper(),
        "analysis_version": settings.analysis_version,
        "detected_domain": analysis_dict.get("detected_domain", scores.get("detected_domain")),
        "scoring_version": settings.scoring_version,
        "recommended_primary_role": analysis_dict.get("recommended_primary_role"),
        "deterministic_overall_score": deterministic_overall_score,
        "llm_overall_score": llm_overall_score,
    }
    return payload


def _build_score_breakdown(scores: dict[str, Any]) -> dict[str, dict[str, Any]]:
    keyword_match = max(
        0,
        min(
            10,
            round(
                (
                    float(scores.get("keyword_score", 0)) * 0.25
                    + float(scores.get("exact_keyword_score", 0)) * 0.45
                    + float(scores.get("domain_score", 0)) * 0.30
                )
                / 10.0
            ),
        ),
    )
    formatting = max(0, min(10, round(float(scores.get("resume_quality_score", 0)) / 10.0)))
    section_structure = max(
        0,
        min(
            10,
            round(
                (
                    float(scores.get("semantic_score", 0)) * 0.35
                    + float(scores.get("domain_score", 0)) * 0.35
                    + float(scores.get("experience_score", 0)) * 0.30
                )
                / 10.0
            ),
        ),
    )
    content_quality = max(
        0,
        min(
            10,
            round(
                (
                    float(scores.get("experience_score", 0)) * 0.40
                    + float(scores.get("skill_depth_score", 0)) * 0.35
                    + float(scores.get("semantic_score", 0)) * 0.25
                )
                / 10.0
            ),
        ),
    )

    return {
        "keywordMatch": {
            "score": keyword_match,
            "maxScore": 10,
            "description": "Keyword and inferred-skill alignment.",
        },
        "formatting": {
            "score": formatting,
            "maxScore": 10,
            "description": "Grammar and readability.",
        },
        "sectionStructure": {
            "score": section_structure,
            "maxScore": 10,
            "description": "Other evidence and role alignment.",
        },
        "contentQuality": {
            "score": content_quality,
            "maxScore": 10,
            "description": "Depth and clarity.",
        },
    }


def _build_weighted_overall_score(score_breakdown: dict[str, dict[str, Any]]) -> int:
    weights = {
        "keywordMatch": 0.40,
        "formatting": 0.10,
        "sectionStructure": 0.20,
        "contentQuality": 0.30,
    }
    total = 0.0
    for key, weight in weights.items():
        component = score_breakdown.get(key, {})
        score = float(component.get("score", 0))
        max_score = float(component.get("maxScore", 10)) or 10.0
        total += (score / max_score) * weight * 100.0
    return int(round(total))


def _build_score_level(overall_score: int) -> dict[str, str]:
    if overall_score >= 85:
        return {
            "level": "Excellent",
            "color": "#10b981",
            "description": "Excellent ATS compatibility",
        }
    if overall_score >= 70:
        return {
            "level": "Good",
            "color": "#3b82f6",
            "description": "Strong ATS compatibility",
        }
    if overall_score >= 50:
        return {
            "level": "Fair",
            "color": "#f59e0b",
            "description": "Moderate ATS compatibility",
        }
    return {
        "level": "Needs Improvement",
        "color": "#ef4444",
        "description": "Low ATS compatibility",
    }


def _remove_markdown_artifacts(text: str) -> str:
    """Remove markdown artifacts like **, *, ##, etc. from text."""
    if not text:
        return text
    # Remove bold/italic markers
    text = text.replace("**", "").replace("__", "")
    text = text.replace("*", "").replace("_", "")
    # Remove headers
    text = text.replace("###", "").replace("##", "").replace("#", "")
    # Remove code blocks
    text = text.replace("```", "").replace("`", "")
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


def _clean_action_plan_item(item: dict[str, str]) -> dict[str, str]:
    """Clean markdown artifacts from action plan item fields."""
    if not isinstance(item, dict):
        return item
    return {
        "priority": _remove_markdown_artifacts(item.get("priority", "")),
        "action": _remove_markdown_artifacts(item.get("action", "")),
        "impact": _remove_markdown_artifacts(item.get("impact", "")),
        "example": _remove_markdown_artifacts(item.get("example", "")),
    }


def _parse_custom_jd(custom_jd_text: str, role: str) -> dict[str, Any]:
    """Parse custom JD text into JD format expected by the system.
    
    This function extracts key information from custom JD text and structures it
    to match the system JD format with required_skills, preferred_skills, keywords, etc.
    """
    # Load system JD as fallback for structure
    try:
        system_jd = load_jd(role)
    except Exception:
        system_jd = {
            "role_key": role.upper(),
            "family": role.upper(),
            "required_skills": [],
            "preferred_skills": [],
            "keywords": [],
            "sub_roles": [],
        }
    
    # Extract skills and keywords from custom JD text
    # Look for common patterns like "Required:", "Must have:", "Skills:", etc.
    text_lower = custom_jd_text.lower()
    
    # Split into sections
    required_section = ""
    preferred_section = ""
    
    # Try to find required skills section
    required_patterns = [
        r'required\s*(?:skills|qualifications|technologies?):?\s*([^\n]+)',
        r'must\s*(?:have|possess):\s*([^\n]+)',
        r'essential\s*(?:skills|requirements):\s*([^\n]+)',
    ]
    
    preferred_patterns = [
        r'preferred\s*(?:skills|qualifications|technologies?):?\s*([^\n]+)',
        r'nice\s*to\s*have:\s*([^\n]+)',
        r'desired\s*(?:skills|qualifications):\s*([^\n]+)',
    ]
    
    for pattern in required_patterns:
        match = re.search(pattern, text_lower)
        if match:
            required_section = match.group(1)
            break
    
    for pattern in preferred_patterns:
        match = re.search(pattern, text_lower)
        if match:
            preferred_section = match.group(1)
            break
    
    # Extract individual skills/keywords
    def extract_skills(text: str) -> list[str]:
        if not text:
            return []
        # Split by common delimiters
        skills = re.split(r'[,;•\n]|and\s+', text)
        # Clean up
        skills = [s.strip().title() for s in skills if s.strip() and len(s.strip()) > 2]
        return skills[:10]  # Limit to top 10
    
    required_skills = extract_skills(required_section)
    preferred_skills = extract_skills(preferred_section)
    
    # If no clear sections found, extract all technical terms from the text
    if not required_skills and not preferred_skills:
        # Extract common technical terms
        tech_keywords = re.findall(r'\b(?:python|java|javascript|typescript|react|angular|vue|node\.js|docker|kubernetes|aws|azure|gcp|sql|nosql|mongodb|postgresql|redis|git|ci/cd|agile|scrum|machine learning|deep learning|ai|nlp|data science|cloud|devops|linux|api|rest|graphql|microservices|frontend|backend|fullstack|database)\b', text_lower, re.IGNORECASE)
        required_skills = [kw.title() for kw in tech_keywords[:8]]
    
    # Build custom JD structure
    custom_jd = {
        "role_key": role.upper(),
        "family": system_jd.get("family", role.upper()),
        "required_skills": required_skills if required_skills else system_jd.get("required_skills", []),
        "preferred_skills": preferred_skills if preferred_skills else system_jd.get("preferred_skills", []),
        "keywords": (required_skills + preferred_skills) if (required_skills or preferred_skills) else system_jd.get("keywords", []),
        "sub_roles": system_jd.get("sub_roles", []),
        "custom_jd": True,  # Flag to indicate this is a custom JD
        "custom_jd_text": custom_jd_text,
    }
    
    return custom_jd
