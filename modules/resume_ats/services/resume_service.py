from __future__ import annotations

import hashlib
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
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        resume_text, structured_proxy = resume_input_to_text_and_proxy(structured_resume)
        entities = self.entity_extractor.extract(structured_proxy)
        jd = load_jd(role)

        # Layer 1: deterministic ATS tournament scorers
        scores = self.scorer.score(structured_proxy, entities, jd)

        # Layer 2: LLM qualitative feedback only
        analysis = self.llm.analyze(role, resume_text, entities, jd)
        analysis_dict = analysis.model_dump()

        # Layer 3: Global role-fitting evaluation across all JDs
        # Determine top best-fit roles (family, sub_roles, fit score, reasoning)
        global_ranking = self.domain_classifier.suggest_best_role(entities, resume=structured_proxy)
        top_10 = global_ranking.get("top_10_best_fit_roles", [])
        career_track_summary = global_ranking.get("career_track_summary", "")

        top_5 = []
        for r in top_10[:5]:
            top_5.append({
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
        recommended_secondaries = [r["role_key"] for r in top_10[1:3]] if len(top_10) > 1 else []

        # If the supplied role is not among top 3, explicitly note it's not optimal
        supplied_role_assessment = (
            "Supplied role is not the optimal match; candidate aligns better with other domains." 
            if role.upper() not in [r["role_key"] for r in top_10[:3]] and top_10 and top_10[0]["fit_score"] > supplied_fit_score + 5
            else "Supplied role is a reasonable match." 
        )

        # Attach global fit info into analysis_dict
        analysis_dict.update({
            "supplied_role_fit_score": supplied_fit_score,
            "supplied_role_assessment": supplied_role_assessment,
            "top_5_best_fit_roles": top_5,
            "top_10_best_fit_roles": top_10,
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
    overall_score = int(scores.get("overall_score", 0))
    strengths = list(analysis_dict.get("strengths", []) or [])
    areas_of_improvement = list(analysis_dict.get("areas_of_improvement", []) or [])
    critical_missing_keywords = list(analysis_dict.get("critical_missing_keywords", []) or [])
    action_plan = list(analysis_dict.get("action_plan", []) or [])

    score_breakdown = _build_score_breakdown(overall_score, scores)
    score_level = _build_score_level(overall_score)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    payload = {
        "success": True,
        "cached": False,
        "analysis": {
            "overallScore": overall_score,
            "scoreBreakdown": score_breakdown,
            "strengths": strengths,
            "areasOfImprovement": areas_of_improvement,
            "criticalMissingKeywords": critical_missing_keywords,
            "actionPlan": action_plan,
        },
        "data": {
            "overallScore": overall_score,
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
        **scores,
        **analysis_dict,
        "strengths": strengths,
        "areas_of_improvement": areas_of_improvement,
        "critical_missing_keywords": critical_missing_keywords,
        "action_plan": action_plan,
    }
    return payload


def _build_score_breakdown(overall_score: int, scores: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_keyword = max(0.0, min(40.0, float(scores.get("keyword_score", 0)) * 0.4))
    raw_formatting = max(0.0, min(10.0, float(scores.get("resume_quality_score", 0)) * 0.1))
    raw_structure = max(
        0.0,
        min(
            20.0,
            ((float(scores.get("semantic_score", 0)) + float(scores.get("resume_quality_score", 0))) / 2.0) * 0.2,
        ),
    )
    raw_content = max(
        0.0,
        min(
            30.0,
            (
                (
                    float(scores.get("experience_score", 0))
                    + float(scores.get("skill_depth_score", 0))
                    + float(scores.get("domain_score", 0))
                )
                / 3.0
            )
            * 0.3,
        ),
    )

    total_raw = raw_keyword + raw_formatting + raw_structure + raw_content
    scale = (overall_score / total_raw) if total_raw > 0 else 0.0

    keyword_match = min(40, max(0, round(raw_keyword * scale)))
    formatting = min(10, max(0, round(raw_formatting * scale)))
    section_structure = min(20, max(0, round(raw_structure * scale)))
    content_quality = min(30, max(0, round(raw_content * scale)))

    diff = overall_score - (keyword_match + formatting + section_structure + content_quality)
    content_quality = min(30, max(0, content_quality + diff))

    return {
        "keywordMatch": {
            "score": keyword_match,
            "maxScore": 40,
            "description": "Relevance or learning potential.",
        },
        "formatting": {
            "score": formatting,
            "maxScore": 10,
            "description": "Grammar and readability.",
        },
        "sectionStructure": {
            "score": section_structure,
            "maxScore": 20,
            "description": "Section completeness.",
        },
        "contentQuality": {
            "score": content_quality,
            "maxScore": 30,
            "description": "Depth and clarity.",
        },
    }


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
