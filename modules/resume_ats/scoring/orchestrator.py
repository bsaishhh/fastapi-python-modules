from __future__ import annotations

import logging

from app.core.config import settings
from modules.resume_ats.contracts import ResumeEntities, RoleJD, StructuredResume
from modules.resume_ats.scoring.bm25_scorer import BM25Scorer
from modules.resume_ats.scoring.domain_scorer import DomainScorer
from modules.resume_ats.scoring.exact_keyword_scorer import ExactKeywordScorer
from modules.resume_ats.scoring.experience_scorer import ExperienceScorer
from modules.resume_ats.scoring.jaccard_scorer import JaccardScorer
from modules.resume_ats.scoring.resume_quality_scorer import ResumeQualityScorer
from modules.resume_ats.scoring.semantic_scorer import SemanticScorer
from modules.resume_ats.scoring.skill_depth_scorer import SkillDepthScorer
from modules.resume_ats.scoring.skill_synonyms import collection_matches_skill, text_matches_skill
from modules.resume_ats.scoring.tfidf_scorer import TfidfScorer
from modules.resume_ats.scoring.utils import SCORING_WEIGHTS, clamp_score, entity_terms, jd_to_text, resume_to_text
from modules.resume_ats.domain_classifier.classifier import DomainClassifier

logger = logging.getLogger(__name__)


class ScoringOrchestrator:
    """ATS v2 deterministic scoring ensemble.

    Pipeline:
      1. Section-weighted Semantic Score  (embedding model)
      2. BM25 Score                       (replaces Jaccard)
      3. TF-IDF Keyword Score
      4. Exact Keyword Score              (with synonym expansion)
      5. Domain Score                     (with synonym expansion)
      6. Experience Score                 (role/impact/seniority/duration)
      7. Skill Depth Score                (contextual usage depth)
      8. Resume Quality Score             (independent of JD)

    Final ATS score = weighted ensemble of 1–7.
    Resume quality score reported separately.
    """

    def __init__(self) -> None:
        self.semantic = SemanticScorer()
        self.tfidf = TfidfScorer()
        self.bm25 = BM25Scorer()
        self.jaccard = JaccardScorer()
        self.exact_keyword = ExactKeywordScorer()
        self.domain = DomainScorer()
        self.experience = ExperienceScorer()
        self.skill_depth = SkillDepthScorer()
        self.resume_quality = ResumeQualityScorer()
        self.domain_classifier = DomainClassifier()

    def score(
        self,
        resume: StructuredResume,
        entities: ResumeEntities,
        jd: RoleJD,
    ) -> dict:
        resume_text = resume_to_text(resume)
        jd_text = jd_to_text(jd)

        # ── Layer 0: domain classification confidence boost ──────────────
        domain_classification = self.domain_classifier.suggest_best_role(entities, resume=resume)
        best_role = domain_classification.get("best_role", "")
        classifier_confidence = domain_classification.get("domain_score", 0)

        # ── Layer 1: section-weighted semantic ────────────────────────────
        semantic_score = self.semantic.score(resume_text, jd, resume=resume)

        # ── Layer 2: BM25 (replaces Jaccard) ──────────────────────────────
        bm25_score = self.bm25.score(resume_text, jd)

        # ── Layer 2b: Jaccard (computed for UI compatibility) ─────────────
        jaccard_score = self.jaccard.score(resume_text, jd)

        # ── Layer 3: TF-IDF keyword relevance ─────────────────────────────
        keyword_score = self.tfidf.score(resume_text, jd_text)

        # ── Layer 4: exact keyword + synonym match ────────────────────────
        exact_keyword_score = self.exact_keyword.score(resume_text, entities, jd)

        # ── Layer 5: domain / required-skill coverage ─────────────────────
        domain_score = self.domain.score(resume_text, entities, jd)

        # ── Layer 6: experience scoring engine ────────────────────────────
        experience_score = self.experience.score(resume, entities, jd)

        # ── Layer 7: skill depth scoring ──────────────────────────────────
        skill_depth_score = self.skill_depth.score(resume, entities, jd)

        # ── Layer 8: resume quality (JD-independent) ──────────────────────
        resume_quality_score = self.resume_quality.score(resume, entities)

        # ── Domain classifier confidence boost ────────────────────────────
        # If classifier strongly agrees this resume matches the target role,
        # give a confidence-based boost to the overall score (max +8 points).
        # Lower the activation threshold to 60 so strong matches are rewarded.
        domain_boost = 0.0
        target_role = jd.get("role_key", "").upper()
        if best_role == target_role and classifier_confidence >= 60:
            # Scale: each 4 points above 60 ≈ +1 score point, capped at +8
            domain_boost = min(8.0, (classifier_confidence - 60) / 4.0)

        # ── Weighted ensemble ─────────────────────────────────────────────
        overall_score = clamp_score(
            SCORING_WEIGHTS["semantic"] * semantic_score
            + SCORING_WEIGHTS["bm25"] * bm25_score
            + SCORING_WEIGHTS["keyword"] * keyword_score
            + SCORING_WEIGHTS["exact_keyword"] * exact_keyword_score
            + SCORING_WEIGHTS["domain"] * domain_score
            + SCORING_WEIGHTS["experience"] * experience_score
            + SCORING_WEIGHTS["skill_depth"] * skill_depth_score
            + domain_boost
        )

        # ── Missing keyword severity levels ───────────────────────────────
        missing_severity = self._classify_missing_severity(resume_text, entities, jd)

        return {
            "overall_score": overall_score,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "bm25_score": bm25_score,
            "jaccard_score": jaccard_score,
            "exact_keyword_score": exact_keyword_score,
            "domain_score": domain_score,
            "experience_score": experience_score,
            "skill_depth_score": skill_depth_score,
            "resume_quality_score": resume_quality_score,
            "domain_boost": round(domain_boost, 1),
            "missing_keyword_severity": missing_severity,
            "detected_domain": best_role,
            "scoring_version": settings.scoring_version,
            "embedding_model": settings.sbert_model,
        }

    def _classify_missing_severity(
        self,
        resume_text: str,
        entities: ResumeEntities,
        jd: RoleJD,
    ) -> dict[str, list[str]]:
        """Classify missing keywords into CRITICAL / IMPORTANT / OPTIONAL tiers."""
        resume_terms = entity_terms(resume_text, entities)
        resume_lower = resume_text.lower()

        def is_present(term: str) -> bool:
            if text_matches_skill(resume_lower, term):
                return True
            if collection_matches_skill(resume_terms, term):
                return True
            return False

        critical: list[str] = []
        important: list[str] = []
        optional: list[str] = []

        # Required skills → CRITICAL if missing
        for skill in jd.get("required_skills", []):
            if not is_present(skill):
                critical.append(skill)

        # Keywords not already in required → IMPORTANT
        required_set = {s.lower() for s in jd.get("required_skills", [])}
        for kw in jd.get("keywords", []):
            if kw.lower() in required_set:
                continue
            if not is_present(kw):
                important.append(kw)

        # Preferred skills → OPTIONAL if missing
        for skill in jd.get("preferred_skills", []):
            if not is_present(skill):
                optional.append(skill)

        return {
            "critical": list(dict.fromkeys(critical))[:5],
            "important": list(dict.fromkeys(important))[:5],
            "optional": list(dict.fromkeys(optional))[:5],
        }
