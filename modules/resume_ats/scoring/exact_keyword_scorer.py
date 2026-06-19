from __future__ import annotations

from modules.resume_ats.contracts import ResumeEntities, RoleJD
from modules.resume_ats.scoring.skill_synonyms import collection_matches_skill, text_matches_skill
from modules.resume_ats.scoring.utils import clamp_score, entity_terms


class ExactKeywordScorer:
    """Hard requirement match: JD keywords found in resume text, entities, or via synonym graph."""

    def score(self, resume_text: str, entities: ResumeEntities, jd: RoleJD) -> int:
        keywords = [k for k in jd.get("keywords", []) if k]
        if not keywords:
            return 0

        resume_lower = resume_text.lower()
        terms = entity_terms(resume_text, entities)

        matched = 0
        for kw in keywords:
            if text_matches_skill(resume_lower, kw):
                matched += 1
                continue
            if collection_matches_skill(terms, kw):
                matched += 1

        return clamp_score((matched / len(keywords)) * 100)
