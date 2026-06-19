from __future__ import annotations

from modules.resume_ats.contracts import ResumeEntities, RoleJD
from modules.resume_ats.scoring.skill_synonyms import synonym_match
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
            kw_lower = kw.lower()
            # Tier 1: exact substring match
            if kw_lower in resume_lower:
                matched += 1
                continue
            # Tier 2: entity term overlap
            if any(kw_lower in term or term in kw_lower for term in terms):
                matched += 1
                continue
            # Tier 3: synonym graph match
            if any(synonym_match(term, kw) for term in terms):
                matched += 1

        return clamp_score((matched / len(keywords)) * 100)
