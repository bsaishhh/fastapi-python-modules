from __future__ import annotations

from modules.resume_ats.contracts import ResumeEntities, RoleJD
from modules.resume_ats.scoring.skill_synonyms import synonym_match
from modules.resume_ats.scoring.utils import clamp_score, entity_terms


class DomainScorer:
    """Required-skill coverage with synonym-aware matching.

    Scores matched_required / total_required from benchmark JD,
    expanding matches via the skill synonym graph.
    """

    def score(self, resume_text: str, entities: ResumeEntities, jd: RoleJD) -> int:
        required = [s for s in jd.get("required_skills", []) if s]
        if not required:
            return 0

        resume_lower = resume_text.lower()
        terms = entity_terms(resume_text, entities)

        matched = 0
        for skill in required:
            skill_lower = skill.lower()
            # Tier 1: exact substring
            if skill_lower in resume_lower:
                matched += 1
                continue
            # Tier 2: entity term overlap
            if any(skill_lower in term or term in skill_lower for term in terms):
                matched += 1
                continue
            # Tier 3: synonym graph
            if any(synonym_match(term, skill) for term in terms):
                matched += 1

        return clamp_score((matched / len(required)) * 100)
