"""Skill depth scorer — rewards contextual skill usage over keyword stuffing.

Scoring tiers:
  1. Keyword exists in skills list       → +1
  2. Used in a project description       → +2
  3. Used in experience bullet           → +3
  4. Has impact metrics alongside skill  → +5

Final score = weighted depth coverage across JD keywords.
"""

from __future__ import annotations

import re

from modules.resume_ats.contracts import ResumeEntities, RoleJD, StructuredResume
from modules.resume_ats.scoring.skill_synonyms import synonym_match
from modules.resume_ats.scoring.utils import clamp_score

# Impact-indicator patterns (numbers near skill usage = deeper application)
IMPACT_NEARBY = re.compile(
    r"\d+\s*%|\$\s*[\d,.]+|\d+[kKmMbB]\+?"
    r"|\d+\s*x\s*(?:faster|speedup|improvement|throughput|reduction)",
    re.I,
)


class SkillDepthScorer:
    """Scores how deeply JD skills are demonstrated across resume sections."""

    def score(
        self,
        resume: StructuredResume,
        entities: ResumeEntities,
        jd: RoleJD,
    ) -> int:
        keywords = [k for k in jd.get("keywords", []) if k]
        required = [s for s in jd.get("required_skills", []) if s]
        preferred = [s for s in jd.get("preferred_skills", []) if s]

        all_jd_skills = list(dict.fromkeys(required + preferred + keywords))
        if not all_jd_skills:
            return 0

        # Build searchable text blocks per section
        skills_list_text = " ".join(resume.get("skills", [])).lower()
        experience_texts = self._experience_texts(resume)
        project_texts = self._project_texts(resume)

        total_depth = 0.0
        max_possible = 0.0

        for skill in all_jd_skills:
            is_required = skill in required
            weight = 1.5 if is_required else 1.0
            max_possible += 5.0 * weight

            depth = self._skill_depth(
                skill, skills_list_text, experience_texts, project_texts
            )
            total_depth += depth * weight

        if max_possible == 0:
            return 0

        return clamp_score((total_depth / max_possible) * 100)

    def _skill_depth(
        self,
        skill: str,
        skills_list_text: str,
        experience_texts: list[str],
        project_texts: list[str],
    ) -> float:
        """Return depth score for a single skill (0–5)."""
        depth = 0.0

        # Tier 1: In skills list
        if self._skill_in_text(skill, skills_list_text):
            depth = 1.0

        # Tier 2: Used in project description
        for proj_text in project_texts:
            if self._skill_in_text(skill, proj_text):
                depth = max(depth, 2.0)
                # Tier 4: Impact metrics near skill in project
                if IMPACT_NEARBY.search(proj_text):
                    depth = max(depth, 5.0)
                break

        # Tier 3: Used in experience bullet
        for exp_text in experience_texts:
            if self._skill_in_text(skill, exp_text):
                depth = max(depth, 3.0)
                # Tier 4: Impact metrics near skill in experience
                if IMPACT_NEARBY.search(exp_text):
                    depth = max(depth, 5.0)
                break

        return depth

    def _skill_in_text(self, skill: str, text: str) -> bool:
        """Check if skill appears in text, including synonym matching."""
        lower_skill = skill.lower()
        if lower_skill in text:
            return True
        # Check synonyms
        words = re.findall(r"[a-z][a-z0-9+#./ -]*", text)
        for word_group in self._sliding_window(words, 3):
            phrase = " ".join(word_group)
            if synonym_match(phrase, skill):
                return True
        return False

    @staticmethod
    def _sliding_window(tokens: list[str], size: int):
        for i in range(len(tokens) - size + 1):
            yield tokens[i : i + size]

    def _experience_texts(self, resume: StructuredResume) -> list[str]:
        """Flatten experience section into searchable text blocks."""
        texts: list[str] = []
        for exp in resume.get("experience", []):
            for bullet in exp.get("bullets", []):
                texts.append(bullet.lower())
            if exp.get("description"):
                texts.append(exp["description"].lower())
            title_company = f"{exp.get('title', '')} {exp.get('company', '')}".lower()
            if title_company.strip():
                texts.append(title_company)
        return texts

    def _project_texts(self, resume: StructuredResume) -> list[str]:
        """Flatten project section into searchable text blocks."""
        texts: list[str] = []
        for proj in resume.get("projects", []):
            parts = []
            if proj.get("description"):
                parts.append(proj["description"])
            if proj.get("technologies"):
                parts.append(" ".join(proj["technologies"]))
            if proj.get("name"):
                parts.append(proj["name"])
            if parts:
                texts.append(" ".join(parts).lower())
        return texts
