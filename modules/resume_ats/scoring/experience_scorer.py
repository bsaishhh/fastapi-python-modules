"""Experience scoring engine.

Scores resume experience on:
  - role_alignment: how well past titles align with target JD
  - impact: quantified achievements (numbers, metrics, percentages)
  - seniority: title level relative to JD expectations
  - duration: total years of relevant experience
"""

from __future__ import annotations

import re

from modules.resume_ats.contracts import ResumeEntities, RoleJD, StructuredResume
from modules.resume_ats.scoring.utils import clamp_score

# Seniority level mapping (higher = more senior)
SENIORITY_LEVELS: dict[str, int] = {
    "intern": 1,
    "trainee": 1,
    "co-op": 1,
    "junior": 2,
    "associate": 2,
    "graduate": 2,
    "engineer": 3,
    "developer": 3,
    "analyst": 3,
    "scientist": 3,
    "designer": 3,
    "consultant": 3,
    "senior": 4,
    "sr": 4,
    "lead": 5,
    "staff": 5,
    "principal": 6,
    "architect": 6,
    "manager": 5,
    "engineering manager": 6,
    "director": 7,
    "vp": 8,
    "head": 8,
    "cto": 9,
    "founder": 7,
    "co-founder": 7,
}

# Impact-indicator regex patterns
IMPACT_PATTERNS = [
    # Percentages: "reduced latency by 40%", "improved accuracy by 25%"
    re.compile(r"(?:reduc|improv|increas|decreas|boost|cut|grew|raised|lowered)\w*\s+\w+\s*(?:by\s+)?\d+\s*%", re.I),
    # Absolute numbers with units: "served 1M users", "processed 500K records"
    re.compile(r"\d+[kKmMbB]?\s*(?:user|record|request|transaction|customer|client|event)s?", re.I),
    # Time improvements: "2x faster", "3x speedup", "10x throughput"
    re.compile(r"\d+\s*x\s*(?:faster|speedup|throughput|improvement|performance)", re.I),
    # Revenue / cost: "$X million", "saved $X"
    re.compile(r"\$\s*\d+[\d,.]*\s*(?:k|m|b|million|billion)?", re.I),
    # Scale indicators: "scaled to X", "supporting X users"
    re.compile(r"(?:scal|support|handl|serv)\w+\s+(?:to\s+)?\d+", re.I),
    # Latency/SLA: "p99 < 200ms", "99.9% uptime"
    re.compile(r"(?:p\d{2}|latency|uptime|sla)\s*[<>=]\s*\d+", re.I),
]

# Action verb list for impact detection
IMPACT_VERBS = {
    "built", "designed", "architected", "developed", "deployed", "launched",
    "optimized", "reduced", "improved", "increased", "scaled", "automated",
    "migrated", "refactored", "implemented", "integrated", "delivered",
    "led", "mentored", "spearheaded", "pioneered", "drove", "achieved",
}


class ExperienceScorer:
    """Scores resume experience on role alignment, impact, seniority, and duration."""

    def score(
        self,
        resume: StructuredResume,
        entities: ResumeEntities,
        jd: RoleJD,
    ) -> int:
        role_alignment = self._role_alignment(resume, jd)
        impact = self._impact_score(resume)
        seniority = self._seniority_score(resume, jd)
        duration = self._duration_score(entities)

        # Weighted combination
        combined = (
            0.35 * role_alignment
            + 0.25 * impact
            + 0.20 * seniority
            + 0.20 * duration
        )
        return clamp_score(combined)

    def _role_alignment(self, resume: StructuredResume, jd: RoleJD) -> float:
        """How well past job titles align with the target JD role."""
        jd_keywords = {k.lower() for k in (
            jd.get("keywords", [])
            + jd.get("required_skills", [])
            + jd.get("responsibilities", [])
        )}
        sub_roles = {r.lower() for r in jd.get("sub_roles", [])}

        titles = [exp.get("title", "").lower() for exp in resume.get("experience", [])]
        if not titles:
            return 0.0

        score = 0.0
        for title in titles:
            title_words = set(re.findall(r"[a-z]+", title))
            # Check direct overlap with JD keywords
            overlap = len(title_words & jd_keywords)
            score += min(overlap * 15, 60)
            # Check if title matches any sub-role
            for sub in sub_roles:
                if sub in title or title in sub:
                    score += 40
                    break
        return min(score / max(len(titles), 1), 100.0)

    def _impact_score(self, resume: StructuredResume) -> float:
        """Quantified achievement density across all experience bullets."""
        all_bullets: list[str] = []
        for exp in resume.get("experience", []):
            all_bullets.extend(exp.get("bullets", []))
            if exp.get("description"):
                all_bullets.append(exp["description"])
        for proj in resume.get("projects", []):
            if proj.get("description"):
                all_bullets.append(proj["description"])

        if not all_bullets:
            return 0.0

        impact_hits = 0
        verb_hits = 0
        for bullet in all_bullets:
            # Check for quantified impact patterns
            if any(p.search(bullet) for p in IMPACT_PATTERNS):
                impact_hits += 1
            # Check for strong action verbs at start of bullet
            first_word = bullet.split()[0].lower().rstrip(".,;:") if bullet.split() else ""
            if first_word in IMPACT_VERBS:
                verb_hits += 1

        bullet_count = max(len(all_bullets), 1)
        impact_ratio = impact_hits / bullet_count
        verb_ratio = verb_hits / bullet_count

        # Score: up to 100 based on impact density
        return min((impact_ratio * 70 + verb_ratio * 30) * 100 / 40, 100.0)

    def _seniority_score(self, resume: StructuredResume, jd: RoleJD) -> float:
        """Seniority level match between resume titles and JD expectations."""
        jd_family = jd.get("family", "").lower()
        # Expected seniority based on JD (default mid-level)
        expected_level = 3  # mid-level default

        titles = [exp.get("title", "").lower() for exp in resume.get("experience", [])]
        if not titles:
            return 0.0

        # Detect expected seniority from JD keywords
        jd_text = " ".join(jd.get("keywords", []) + jd.get("responsibilities", [])).lower()
        if "senior" in jd_text or "lead" in jd_text:
            expected_level = 4
        elif "junior" in jd_text or "entry" in jd_text or "intern" in jd_text:
            expected_level = 2

        # Get max seniority from resume
        max_level = 0
        for title in titles:
            for keyword, level in SENIORITY_LEVELS.items():
                if keyword in title:
                    max_level = max(max_level, level)

        if max_level == 0:
            return 30.0  # Unknown title — baseline

        # Score based on how close resume seniority is to expected
        diff = abs(max_level - expected_level)
        if diff == 0:
            return 100.0
        elif diff == 1:
            return 80.0
        elif diff == 2:
            return 60.0
        elif diff == 3:
            return 40.0
        else:
            return 20.0

    def _duration_score(self, entities: ResumeEntities) -> float:
        """Score based on total years of experience."""
        years = entities.get("experience_years", 0.0)
        if years >= 5:
            return 100.0
        elif years >= 3:
            return 80.0
        elif years >= 2:
            return 60.0
        elif years >= 1:
            return 40.0
        elif years > 0:
            return 20.0
        return 0.0
