"""Resume quality scorer — separate from ATS match score.

Evaluates resume quality on:
  - quantified_impact: bullets with metrics/numbers
  - action_verbs: strong action verbs at bullet start
  - section_completeness: presence of key sections
  - skill_diversity: breadth of skills listed
  - bullet_quality: non-trivial bullet length and structure
  - grammar_signals: basic grammar/professionalism indicators
"""

from __future__ import annotations

import re

from modules.resume_ats.contracts import ResumeEntities, StructuredResume
from modules.resume_ats.scoring.utils import clamp_score

# Strong action verbs (past tense preferred for experience)
STRONG_ACTION_VERBS = {
    # Leadership
    "led", "spearheaded", "directed", "orchestrated", "championed", "pioneered",
    "founded", "established", "launched", "initiated",
    # Technical
    "architected", "designed", "built", "developed", "implemented", "deployed",
    "engineered", "programmed", "coded", "automated", "optimized", "refactored",
    # Achievement
    "achieved", "delivered", "exceeded", "accelerated", "reduced", "improved",
    "increased", "scaled", "migrated", "transformed", "streamlined",
    # Collaboration
    "collaborated", "mentored", "coordinated", "facilitated", "partnered",
    # Research/Analysis
    "analyzed", "investigated", "researched", "evaluated", "benchmarked",
}

# Weak/passive verbs to penalize
WEAK_VERBS = {
    "helped", "assisted", "worked", "was", "were", "responsible for",
    "participated", "contributed", "involved", "supported",
}

# Quantified impact patterns
QUANT_PATTERNS = [
    re.compile(r"\d+\s*%"),                          # 40%
    re.compile(r"\$\s*[\d,.]+"),                      # $1.5M
    re.compile(r"\d+\s*[kKmMbB]\+?\s*(?:user|record|customer|event|request)"),  # 1M users
    re.compile(r"\d+\s*x\s*(?:faster|speedup|improvement|increase|decrease)"),  # 3x faster
    re.compile(r"(?:top|ranked)\s*\d+"),              # top 5
    re.compile(r"\d+\s*(?:team|member|person|people)"),  # 10-person team
]


class ResumeQualityScorer:
    """Evaluates resume quality independent of JD match."""

    def score(self, resume: StructuredResume, entities: ResumeEntities) -> int:
        sections = self._section_completeness(resume)
        impact = self._quantified_impact(resume)
        verbs = self._action_verb_density(resume)
        bullets = self._bullet_quality(resume)
        skills = self._skill_diversity(resume, entities)
        grammar = self._grammar_signals(resume)

        combined = (
            0.20 * sections
            + 0.25 * impact
            + 0.15 * verbs
            + 0.15 * bullets
            + 0.10 * skills
            + 0.15 * grammar
        )
        return clamp_score(combined)

    def _section_completeness(self, resume: StructuredResume) -> float:
        """Score presence of key resume sections."""
        score = 0.0
        max_score = 100.0

        # Required sections
        if resume.get("profile", {}).get("name"):
            score += 15
        if resume.get("profile", {}).get("email"):
            score += 10
        if resume.get("profile", {}).get("summary"):
            score += 15
        if resume.get("education"):
            score += 15
        if resume.get("experience"):
            score += 20
        if resume.get("projects"):
            score += 10
        if resume.get("skills"):
            score += 10

        # Bonus for optional sections
        if resume.get("certifications"):
            score += 2.5
        if resume.get("publications"):
            score += 2.5

        return min(score, max_score)

    def _quantified_impact(self, resume: StructuredResume) -> float:
        """Percentage of experience bullets with quantified metrics."""
        bullets = self._all_bullets(resume)
        if not bullets:
            return 0.0

        hits = sum(1 for b in bullets if any(p.search(b) for p in QUANT_PATTERNS))
        ratio = hits / len(bullets)

        # 50%+ quantified bullets = perfect score
        return min(ratio * 200, 100.0)

    def _action_verb_density(self, resume: StructuredResume) -> float:
        """Percentage of bullets starting with strong action verbs."""
        bullets = self._all_bullets(resume)
        if not bullets:
            return 0.0

        strong_hits = 0
        weak_hits = 0
        for bullet in bullets:
            first_word = bullet.split()[0].lower().rstrip(".,;:") if bullet.split() else ""
            if first_word in STRONG_ACTION_VERBS:
                strong_hits += 1
            # Check for weak openings
            first_two = " ".join(bullet.split()[:2]).lower()
            if any(wv in first_two for wv in WEAK_VERBS):
                weak_hits += 1

        strong_ratio = strong_hits / len(bullets)
        weak_penalty = weak_hits / len(bullets)

        return max(0, min((strong_ratio * 100 - weak_penalty * 50) * 2, 100.0))

    def _bullet_quality(self, resume: StructuredResume) -> float:
        """Average bullet length quality (not too short, not too long)."""
        bullets = self._all_bullets(resume)
        if not bullets:
            return 0.0

        scores = []
        for bullet in bullets:
            word_count = len(bullet.split())
            if word_count < 3:
                scores.append(0.0)     # Too short — likely fragment
            elif word_count < 6:
                scores.append(30.0)    # Somewhat thin
            elif word_count <= 25:
                scores.append(100.0)   # Ideal length
            elif word_count <= 35:
                scores.append(70.0)    # A bit long
            else:
                scores.append(40.0)    # Too long / paragraph

        return sum(scores) / len(scores) if scores else 0.0

    def _skill_diversity(self, resume: StructuredResume, entities: ResumeEntities) -> float:
        """Breadth of skills across categories."""
        skill_count = len(resume.get("skills", []))
        categories_present = sum(1 for cat in (
            entities.get("languages", []),
            entities.get("frameworks", []),
            entities.get("tools", []),
        ) if cat)

        # Score: more skills + more categories = higher
        count_score = min(skill_count * 5, 60)
        category_score = categories_present * 13.3
        return min(count_score + category_score, 100.0)

    def _grammar_signals(self, resume: StructuredResume) -> float:
        """Basic grammar and professionalism signals."""
        bullets = self._all_bullets(resume)
        if not bullets:
            return 50.0  # No bullets — neutral

        score = 60.0  # Base score

        # Bonus for consistent punctuation
        ending_punct = sum(1 for b in bullets if b.rstrip().endswith((".", ";")))
        if ending_punct / len(bullets) > 0.5:
            score += 10

        # Bonus for no ALL CAPS bullets (shouting)
        all_caps = sum(1 for b in bullets if b.isupper() and len(b) > 5)
        if all_caps / len(bullets) < 0.1:
            score += 10

        # Bonus for summary section with reasonable length
        summary = resume.get("profile", {}).get("summary", "")
        if summary:
            word_count = len(summary.split())
            if 20 <= word_count <= 80:
                score += 20
            elif 10 <= word_count <= 120:
                score += 10

        return min(score, 100.0)

    def _all_bullets(self, resume: StructuredResume) -> list[str]:
        """Collect all bullet points from experience and projects."""
        bullets: list[str] = []
        for exp in resume.get("experience", []):
            bullets.extend(exp.get("bullets", []))
            if exp.get("description"):
                bullets.append(exp["description"])
        for proj in resume.get("projects", []):
            if proj.get("description"):
                bullets.append(proj["description"])
        return [b for b in bullets if b.strip()]
