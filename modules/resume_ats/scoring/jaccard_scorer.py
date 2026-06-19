from __future__ import annotations

from modules.resume_ats.contracts import RoleJD
from modules.resume_ats.scoring.utils import clamp_score, jd_to_text, tokenize_words


class JaccardScorer:
    """Exact token overlap via Jaccard similarity on word sets."""

    def score(self, resume_text: str, jd: RoleJD) -> int:
        resume_words = tokenize_words(resume_text)
        jd_words = tokenize_words(jd_to_text(jd))
        if not resume_words and not jd_words:
            return 0
        intersection = len(resume_words & jd_words)
        union = len(resume_words | jd_words)
        return clamp_score((intersection / union) * 100) if union else 0
