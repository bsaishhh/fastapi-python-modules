"""BM25 scorer — replaces Jaccard with industry-standard retrieval scoring.

Uses rank_bm25 for phrase-aware keyword matching with term frequency handling.
"""

from __future__ import annotations

import re
import logging

from modules.resume_ats.contracts import RoleJD
from modules.resume_ats.scoring.utils import clamp_score, jd_to_text

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"[a-z][a-z0-9+#./-]*", text.lower())


class BM25Scorer:
    """BM25 retrieval score — better phrase matching and frequency handling than Jaccard."""

    def score(self, resume_text: str, jd: RoleJD) -> int:
        jd_text = jd_to_text(jd)
        if not resume_text.strip() or not jd_text.strip():
            return 0

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed; BM25 score returns 0")
            return 0

        resume_tokens = _tokenize(resume_text)
        jd_tokens = _tokenize(jd_text)

        if not resume_tokens or not jd_tokens:
            return 0

        # Build BM25 index from resume, query with JD terms
        corpus = [resume_tokens]
        bm25 = BM25Okapi(corpus)
        score = float(bm25.get_scores(jd_tokens)[0])

        # Normalize: BM25 scores are unbounded; cap at reasonable max
        # Typical BM25 scores for resume/JD matching range 0–30
        normalized = min(score / 20.0, 1.0) * 100
        return clamp_score(normalized)
