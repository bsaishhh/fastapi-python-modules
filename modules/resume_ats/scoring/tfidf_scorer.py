from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.resume_ats.scoring.utils import clamp_score


class TfidfScorer:
    """Keyword relevance via TF-IDF cosine similarity (scikit-learn, open source)."""

    def score(self, resume_text: str, jd_text: str) -> int:
        if not resume_text.strip() or not jd_text.strip():
            return 0
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([resume_text, jd_text])
        sim = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
        return clamp_score(sim * 100)
