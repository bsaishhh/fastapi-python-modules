from __future__ import annotations

import re
from functools import lru_cache

from modules.resume_ats.contracts import ResumeEntities, StructuredResume

PROGRAMMING_LANGUAGES = {
    "python", "java", "javascript", "typescript", "c++", "c", "go", "rust", "ruby", "scala", "r", "matlab", "sql",
}
FRAMEWORKS = {
    "pytorch", "tensorflow", "keras", "fastapi", "django", "flask", "react", "angular", "vue", "spring", "langchain",
    "transformers", "scikit-learn", "pandas", "numpy", "spark", "hadoop", "airflow", "dbt",
}
TOOLS = {
    "docker", "kubernetes", "aws", "gcp", "azure", "git", "jenkins", "terraform", "redis", "postgresql", "mongodb",
    "pinecone", "qdrant", "weaviate", "mlflow", "kubeflow", "grafana", "prometheus",
}
DEGREE_PATTERNS = [
    r"\bb\.?\s*tech\b", r"\bm\.?\s*tech\b", r"\bb\.?\s*e\b", r"\bm\.?\s*e\b",
    r"\bbachelor\b", r"\bmaster\b", r"\bph\.?d\b", r"\bmba\b", r"\bbsc\b", r"\bmsc\b",
]


class EntityExtractor:
    """Extract normalized entities from structured resume JSON."""

    def extract(self, resume: StructuredResume) -> ResumeEntities:
        text_corpus = self._build_corpus(resume)
        tokens = self._tokenize(text_corpus)
        token_set = {t.lower() for t in tokens}

        skills = list(dict.fromkeys(resume.get("skills", [])))
        languages = sorted({t for t in token_set if t in PROGRAMMING_LANGUAGES})
        frameworks = sorted({t for t in token_set if t in FRAMEWORKS})
        tools = sorted({t for t in token_set if t in TOOLS})

        for item in skills:
            lower = item.lower()
            if lower in PROGRAMMING_LANGUAGES and lower not in languages:
                languages.append(lower)
            if lower in FRAMEWORKS and lower not in frameworks:
                frameworks.append(lower)
            if lower in TOOLS and lower not in tools:
                tools.append(lower)

        degrees = self._extract_degrees(resume)
        companies = [exp.get("company", "") for exp in resume.get("experience", []) if exp.get("company")]
        projects = [p.get("name", "") for p in resume.get("projects", []) if p.get("name")]
        research = list(resume.get("publications", []))
        certifications = list(resume.get("certifications", []))
        experience_years = self._estimate_experience_years(resume)

        all_skills = list(dict.fromkeys(skills + languages + frameworks + tools))

        return ResumeEntities(
            skills=all_skills,
            tools=tools,
            frameworks=frameworks,
            languages=languages,
            degrees=degrees,
            companies=companies,
            projects=projects,
            research=research,
            certifications=certifications,
            experience_years=experience_years,
        )

    def _build_corpus(self, resume: StructuredResume) -> str:
        parts: list[str] = []
        profile = resume.get("profile", {})
        for key in ("name", "summary", "email"):
            if profile.get(key):
                parts.append(str(profile[key]))
        for section in ("education", "experience", "projects"):
            for entry in resume.get(section, []):
                parts.extend(str(v) for v in entry.values() if v)
        parts.extend(resume.get("skills", []))
        parts.extend(resume.get("certifications", []))
        parts.extend(resume.get("publications", []))
        parts.extend(resume.get("achievements", []))
        return " ".join(parts)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z][A-Za-z0-9+#./-]*", text)

    def _extract_degrees(self, resume: StructuredResume) -> list[str]:
        degrees: list[str] = []
        for edu in resume.get("education", []):
            if edu.get("degree"):
                degrees.append(edu["degree"])
        corpus = self._build_corpus(resume).lower()
        for pattern in DEGREE_PATTERNS:
            if re.search(pattern, corpus, re.IGNORECASE):
                degrees.append(pattern.replace(r"\b", "").replace("\\", ""))
        return list(dict.fromkeys(degrees))

    def _estimate_experience_years(self, resume: StructuredResume) -> float:
        total_months = 0
        date_range = re.compile(
            r"(\d{4})\s*[-–/]\s*(\d{4}|Present|Current)",
            re.IGNORECASE,
        )
        for exp in resume.get("experience", []):
            text = f"{exp.get('start_date', '')} {exp.get('end_date', '')} {exp.get('title', '')}"
            match = date_range.search(text)
            if match:
                start = int(match.group(1))
                end_str = match.group(2)
                end = 2026 if end_str.lower() in ("present", "current") else int(end_str)
                total_months += max(0, (end - start) * 12)
        if total_months == 0 and resume.get("experience"):
            return float(len(resume["experience"]) * 1.5)
        return round(total_months / 12, 1)
