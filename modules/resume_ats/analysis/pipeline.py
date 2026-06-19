from __future__ import annotations

from modules.resume_ats.contracts import FeedbackResult, ResumeEntities, RoleJD


ACTION_MAP = {
    "rag": "Add Retrieval-Augmented Generation project experience",
    "langchain": "Showcase LangChain orchestration framework usage",
    "vector database": "Add Pinecone/Qdrant/Weaviate vector DB project",
    "kubernetes": "Highlight Kubernetes deployment experience",
    "mlops": "Include MLOps pipeline and model monitoring work",
    "pytorch": "Add deep learning projects using PyTorch",
    "docker": "Document containerized deployment workflows",
    "aws": "Highlight AWS cloud infrastructure experience",
    "tensorflow": "Add TensorFlow deep learning project experience",
    "transformers": "Showcase Hugging Face Transformers model fine-tuning",
    "fastapi": "Highlight FastAPI microservice development",
    "llm": "Add LLM integration or fine-tuning experience",
    "system design": "Document system design and architecture decisions",
    "data structures": "Strengthen DSA fundamentals in project descriptions",
}

# Severity labels for output
SEVERITY_LABELS = {
    "critical": "CRITICAL — must-have for this role",
    "important": "IMPORTANT — significantly strengthens application",
    "optional": "OPTIONAL — nice-to-have differentiator",
}


class AnalysisPipeline:
    def build(
        self,
        entities: ResumeEntities,
        jd: RoleJD,
        scores: dict,
    ) -> FeedbackResult:
        strengths = self._strengths(entities, jd, scores)
        weaknesses = self._weaknesses(entities, jd, scores)
        critical_missing = self._critical_missing(entities, jd, scores)
        action_items = self._action_items(critical_missing, weaknesses, scores)
        return FeedbackResult(
            strengths=strengths,
            weaknesses=weaknesses,
            critical_missing_keywords=critical_missing,
            action_items=action_items,
        )

    def _strengths(self, entities: ResumeEntities, jd: RoleJD, scores: dict) -> list[str]:
        strengths: list[str] = []
        resume_terms = {t.lower() for t in entities["skills"] + entities["frameworks"] + entities["languages"]}

        for skill in jd.get("required_skills", []):
            if any(skill.lower() in t for t in resume_terms):
                strengths.append(f"Strong {skill} background")

        if entities.get("experience_years", 0) >= 3:
            strengths.append("Solid professional experience")
        if entities.get("research"):
            strengths.append("Strong research profile")
        if len(entities.get("projects", [])) >= 2:
            strengths.append("Strong project portfolio")
        if scores.get("semantic_score", 0) >= 80:
            strengths.append("High semantic alignment with role")
        if scores.get("skill_depth_score", 0) >= 70:
            strengths.append("Skills demonstrated with depth and context")
        if scores.get("experience_score", 0) >= 70:
            strengths.append("Strong experience alignment")
        if scores.get("resume_quality_score", 0) >= 80:
            strengths.append("Well-structured, high-quality resume")

        return list(dict.fromkeys(strengths))[:8]

    def _weaknesses(self, entities: ResumeEntities, jd: RoleJD, scores: dict) -> list[str]:
        resume_terms = {t.lower() for t in (
            entities["skills"] + entities["tools"] + entities["frameworks"]
        )}
        weaknesses: list[str] = []

        # Use severity levels from scores if available
        severity = scores.get("missing_keyword_severity", {})
        for kw in severity.get("critical", []):
            weaknesses.append(f"[CRITICAL] Missing {kw}")
        for kw in severity.get("important", []):
            weaknesses.append(f"[IMPORTANT] Missing {kw}")

        # Fallback: traditional missing skill detection
        if not weaknesses:
            for skill in jd.get("required_skills", []) + jd.get("preferred_skills", []):
                if not any(skill.lower() in t or t in skill.lower() for t in resume_terms):
                    weaknesses.append(f"Missing {skill}")

        if scores.get("resume_quality_score", 0) < 50:
            weaknesses.append("Resume quality needs improvement — add quantified impact metrics")

        return list(dict.fromkeys(weaknesses))[:8]

    def _critical_missing(
        self,
        entities: ResumeEntities,
        jd: RoleJD,
        scores: dict,
    ) -> list[str]:
        # Use severity from orchestrator if available
        severity = scores.get("missing_keyword_severity", {})
        if severity:
            return (
                severity.get("critical", [])
                + severity.get("important", [])[:3]
            )

        # Fallback
        resume_terms = {t.lower() for t in (
            entities["skills"] + entities["tools"] + entities["frameworks"] + entities["languages"]
        )}
        missing: list[str] = []
        for kw in jd.get("keywords", []):
            if not any(kw.lower() in t or t in kw.lower() for t in resume_terms):
                missing.append(kw)
        return missing[:10]

    def _action_items(
        self,
        critical_missing: list[str],
        weaknesses: list[str],
        scores: dict,
    ) -> list[str]:
        items: list[str] = []
        for kw in critical_missing:
            lower = kw.lower()
            matched = False
            for key, action in ACTION_MAP.items():
                if key in lower:
                    items.append(action)
                    matched = True
                    break
            if not matched:
                items.append(f"Add {kw} experience to resume")
        for w in weaknesses[:3]:
            # Strip severity label if present
            skill = w
            for prefix in ("[CRITICAL] Missing ", "[IMPORTANT] Missing ", "Missing "):
                if skill.startswith(prefix):
                    skill = skill[len(prefix):]
                    break
            if f"Highlight {skill}" not in items:
                items.append(f"Highlight {skill} in projects or skills section")
        if scores.get("resume_quality_score", 0) < 60:
            items.append("Improve resume quality: add quantified metrics and action verbs to bullets")
        return list(dict.fromkeys(items))[:10]
