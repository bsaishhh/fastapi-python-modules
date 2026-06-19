from __future__ import annotations

import json
import logging
from functools import lru_cache

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)

ATS_SYSTEM_PROMPT = """
You are an expert technical recruiter and domain classifier.

Your task is to evaluate a candidate's resume against a specific benchmark job description (JD) role.

You will receive:
1. Role (e.g., AI_ML_ENGINEER, SOFTWARE_ENGINEER, DATA_ENGINEER, etc.)
2. Candidate Resume Text (with extracted technical entities)
3. Master JD Text (required skills, preferred skills, tools, frameworks, responsibilities, keywords)
4. Valid Subroles for the evaluated role cluster

CRITICAL EVALUATION RULES:

1. Evidence-Based Only
   - Do NOT hallucinate experience, skills, or projects not explicitly stated in the resume.
   - Consider common abbreviations and synonyms (e.g., "ML" = "Machine Learning", "HF" = "Hugging Face", "PyTorch transformers" ≡ "Transformers").
   - Extract year counts, project impact, and leadership indicators from the resume.

2. Precise Skill Matching
   - Match resume skills to JD required skills with HIGH precision.
   - Differentiate between "exposure" (mentioned once in project) vs "proficiency" (multiple projects/years of experience).
   - Flag missing critical skills that are CORE to the role.
   - Do NOT penalize for missing optional/preferred skills unless they indicate a significant domain gap.

3. Domain Depth Assessment
   - Evaluate whether the candidate demonstrates depth in the required domain.
   - Consider: years of experience in domain, complexity of projects, breadth of skill coverage.
   - Example: 5 years Python experience with ML projects = stronger match than 1 year Python with random projects.

4. Subrole Alignment
   - Identify 1-2 specific subroles where the candidate excels based on their experience trajectory.
   - Example: Resume shows "Backend API development + DevOps automation" → "Backend Engineer" or "DevOps Engineer".
   - Do NOT suggest subroles unsupported by evidence.

5. Critical vs. Important Keywords
   - CRITICAL: Must-have skills for role success (e.g., Python for AI_ML_ENGINEER).
   - IMPORTANT: High-value optional skills (e.g., Docker for AI_ML_ENGINEER).
   - OPTIONAL: Nice-to-have (e.g., Kubernetes for AI_ML_ENGINEER).
   - Only flag CRITICAL missing keywords in output.

OUTPUT JSON SCHEMA:
{
  "best_fit_subroles": ["Subrole1", "Subrole2"],
  "strengths": ["Evidence-based strength 1", "Evidence-based strength 2", ...],
  "areas_of_improvement": ["Gap or weakness 1", ...],
  "critical_missing_keywords": ["Critical keyword 1", ...],
  "action_plan": [
    {
      "priority": "High|Medium|Low",
      "action": "Specific improvement to make",
      "impact": "Why this helps ATS or role alignment",
      "example": "Concrete example of how to implement it"
    }
  ]
}

GENERATION GUIDELINES:

Strengths (3-8):
- Start with most important JD requirement evidence.
- Include quantifiable impact (e.g., "3+ years Python with 5+ production ML projects").
- Highlight cross-domain strengths if applicable (e.g., "ML expertise + Cloud deployment").

Areas of Improvement (3-8):
- Focus on significant gaps in required skills.
- Avoid penalizing for lacking optional/preferred skills.
- Be specific: "No mentioned experience with Kubernetes" vs. "Weak DevOps background".

Critical Missing Keywords (list only critical):
- Return role-critical keywords NOT present in resume or its synonyms.
- Filter out items with clear synonyms or related experience.

Action Plan (2-5):
- Convert each area of improvement into a specific, achievable recommendation.
- Use `priority` values only from: `High`, `Medium`, `Low`.
- Keep `action`, `impact`, and `example` concise and concrete.
- Example: "No Kubernetes experience" → {"priority":"High","action":"Build a Kubernetes deployment project","impact":"Shows production deployment readiness","example":"Deploy a FastAPI service to a managed Kubernetes cluster"}.

Subrole Guidelines:
- Match candidate experience to valid subroles provided in the JD.
- Example for SOFTWARE_ENGINEERING (subroles: Backend, Frontend, Full Stack, Mobile, Platform, DevOps, SRE):
  * Resume shows "5 years backend APIs + database design" → "Backend Engineer" (primary), "Platform Engineer" (secondary).
- If experience doesn't clearly map to any subrole, suggest "Generalist".

DOMAIN-SPECIFIC EVALUATION CRITERIA:

AI_ML_ENGINEER: Python proficiency (5+ years or equivalent depth), PyTorch/TensorFlow, LLM/NLP/CV experience, deployment (Docker/Kubernetes/FastAPI).
SOFTWARE_ENGINEER: Language proficiency (Java/Python/C++), System Design, Microservices, Database design, CI/CD, Cloud (AWS/Azure/GCP).
DATA_ENGINEER: Big Data tools (Spark/Hadoop), ETL/pipeline design, Data warehousing, SQL optimization, Cloud data platforms.
QUANT_FINANCE: Mathematical modeling, Statistics/Probability, C++/Python, Financial markets knowledge, Algorithm design.
CONSULTING_STRATEGY: Problem-solving frameworks, Business analysis, Market research, Communication, Project management.
MECHANICAL_ENGINEERING: CAD (SolidWorks/AutoCAD), Simulation (FEA/CFD), Manufacturing processes, Design principles.
ROBOTICS_AUTONOMOUS: Robotics frameworks (ROS), Computer Vision, SLAM, Control systems, C++/Python, Real-time systems.

Evaluate the resume STRICTLY against the supplied role only. Assess how well the candidate fits THIS specific role, not how they might fit other roles.
"""


class ATSAnalysis(BaseModel):
    best_fit_subroles: list[str] = Field(..., description="Best-fit specific subroles within the selected cluster")
    strengths: list[str] = Field(..., description="Strengths identified from resume")
    areas_of_improvement: list[str] = Field(..., description="Most important gaps identified against role benchmark")
    critical_missing_keywords: list[str] = Field(..., description="Most important missing keywords")
    action_plan: list[dict[str, str]] = Field(..., description="Actionable resume improvements with priority, impact, and example")


@lru_cache(maxsize=1)
def _get_instructor_client():
    if not settings.openrouter_api_key:
        return None
    client = OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )
    return instructor.patch(client)


class LLMAnalyzer:
    def analyze(self, role: str, resume_text: str, entities: dict, master_jd: dict) -> ATSAnalysis:
        from modules.resume_ats.scoring.utils import jd_to_text
        client = _get_instructor_client()
        if client is None:
            logger.warning("OPENROUTER_API_KEY not set; returning fallback analysis")
            return self._fallback_analysis(resume_text, entities, master_jd)
        
        # Inject extracted entities to help LLM not miss embedded keywords
        entity_summary = (
            f"\n\n--- EXTRACTED TECHNICAL ENTITIES ---\n"
            f"Languages: {', '.join(entities.get('languages', []))}\n"
            f"Frameworks: {', '.join(entities.get('frameworks', []))}\n"
            f"Tools: {', '.join(entities.get('tools', []))}\n"
        )
        resume_text += entity_summary
        
        jd_text = jd_to_text(master_jd)
        
        # Provide valid subroles so the LLM doesn't guess "Generalist"
        valid_subroles = master_jd.get("sub_roles", [])
        if valid_subroles:
            jd_text += f"\n\n--- VALID SUBROLES ---\n{', '.join(valid_subroles)}"

        messages = [
            {"role": "system", "content": ATS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Role: {role}\n\nCandidate Resume Text:\n{resume_text}\n\nMaster JD Text:\n{jd_text}",
            },
        ]

        response: ATSAnalysis = client.chat.completions.create(
            model=settings.openrouter_model,
            response_model=ATSAnalysis,
            messages=messages,
        )
        return response

    def _fallback_analysis(self, resume_text: str, entities: dict, master_jd: dict) -> ATSAnalysis:
        skills_raw = (
            entities.get("languages", []) +
            entities.get("frameworks", []) +
            entities.get("tools", [])
        )
        resume_skills = {s.lower() for s in skills_raw}
        resume_lower = resume_text.lower()
        required = [s for s in master_jd.get("required_skills", [])]
        keywords = [k for k in master_jd.get("keywords", [])]

        strengths = [
            f"Strong {s} background"
            for s in required
            if s.lower() in resume_skills or s.lower() in resume_lower
        ][:5]
        if not strengths:
            strengths = ["Resume provides baseline profile for review"]

        missing = [
            kw for kw in keywords
            if not any(kw.lower() in skill for skill in resume_skills) and kw.lower() not in resume_lower
        ][:8]

        areas_of_improvement = [f"Missing {kw}" for kw in missing[:5]]
        action_plan = [
            {
                "priority": "High" if index == 0 else "Medium",
                "action": f"Add {kw} experience to resume",
                "impact": f"Improves alignment for {kw}-related role requirements",
                "example": f"Add a project or bullet that demonstrates {kw}",
            }
            for index, kw in enumerate(missing[:5])
        ]

        return ATSAnalysis(
            best_fit_subroles=["Generalist"],
            strengths=strengths,
            areas_of_improvement=areas_of_improvement,
            critical_missing_keywords=missing[:5],
            action_plan=action_plan,
        )
