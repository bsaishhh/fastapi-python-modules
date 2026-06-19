from __future__ import annotations

import re

from modules.resume_ats.contracts import RoleJD, StructuredResume

# ─────────────────────────────────────────────────────────────────────────────
# ATS v2 tournament weights
# Semantic (section-weighted) + BM25 + Keyword + Exact + Domain + Skill Depth
# ─────────────────────────────────────────────────────────────────────────────
SCORING_WEIGHTS = {
    "semantic":       0.40,
    "bm25":           0.15,
    "keyword":        0.10,
    "exact_keyword":  0.05,
    "domain":         0.20,
    "experience":     0.10,
    "skill_depth":    0.05,
}


def clamp_score(value: float) -> int:
    return int(min(100, max(0, round(value))))


# ─────────────────────────────────────────────────────────────────────────────
# resume_to_text: section-aware plain-text builder (no raw JSON)
# ─────────────────────────────────────────────────────────────────────────────

def resume_to_text(resume: StructuredResume) -> str:
    """Build clean, section-labelled text for embedding/BM25 scorers."""
    sections: list[str] = []

    # Summary
    summary = resume.get("profile", {}).get("summary", "")
    if summary:
        sections.append(f"Summary:\n{summary}")

    # Experience
    exp_parts: list[str] = []
    for entry in resume.get("experience", []):
        lines = []
        title = entry.get("title", "")
        company = entry.get("company", "")
        if title or company:
            lines.append(f"{title} at {company}".strip())
        for bullet in entry.get("bullets", []):
            lines.append(bullet)
        if entry.get("description"):
            lines.append(entry["description"])
        if lines:
            exp_parts.append("\n".join(lines))
    if exp_parts:
        sections.append("Experience:\n" + "\n\n".join(exp_parts))

    # Projects
    proj_parts: list[str] = []
    for entry in resume.get("projects", []):
        lines = []
        if entry.get("name"):
            lines.append(entry["name"])
        if entry.get("description"):
            lines.append(entry["description"])
        techs = entry.get("technologies", [])
        if techs:
            lines.append("Technologies: " + ", ".join(techs))
        if lines:
            proj_parts.append("\n".join(lines))
    if proj_parts:
        sections.append("Projects:\n" + "\n\n".join(proj_parts))

    # Technical Skills / Tools
    skills = resume.get("skills", [])
    languages = resume.get("languages", [])
    frameworks = resume.get("frameworks", [])
    tools = resume.get("tools", [])
    
    tech_parts = []
    if skills:
        tech_parts.append("Skills: " + ", ".join(skills))
    if languages:
        tech_parts.append("Languages: " + ", ".join(languages))
    if frameworks:
        tech_parts.append("Frameworks: " + ", ".join(frameworks))
    if tools:
        tech_parts.append("Tools: " + ", ".join(tools))
        
    if tech_parts:
        sections.append("Technical Competencies:\n" + "\n".join(tech_parts))

    # Education
    edu_parts: list[str] = []
    for entry in resume.get("education", []):
        parts = []
        if entry.get("degree"):
            parts.append(entry["degree"])
        if entry.get("school"):
            parts.append(entry["school"])
        if entry.get("field"):
            parts.append(entry["field"])
        if parts:
            edu_parts.append(", ".join(parts))
    if edu_parts:
        sections.append("Education:\n" + "\n".join(edu_parts))

    # Certifications
    certs = resume.get("certifications", [])
    if certs:
        sections.append("Certifications:\n" + ", ".join(certs))

    # Publications
    pubs = resume.get("publications", [])
    if pubs:
        sections.append("Publications:\n" + "\n".join(pubs))

    # Achievements
    achs = resume.get("achievements", [])
    if achs:
        sections.append("Achievements:\n" + "\n".join(achs))

    return "\n\n".join(sections)


# ─────────────────────────────────────────────────────────────────────────────
# Per-section text extractors (for section-weighted semantic scoring)
# ─────────────────────────────────────────────────────────────────────────────

def resume_experience_text(resume: StructuredResume) -> str:
    """Experience section text for section-weighted semantic matching."""
    parts: list[str] = []
    for entry in resume.get("experience", []):
        title = entry.get("title", "")
        company = entry.get("company", "")
        if title or company:
            parts.append(f"{title} at {company}")
        parts.extend(entry.get("bullets", []))
        if entry.get("description"):
            parts.append(entry["description"])
    return "\n".join(parts)


def resume_project_text(resume: StructuredResume) -> str:
    """Projects section text for section-weighted semantic matching."""
    parts: list[str] = []
    for entry in resume.get("projects", []):
        if entry.get("name"):
            parts.append(entry["name"])
        if entry.get("description"):
            parts.append(entry["description"])
        techs = entry.get("technologies", [])
        if techs:
            parts.append("Technologies: " + ", ".join(techs))
    return "\n".join(parts)


def resume_skills_text(resume: StructuredResume) -> str:
    """Skills section text."""
    return ", ".join(resume.get("skills", []))


def resume_education_text(resume: StructuredResume) -> str:
    """Education section text."""
    parts: list[str] = []
    for entry in resume.get("education", []):
        for key in ("degree", "school", "field", "description"):
            val = entry.get(key)
            if val:
                parts.append(str(val))
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# JD text builders
# ─────────────────────────────────────────────────────────────────────────────

def jd_to_text(jd: RoleJD) -> str:
    parts = (
        jd.get("required_skills", [])
        + jd.get("preferred_skills", [])
        + jd.get("tools", [])
        + jd.get("frameworks", [])
        + jd.get("responsibilities", [])
        + jd.get("keywords", [])
    )
    return " ".join(str(p) for p in parts if p)


def jd_responsibilities_text(jd: RoleJD) -> str:
    """JD responsibilities text (for experience/project matching)."""
    return " ".join(jd.get("responsibilities", []))


def jd_skills_text(jd: RoleJD) -> str:
    """JD skills text (for skills matching)."""
    parts = (
        jd.get("required_skills", [])
        + jd.get("preferred_skills", [])
        + jd.get("tools", [])
        + jd.get("frameworks", [])
    )
    return " ".join(str(p) for p in parts if p)


def jd_degree_requirements_text(jd: RoleJD) -> str:
    """JD education/degree requirements text (if present)."""
    # Most JDs embed degree info in keywords or responsibilities
    degree_keywords = [
        k for k in jd.get("keywords", [])
        if any(d in k.lower() for d in ("bachelor", "master", "phd", "degree", "b.tech", "m.tech"))
    ]
    return " ".join(degree_keywords)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z0-9+#./-]*", text.lower())}


def entity_terms(resume_text: str, entities) -> set[str]:
    terms = {
        t.lower()
        for t in entities.get("skills", [])
        + entities.get("tools", [])
        + entities.get("frameworks", [])
        + entities.get("languages", [])
    }
    terms.update(tokenize_words(resume_text))
    return terms
