from __future__ import annotations

import re
from typing import Any

from modules.resume_ats.contracts import StructuredResume
from modules.resume_ats.entities.extractor import FRAMEWORKS, PROGRAMMING_LANGUAGES, TOOLS
from modules.resume_ats.scoring.utils import resume_to_text


SECTION_ALIASES = {
    "summary": {"summary", "overview", "objective", "profile"},
    "experience": {"experience", "work experience", "professional experience", "employment"},
    "projects": {"projects", "project experience"},
    "education": {"education", "academics", "academic background"},
    "skills": {"skills", "technical skills", "technical competencies", "competencies"},
    "certifications": {"certifications", "licenses"},
    "publications": {"publications", "research", "papers"},
    "achievements": {"achievements", "awards", "accomplishments", "extracurriculars"},
}


def resume_input_to_text_and_proxy(resume_input: str | dict[str, Any]) -> tuple[str, StructuredResume]:
    if isinstance(resume_input, str):
        text = _clean_text(resume_input)
        return text, text_to_structured_resume(text)

    proxy = _normalize_structured_resume(resume_input)
    return resume_to_text(proxy), proxy


def text_to_structured_resume(text: str) -> StructuredResume:
    cleaned = _clean_text(text)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    sections = _split_sections(lines)

    profile = {
        "name": _extract_name(lines),
        "email": _first_match(cleaned, r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "phone": _first_match(cleaned, r"(?:\+?\d[\d\s().-]{7,}\d)"),
        "summary": _extract_summary(lines, sections),
    }

    skills = _extract_skills(cleaned, sections.get("skills", []))

    experience_lines = sections.get("experience", [])
    project_lines = sections.get("projects", [])

    structured: StructuredResume = {
        "profile": {key: value for key, value in profile.items() if value},
        "education": _lines_to_entries(sections.get("education", []), "education"),
        "experience": _lines_to_entries(experience_lines, "experience"),
        "projects": _lines_to_entries(project_lines, "projects"),
        "skills": skills,
        "certifications": sections.get("certifications", []),
        "publications": sections.get("publications", []),
        "achievements": sections.get("achievements", []),
    }

    return _normalize_structured_resume(structured)


def _normalize_structured_resume(resume: dict[str, Any]) -> StructuredResume:
    profile = resume.get("profile") or {}
    return {
        "profile": {
            "name": str(profile.get("name", "") or ""),
            "email": str(profile.get("email", "") or ""),
            "phone": str(profile.get("phone", "") or ""),
            "location": str(profile.get("location", "") or ""),
            "summary": str(profile.get("summary", "") or ""),
            "linkedin": str(profile.get("linkedin", "") or ""),
            "github": str(profile.get("github", "") or ""),
        },
        "education": list(resume.get("education", []) or []),
        "experience": list(resume.get("experience", []) or []),
        "projects": list(resume.get("projects", []) or []),
        "skills": [str(item) for item in (resume.get("skills", []) or []) if str(item).strip()],
        "certifications": [str(item) for item in (resume.get("certifications", []) or []) if str(item).strip()],
        "publications": [str(item) for item in (resume.get("publications", []) or []) if str(item).strip()],
        "achievements": [str(item) for item in (resume.get("achievements", []) or []) if str(item).strip()],
    }


def _clean_text(text: str) -> str:
    return re.sub(r"\r\n?", "\n", text or "").strip()


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in lines:
        normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()
        matched = next(
            (section for section, aliases in SECTION_ALIASES.items() if normalized in aliases),
            None,
        )
        if matched:
            current = matched
            sections.setdefault(current, [])
            continue
        if current:
            sections.setdefault(current, []).append(line)

    return sections


def _extract_name(lines: list[str]) -> str:
    for line in lines[:4]:
        if "@" in line or re.search(r"\d", line):
            continue
        if len(line.split()) >= 2 and len(line) <= 80:
            return line
    return ""


def _extract_summary(lines: list[str], sections: dict[str, list[str]]) -> str:
    summary_lines = sections.get("summary")
    if summary_lines:
        return " ".join(summary_lines[:4])

    collected: list[str] = []
    for line in lines[1:6]:
        normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()
        if any(normalized in aliases for aliases in SECTION_ALIASES.values()):
            break
        if "@" in line:
            continue
        collected.append(line)
    return " ".join(collected[:3])


def _extract_skills(text: str, section_lines: list[str]) -> list[str]:
    raw = " ".join(section_lines) if section_lines else text
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+#./-]*", raw)
    }
    matched = sorted(
        {
            skill
            for skill in PROGRAMMING_LANGUAGES | FRAMEWORKS | TOOLS
            if skill.lower() in tokens or skill.lower() in raw.lower()
        }
    )
    return [skill if any(ch.isupper() for ch in skill) else skill.title() for skill in matched]


def _lines_to_entries(lines: list[str], section: str) -> list[dict[str, Any]]:
    if not lines:
        return []

    if section == "education":
        return [{"degree": line, "school": "", "field": "", "description": line} for line in lines[:5]]

    if section == "projects":
        return [
            {
                "name": _truncate_name(line),
                "description": line,
                "technologies": _extract_inline_technologies(line),
            }
            for line in lines[:8]
        ]

    return [
        {
            "company": "",
            "title": _truncate_name(line),
            "description": line,
            "bullets": [line],
        }
        for line in lines[:10]
    ]


def _extract_inline_technologies(line: str) -> list[str]:
    found: list[str] = []
    lower = line.lower()
    for tech in PROGRAMMING_LANGUAGES | FRAMEWORKS | TOOLS:
        if tech.lower() in lower:
            found.append(tech if any(ch.isupper() for ch in tech) else tech.title())
    return list(dict.fromkeys(found))


def _truncate_name(line: str) -> str:
    parts = re.split(r"[-:|]", line, maxsplit=1)
    return parts[0].strip()[:100]


def _first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return match.group(0).strip() if match else ""
