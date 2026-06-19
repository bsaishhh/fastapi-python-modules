from __future__ import annotations

import re

from modules.resume_ats.extraction.models.text_block import Line

# Primary keywords – every resume has these
SECTION_TITLE_PRIMARY = ["experience", "education", "project", "skill"]

# Secondary / fallback keywords
SECTION_TITLE_SECONDARY = [
    "job", "course", "extracurricular", "objective", "summary",
    "award", "honor", "certification", "publication", "achievement",
    "volunteer", "activity", "interest", "language", "reference",
]

_ALL_KEYWORDS = SECTION_TITLE_PRIMARY + SECTION_TITLE_SECONDARY

# Maps a normalised section-title string → canonical section key
_KEYWORD_TO_SECTION: dict[str, str] = {
    "experience": "experience",
    "employment": "experience",
    "work history": "experience",
    "work experience": "experience",
    "professional experience": "experience",
    "education": "education",
    "academic": "education",
    "qualification": "education",
    "project": "projects",
    "projects": "projects",
    "personal projects": "projects",
    "portfolio": "projects",
    "skill": "skills",
    "skills": "skills",
    "technical skills": "skills",
    "core competencies": "skills",
    "technologies": "skills",
    "certification": "certifications",
    "certifications": "certifications",
    "certificates": "certifications",
    "license": "certifications",
    "licenses": "certifications",
    "publication": "publications",
    "publications": "publications",
    "papers": "publications",
    "research": "publications",
    "achievement": "achievements",
    "achievements": "achievements",
    "awards": "achievements",
    "honors": "achievements",
    "accomplishments": "achievements",
    "summary": "summary",
    "objective": "objective",
    "profile": "profile",
    "about": "profile",
    "volunteer": "volunteer",
    "activities": "activities",
}


class SectionDetector:
    """Stage 3: Detect resume sections.

    Primary heuristic (OpenResume):
        line is bold AND all-uppercase AND ≤ 4 words  →  section title

    Fallback heuristic:
        ≤ 3 words, starts with capital, letters/spaces/ampersands only,
        AND contains a known section keyword  →  section title
    """

    def detect_sections(self, lines: list[Line]) -> dict[str, list[Line]]:
        sections: dict[str, list[Line]] = {}
        current_section = "profile"
        sections[current_section] = []

        for i, line in enumerate(lines):
            new_section = self._classify_header(line, i)
            if new_section is not None:
                current_section = new_section
                sections.setdefault(current_section, [])
            else:
                sections.setdefault(current_section, []).append(line)

        return sections

    # ------------------------------------------------------------------

    @staticmethod
    def _classify_header(line: Line, line_idx: int) -> str | None:
        text = line.text.strip()
        if not text:
            return None

        words = text.split()
        # First two lines are never section headers (they're the name)
        if line_idx < 2:
            return None
        # Too many words → not a header
        if len(words) > 4:
            return None

        # --- Primary: bold + all uppercase ---
        if line.is_bold:
            alpha = "".join(c for c in text if c.isalpha())
            if len(alpha) >= 2 and alpha.isupper():
                return _map_section_name(text)

        # --- Fallback: keyword match with strict word/syntax constraints ---
        clean = re.sub(r"[^a-zA-Z\s&]", "", text).strip()
        if not clean:
            return None

        word_count = len([w for w in clean.split() if w != "&"])
        if word_count > 3:
            return None

        if not clean[0].isupper():
            return None

        # Must be letters, spaces, ampersands only
        if not re.match(r"^[A-Za-z\s&]+$", clean):
            return None

        lower = clean.lower()
        if any(kw in lower for kw in _ALL_KEYWORDS):
            return _map_section_name(text)

        return None


def _map_section_name(text: str) -> str:
    normalised = re.sub(r"[^a-zA-Z\s]", "", text).strip().lower()
    for keyword, section in _KEYWORD_TO_SECTION.items():
        if keyword in normalised:
            return section
    # Return as-is so we still create a bucket for unknown headers
    return normalised or text.strip().lower()
