from __future__ import annotations

import re
from collections import Counter
from typing import Any

from modules.resume_ats.extraction.models.text_block import Line, TextBlock

# ──────────────────────────────────────────────────────────────────────
# Common patterns
# ──────────────────────────────────────────────────────────────────────

EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
PHONE_RE = re.compile(r"\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}")
URL_RE = re.compile(r"\S+\.[a-z]+/\S+")
URL_HTTP_RE = re.compile(r"https?://\S+\.\S+")
URL_WWW_RE = re.compile(r"www\.\S+\.\S+")
CITY_STATE_RE = re.compile(r"[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}")
GPA_RE = re.compile(r"[0-4]\.\d{1,2}")
YEAR_RE = re.compile(r"(?:19|20)\d{2}")
NAME_RE = re.compile(r"^[a-zA-Z\s.]+$")

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
SEASONS = ["Summer", "Fall", "Spring", "Winter"]

SCHOOLS = ["College", "University", "Institute", "School", "Academy", "BASIS", "Magnet"]
DEGREES = ["Associate", "Bachelor", "Master", "PhD", "Ph.", "B.Tech", "M.Tech",
           "B.E", "M.E", "MBA", "BSc", "MSc", "B.S.", "M.S.", "AA", "BS", "MS"]

JOB_TITLES = [
    "Accountant", "Administrator", "Advisor", "Agent", "Analyst", "Apprentice",
    "Architect", "Assistant", "Associate", "Auditor", "Bartender", "Biologist",
    "Bookkeeper", "Buyer", "Carpenter", "Cashier", "CEO", "Clerk", "Co-op",
    "Co-Founder", "Consultant", "Coordinator", "CTO", "Developer", "Designer",
    "Director", "Driver", "Editor", "Electrician", "Engineer", "Extern",
    "Founder", "Freelancer", "Head", "Intern", "Janitor", "Journalist",
    "Laborer", "Lawyer", "Lead", "Manager", "Mechanic", "Member", "Nurse",
    "Officer", "Operator", "Operation", "Photographer", "President", "Producer",
    "Recruiter", "Representative", "Researcher", "Sales", "Server", "Scientist",
    "Specialist", "Supervisor", "Teacher", "Technician", "Trader", "Trainee",
    "Treasurer", "Tutor", "Vice", "VP", "Volunteer", "Webmaster", "Worker",
]

# Bullet-point characters (same list as OpenResume)
BULLET_POINTS = {"⋅", "∙", "🞄", "•", "⦁", "⚫", "●", "⬤", "⚬", "○"}


# ──────────────────────────────────────────────────────────────────────
# Low-level feature predicates  (each takes a TextBlock, returns bool)
# ──────────────────────────────────────────────────────────────────────

def _is_bold(b: TextBlock) -> bool:
    return b.bold


def _has_letter(b: TextBlock) -> bool:
    return bool(re.search(r"[a-zA-Z]", b.text))


def _has_number(b: TextBlock) -> bool:
    return bool(re.search(r"[0-9]", b.text))


def _has_comma(b: TextBlock) -> bool:
    return "," in b.text


def _has_at(b: TextBlock) -> bool:
    return "@" in b.text


def _has_paren(b: TextBlock) -> bool:
    return bool(re.search(r"\([0-9]+\)", b.text))


def _has_slash(b: TextBlock) -> bool:
    return "/" in b.text


def _all_upper(b: TextBlock) -> bool:
    alpha = "".join(c for c in b.text if c.isalpha())
    return len(alpha) >= 2 and alpha.isupper()


def _only_letters_spaces_period(b: TextBlock) -> bool:
    return bool(NAME_RE.match(b.text))


def _only_letters_spaces_amp(b: TextBlock) -> bool:
    return bool(re.match(r"^[A-Za-z\s&]+$", b.text))


def _has_4plus_words(b: TextBlock) -> bool:
    return len(b.text.split()) >= 4


def _has_5plus_words(b: TextBlock) -> bool:
    return len(b.text.split()) >= 5


def _has_8plus_words(b: TextBlock) -> bool:
    return len(b.text.split()) >= 8


def _match_email(b: TextBlock) -> str | None:
    m = EMAIL_RE.search(b.text)
    return m.group(0) if m else None


def _match_phone(b: TextBlock) -> str | None:
    m = PHONE_RE.search(b.text)
    return m.group(0) if m else None


def _match_url(b: TextBlock) -> str | None:
    m = URL_RE.search(b.text)
    return m.group(0) if m else None


def _match_url_http(b: TextBlock) -> str | None:
    m = URL_HTTP_RE.search(b.text)
    return m.group(0) if m else None


def _match_url_www(b: TextBlock) -> str | None:
    m = URL_WWW_RE.search(b.text)
    return m.group(0) if m else None


def _match_city_state(b: TextBlock) -> str | None:
    m = CITY_STATE_RE.search(b.text)
    return m.group(0) if m else None


def _match_gpa(b: TextBlock) -> str | None:
    m = GPA_RE.search(b.text)
    return m.group(0) if m else None


def _match_grade(b: TextBlock) -> str | None:
    try:
        val = float(b.text.strip())
        if val <= 110:
            return str(val)
    except (ValueError, TypeError):
        pass
    return None


def _has_year(b: TextBlock) -> bool:
    return bool(YEAR_RE.search(b.text))


def _has_month(b: TextBlock) -> bool:
    return any(m in b.text or m[:4] in b.text for m in MONTHS)


def _has_season(b: TextBlock) -> bool:
    return any(s in b.text for s in SEASONS)


def _has_present(b: TextBlock) -> bool:
    return "Present" in b.text or "present" in b.text


def _has_school(b: TextBlock) -> bool:
    return any(s in b.text for s in SCHOOLS)


def _has_degree(b: TextBlock) -> bool:
    return any(d in b.text for d in DEGREES) or bool(re.search(r"[ABM][A-Z.]", b.text))


def _has_job_title(b: TextBlock) -> bool:
    words = b.text.split()
    return any(title in words for title in JOB_TITLES)


def _has_text(text: str):
    """Factory: returns a predicate that checks if the block contains *text*."""
    def pred(b: TextBlock) -> bool:
        return text in b.text
    return pred


# ──────────────────────────────────────────────────────────────────────
# Feature-set definitions  (OpenResume scoring tables)
# ──────────────────────────────────────────────────────────────────────
#
# Each entry: (predicate, score [, extract_match])
#   predicate     – callable(TextBlock) → bool | str | None
#   score         – integer weight  (-4 … +4)
#   extract_match – if True, use the regex match group instead of full text

NAME_FEATURES: list[tuple] = [
    (_only_letters_spaces_period, 3, True),
    (_is_bold, 2),
    (_all_upper, 2),
    (_has_at, -4),
    (_has_number, -4),
    (_has_paren, -4),
    (_has_comma, -4),
    (_has_slash, -4),
    (_has_4plus_words, -2),
]

EMAIL_FEATURES: list[tuple] = [
    (_match_email, 4, True),
    (_is_bold, -1),
    (_all_upper, -1),
    (_has_paren, -4),
    (_has_comma, -4),
    (_has_slash, -4),
    (_has_4plus_words, -4),
]

PHONE_FEATURES: list[tuple] = [
    (_match_phone, 4, True),
    (_has_letter, -4),
]

LOCATION_FEATURES: list[tuple] = [
    (_match_city_state, 4, True),
    (_is_bold, -1),
    (_has_at, -4),
    (_has_paren, -3),
    (_has_slash, -4),
]

URL_FEATURES: list[tuple] = [
    (_match_url, 4, True),
    (_match_url_http, 3, True),
    (_match_url_www, 3, True),
    (_is_bold, -1),
    (_has_at, -4),
    (_has_paren, -3),
    (_has_comma, -4),
    (_has_4plus_words, -4),
]

SUMMARY_FEATURES: list[tuple] = [
    (_has_4plus_words, 4),
    (_is_bold, -1),
    (_has_at, -4),
    (_has_paren, -3),
    (_match_city_state, -4),
]

DATE_FEATURES: list[tuple] = [
    (_has_year, 1),
    (_has_month, 1),
    (_has_season, 1),
    (_has_present, 1),
    (_has_comma, -1),
]

SCHOOL_FEATURES: list[tuple] = [
    (_has_school, 4),
    (_has_degree, -4),
    (_has_number, -4),
]

DEGREE_FEATURES: list[tuple] = [
    (_has_degree, 4),
    (_has_school, -4),
    (_has_number, -3),
]

GPA_FEATURES: list[tuple] = [
    (_match_gpa, 4, True),
    (_match_grade, 3, True),
    (_has_comma, -3),
    (_has_letter, -4),
]

JOB_TITLE_FEATURES: list[tuple] = [
    (_has_job_title, 4),
    (_has_number, -4),
    (_has_5plus_words, -2),
]


# ──────────────────────────────────────────────────────────────────────
# Core scoring engine
# ──────────────────────────────────────────────────────────────────────

def _run_feature_scores(
    items: list[TextBlock],
    feature_sets: list[tuple],
) -> list[dict[str, Any]]:
    """Run every feature set against every text item, return per-item scores."""
    results: list[dict[str, Any]] = []
    for item in items:
        score = 0
        matched_text = item.text
        is_match = False
        for entry in feature_sets:
            pred = entry[0]
            pts = entry[1]
            extract = entry[2] if len(entry) > 2 else False
            res = pred(item)
            if res:
                score += pts
                if extract and isinstance(res, str):
                    matched_text = res
                    is_match = True
        results.append({"text": matched_text, "score": score, "match": is_match})
    return results


def _highest_score(
    items: list[TextBlock],
    feature_sets: list[tuple],
    *,
    return_empty_if_not_positive: bool = True,
    concatenate_ties: bool = False,
) -> tuple[str, list[dict]]:
    """Return the text with the highest feature score."""
    scores = _run_feature_scores(items, feature_sets)
    if not scores:
        return "", scores

    best_score = max(s["score"] for s in scores)
    if return_empty_if_not_positive and best_score <= 0:
        return "", scores

    best_texts = [s["text"] for s in scores if s["score"] == best_score]
    if concatenate_ties:
        return " ".join(t.strip() for t in best_texts), scores
    return best_texts[0] if best_texts else "", scores


def _get_section_lines(
    sections: dict[str, list[Line]], keywords: list[str]
) -> list[Line]:
    """Return lines from the first section whose name contains any keyword."""
    for name, lines in sections.items():
        lower = name.lower()
        if any(kw in lower for kw in keywords):
            return lines
    return []


def _flatten_blocks(lines: list[Line]) -> list[TextBlock]:
    return [b for line in lines for b in line.blocks]


# ──────────────────────────────────────────────────────────────────────
# Sub-section divider  (for education / experience / projects)
# ──────────────────────────────────────────────────────────────────────

def _divide_subsections(lines: list[Line]) -> list[list[Line]]:
    """Split section lines into sub-entries.

    Strategy: detect 'header' lines that start a new entry.
    A header line is one that:
      - contains a year (e.g. 2023), OR
      - is bold AND has ≤ 5 words AND the previous line is NOT bold, OR
      - has a large Y gap (> 1.4× typical gap) from the previous line.
    """
    if not lines:
        return []

    # Compute typical line gap
    ys = [l.y for l in lines]
    gap_counter: Counter[int] = Counter()
    for i in range(1, len(ys)):
        gap = round(ys[i - 1] - ys[i])
        if gap > 0:
            gap_counter[gap] += 1

    if gap_counter:
        common_gap, _ = gap_counter.most_common(1)[0]
        gap_threshold = common_gap * 1.4
    else:
        gap_threshold = 999.0

    def _is_header(line: Line, prev: Line | None) -> bool:
        # Contains a year → new entry
        if YEAR_RE.search(line.text):
            return True
        # Large vertical gap
        if prev is not None:
            gap = prev.y - line.y
            if gap > gap_threshold:
                return True
        # Bold transition (prev not bold, this bold, not a bullet)
        if prev is not None and not prev.is_bold and line.is_bold:
            first_char = line.text[:1] if line.text else ""
            if first_char not in BULLET_POINTS:
                return True
        return False

    subs: list[list[Line]] = []
    cur: list[Line] = [lines[0]]

    for i in range(1, len(lines)):
        prev_line = lines[i - 1]
        this_line = lines[i]
        if _is_header(this_line, prev_line):
            subs.append(cur)
            cur = []
        cur.append(this_line)
    if cur:
        subs.append(cur)

    return subs


# ──────────────────────────────────────────────────────────────────────
# Bullet-point / description extraction
# ──────────────────────────────────────────────────────────────────────

def _get_descriptions_line_idx(lines: list[Line]) -> int | None:
    """Find the first line that is a description (bullet or 8+ words)."""
    # Primary: bullet-point line
    for i, line in enumerate(lines):
        if any(bp in line.text for bp in BULLET_POINTS):
            return i
    # Fallback: first line with ≥ 8 words (single block)
    for i, line in enumerate(lines):
        if len(line.blocks) == 1 and len(line.text.split()) >= 8:
            return i
    return None


def _extract_bullet_points(lines: list[Line]) -> list[str]:
    """Convert lines into a list of bullet-point strings."""
    if not lines:
        return []

    all_text = " ".join(l.text for l in lines)

    # Find most common bullet char
    bullet_counter: Counter[str] = Counter()
    for ch in all_text:
        if ch in BULLET_POINTS:
            bullet_counter[ch] += 1

    if bullet_counter:
        bullet = bullet_counter.most_common(1)[0][0]
        parts = all_text.split(bullet)
        return [p.strip() for p in parts if p.strip()]

    # No bullets → return each non-empty line
    return [l.text.strip() for l in lines if l.text.strip()]


# ──────────────────────────────────────────────────────────────────────
# Public API – FeatureScorer
# ──────────────────────────────────────────────────────────────────────

class FeatureScorer:
    """Stage 4: OpenResume-style feature scoring for resume extraction."""

    # ── Profile ──────────────────────────────────────────────────────

    def extract_profile(self, sections: dict[str, list[Line]]) -> dict:
        profile_lines = _get_section_lines(sections, ["profile"])
        items = _flatten_blocks(profile_lines)

        name, _ = _highest_score(items, NAME_FEATURES)
        email, _ = _highest_score(items, EMAIL_FEATURES)
        phone, _ = _highest_score(items, PHONE_FEATURES)
        location, _ = _highest_score(items, LOCATION_FEATURES)
        url, _ = _highest_score(items, URL_FEATURES)
        summary_text, _ = _highest_score(items, SUMMARY_FEATURES, concatenate_ties=True)

        # Check for dedicated "summary" / "objective" sections
        summary_section = _get_section_lines(sections, ["summary"])
        if summary_section:
            summary_text = " ".join(l.text for l in summary_section)
        else:
            objective_section = _get_section_lines(sections, ["objective"])
            if objective_section:
                summary_text = " ".join(l.text for l in objective_section)

        # LinkedIn / GitHub URLs
        linkedin = None
        github = None
        for item in items:
            text_lower = item.text.lower()
            if "linkedin" in text_lower:
                m = _match_url(item) or _match_url_http(item) or _match_url_www(item)
                if m:
                    linkedin = m
                else:
                    linkedin = item.text
            if "github" in text_lower:
                m = _match_url(item) or _match_url_http(item) or _match_url_www(item)
                if m:
                    github = m
                else:
                    github = item.text

        profile: dict[str, Any] = {}
        if name:
            profile["name"] = name
        if email:
            profile["email"] = email
        if phone:
            profile["phone"] = phone
        if location:
            profile["location"] = location
        if summary_text:
            profile["summary"] = summary_text
        if linkedin:
            profile["linkedin"] = linkedin
        if github:
            profile["github"] = github
        return profile

    # ── Education ────────────────────────────────────────────────────

    def extract_education(self, sections: dict[str, list[Line]]) -> list[dict]:
        edu_lines = _get_section_lines(sections, ["education"])
        subsections = _divide_subsections(edu_lines)
        entries: list[dict] = []

        for sub in subsections:
            items = _flatten_blocks(sub)
            school, _ = _highest_score(items, SCHOOL_FEATURES)
            degree, _ = _highest_score(items, DEGREE_FEATURES)
            gpa, _ = _highest_score(items, GPA_FEATURES)
            date, _ = _highest_score(items, DATE_FEATURES)

            desc_idx = _get_descriptions_line_idx(sub)
            descriptions: list[str] = []
            if desc_idx is not None:
                descriptions = _extract_bullet_points(sub[desc_idx:])

            entry: dict[str, Any] = {}
            if school:
                entry["school"] = school
            if degree:
                entry["degree"] = degree
            if gpa:
                entry["gpa"] = gpa
            if date:
                entry["date"] = date
            if descriptions:
                entry["description"] = " | ".join(descriptions)
            if entry:
                entries.append(entry)

        return entries

    # ── Work Experience ──────────────────────────────────────────────

    def extract_experience(self, sections: dict[str, list[Line]]) -> list[dict]:
        exp_lines = _get_section_lines(sections, ["experience"])
        subsections = _divide_subsections(exp_lines)

        # Merge orphan subsections (no date) with the next subsection
        merged: list[list[Line]] = []
        i = 0
        while i < len(subsections):
            sub = subsections[i]
            has_date = any(YEAR_RE.search(l.text) for l in sub)
            if not has_date and i + 1 < len(subsections):
                # This sub is likely a company-name-only line – merge with next
                merged.append(sub + subsections[i + 1])
                i += 2
            else:
                merged.append(sub)
                i += 1
        subsections = merged

        entries: list[dict] = []

        for sub in subsections:
            # Find the first line with a date – that and the line before are header
            date_line_idx = None
            for idx, line in enumerate(sub):
                if YEAR_RE.search(line.text):
                    date_line_idx = idx
                    break

            if date_line_idx is None:
                # No date found – treat all as bullets
                bullets = _extract_bullet_points(sub)
                if bullets:
                    entries.append({"bullets": bullets})
                continue

            # Header = lines from start to date_line_idx (inclusive)
            header_end = date_line_idx + 1
            header_lines = sub[:header_end]
            desc_lines = sub[header_end:]

            info_items = _flatten_blocks(header_lines)
            date, _ = _highest_score(info_items, DATE_FEATURES)
            job_title, _ = _highest_score(info_items, JOB_TITLE_FEATURES)

            # Company: first line of header if it doesn't match date/title
            company = None
            if header_lines:
                first_line_text = header_lines[0].text.strip()
                # Skip if it's a bullet or matches date/title
                first_char = first_line_text[:1] if first_line_text else ""
                is_bullet = first_char in BULLET_POINTS
                if not is_bullet:
                    # Check if first line is the company (not the job title or date)
                    if job_title and job_title.lower() not in first_line_text.lower():
                        company = first_line_text
                    elif not job_title and not YEAR_RE.search(first_line_text):
                        company = first_line_text
                    elif job_title and job_title.lower() in first_line_text.lower() and len(header_lines) > 1:
                        # Title and company on same line or company on second line
                        pass

                # Also check second line for company name
                if not company and len(header_lines) > 1:
                    second_text = header_lines[1].text.strip()
                    second_first_char = second_text[:1] if second_text else ""
                    if (second_first_char not in BULLET_POINTS
                            and not YEAR_RE.search(second_text)
                            and (not job_title or job_title.lower() not in second_text.lower())):
                        company = second_text

            bullets = _extract_bullet_points(desc_lines) if desc_lines else []

            entry: dict[str, Any] = {}
            if company:
                entry["company"] = company
            if job_title:
                entry["title"] = job_title
            if date:
                entry["date"] = date
            if bullets:
                entry["bullets"] = bullets
            if entry:
                entries.append(entry)

        return entries

    # ── Projects ─────────────────────────────────────────────────────

    def extract_projects(self, sections: dict[str, list[Line]]) -> list[dict]:
        proj_lines = _get_section_lines(sections, ["project"])
        subsections = _divide_subsections(proj_lines)
        entries: list[dict] = []

        for sub in subsections:
            desc_idx = _get_descriptions_line_idx(sub)
            if desc_idx is None:
                desc_idx = min(1, len(sub))

            info_items = _flatten_blocks(sub[:desc_idx])
            date, _ = _highest_score(info_items, DATE_FEATURES)

            project_features: list[tuple] = [
                (_is_bold, 2),
                (_has_text(date) if date else lambda b: False, -4),
            ]
            project_name, _ = _highest_score(
                info_items, project_features,
                return_empty_if_not_positive=False,
            )

            descriptions: list[str] = []
            if desc_idx < len(sub):
                descriptions = _extract_bullet_points(sub[desc_idx:])

            entry: dict[str, Any] = {}
            if project_name:
                entry["name"] = project_name
            if date:
                entry["date"] = date
            if descriptions:
                entry["description"] = " | ".join(descriptions)
            if entry:
                entries.append(entry)

        return entries

    # ── Skills ───────────────────────────────────────────────────────

    def extract_skills(self, sections: dict[str, list[Line]]) -> list[str]:
        skill_lines = _get_section_lines(sections, ["skill"])
        if not skill_lines:
            return []

        desc_idx = _get_descriptions_line_idx(skill_lines)
        if desc_idx is not None:
            lines = skill_lines[desc_idx:]
        else:
            lines = skill_lines

        skills: list[str] = []
        for line in lines:
            text = line.text.strip()
            # Split by common delimiters
            parts = re.split(r"[,;|•·]", text)
            for part in parts:
                skill = re.sub(r"^[\-\*\u2022\u25cf\s]+", "", part).strip()
                if skill and len(skill) < 60:
                    skills.append(skill)
        return skills
