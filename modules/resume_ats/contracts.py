from typing import TypedDict


class Profile(TypedDict, total=False):
    name: str
    email: str
    phone: str
    location: str
    summary: str
    linkedin: str
    github: str


class EducationEntry(TypedDict, total=False):
    school: str
    degree: str
    field: str
    start_date: str
    end_date: str
    gpa: str
    description: str


class ExperienceEntry(TypedDict, total=False):
    company: str
    title: str
    location: str
    start_date: str
    end_date: str
    description: str
    bullets: list[str]


class ProjectEntry(TypedDict, total=False):
    name: str
    description: str
    technologies: list[str]
    url: str


class StructuredResume(TypedDict):
    profile: Profile
    education: list[EducationEntry]
    experience: list[ExperienceEntry]
    projects: list[ProjectEntry]
    skills: list[str]
    certifications: list[str]
    publications: list[str]
    achievements: list[str]


class ResumeEntities(TypedDict):
    skills: list[str]
    tools: list[str]
    frameworks: list[str]
    languages: list[str]
    degrees: list[str]
    companies: list[str]
    projects: list[str]
    research: list[str]
    certifications: list[str]
    experience_years: float


class RoleJD(TypedDict):
    role_key: str
    family: str
    sub_roles: list[str]
    required_skills: list[str]
    preferred_skills: list[str]
    tools: list[str]
    frameworks: list[str]
    responsibilities: list[str]
    keywords: list[str]


class MissingKeywordSeverity(TypedDict, total=False):
    critical: list[str]
    important: list[str]
    optional: list[str]


class ScoreBreakdown(TypedDict):
    overall_score: int
    semantic_score: int
    bm25_score: int
    keyword_score: int
    exact_keyword_score: int
    domain_score: int
    experience_score: int
    skill_depth_score: int
    resume_quality_score: int
    domain_boost: float
    missing_keyword_severity: MissingKeywordSeverity
    detected_domain: str
    scoring_version: str
    embedding_model: str


class FeedbackResult(TypedDict):
    strengths: list[str]
    weaknesses: list[str]
    critical_missing_keywords: list[str]
    action_items: list[str]
