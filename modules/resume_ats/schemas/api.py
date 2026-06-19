from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProfileSchema(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    summary: str | None = None
    linkedin: str | None = None
    github: str | None = None


class StructuredResumeSchema(BaseModel):
    profile: ProfileSchema = Field(default_factory=ProfileSchema)
    education: list[dict[str, Any]] = Field(default_factory=list)
    experience: list[dict[str, Any]] = Field(default_factory=list)
    projects: list[dict[str, Any]] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    publications: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)


class ExtractResponse(BaseModel):
    resume_id: str
    extraction_id: str
    extracted_text: str
    extraction_version: str


class MissingKeywordSeveritySchema(BaseModel):
    critical: list[str] = Field(default_factory=list)
    important: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)


class ATSScoreBreakdown(BaseModel):
    overall_score: int
    semantic_score: int
    keyword_score: int
    bm25_score: int = 0
    jaccard_score: int = 0
    exact_keyword_score: int
    domain_score: int
    experience_score: int = 0
    skill_depth_score: int = 0
    resume_quality_score: int = 0
    domain_boost: float = 0.0
    missing_keyword_severity: MissingKeywordSeveritySchema = Field(
        default_factory=MissingKeywordSeveritySchema
    )
    detected_domain: str | None = None
    scoring_version: str
    embedding_model: str | None = None


class AnalysisScoreItem(BaseModel):
    score: int
    maxScore: int
    description: str


class AnalysisScoreBreakdownSchema(BaseModel):
    keywordMatch: AnalysisScoreItem
    formatting: AnalysisScoreItem
    sectionStructure: AnalysisScoreItem
    contentQuality: AnalysisScoreItem


class AnalysisSectionSchema(BaseModel):
    overallScore: int
    scoreBreakdown: AnalysisScoreBreakdownSchema
    strengths: list[str] = Field(default_factory=list)
    areasOfImprovement: list[str] = Field(default_factory=list)
    criticalMissingKeywords: list[str] = Field(default_factory=list)
    actionPlan: list[Any] = Field(default_factory=list)


class ScoreLevelSchema(BaseModel):
    level: str
    color: str
    description: str


class AnalysisDataSchema(BaseModel):
    llm_overall_score: int
    scoreLevel: ScoreLevelSchema
    scoreBreakdown: AnalysisScoreBreakdownSchema
    strengths: list[str] = Field(default_factory=list)
    areasOfImprovement: list[str] = Field(default_factory=list)
    criticalMissingKeywords: list[str] = Field(default_factory=list)
    actionPlan: list[Any] = Field(default_factory=list)
    screenshot: str | None = None
    role: str
    processingTimeMs: int


class ATSAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: bool = True
    cached: bool = False
    data: AnalysisDataSchema
    timestamp: str
    requestId: str
    resume_id: str | None = None
    extraction_id: str | None = None
    analysis_id: str | None = None
    role: str
    analysis_version: str
    detected_domain: str | None = None
    scoring_version: str
    recommended_primary_role: str | None = None
    deterministic_overall_score: int
    llm_overall_score: int


class ResumeAnalysisRequest(BaseModel):
    role: str
    extracted_resume: str | dict[str, Any]
    resume_id: str | None = None
    extraction_id: str | None = None


class DomainDetectResponse(BaseModel):
    role: str | None = None
    family: str | None = None
    domain_score: int | None = None
    best_role: str | None = None
    department: str | None = None
    suggested_roles: list[str] = Field(default_factory=list)
    matched_required: list[str] = Field(default_factory=list)
    matched_preferred: list[str] = Field(default_factory=list)
    missing_critical: list[str] = Field(default_factory=list)
    rankings: list[list] = Field(default_factory=list)
