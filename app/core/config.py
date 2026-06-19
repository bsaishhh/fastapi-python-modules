from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "FastAPI Features"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    mongo_uri: str = ""
    mongo_db_name: str = "fastapi_features"

    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    collection_resume: str = "resume-new"
    collection_extraction: str = "extraction"
    collection_domain_analysis: str = "domain-analysis"

    resume_extraction_api_url: str = Field(
        default="https://resume-builder.cantileverlabs.com/api/resume-parsing/extract-text",
        validation_alias=AliasChoices("RESUME_EXTRACTION_API_URL", "resumeExtractionApiUrl"),
    )
    bearer_token: str = Field(
        default="",
        validation_alias=AliasChoices("BEARER_TOKEN", "bearerToken"),
    )
    resume_parser_url: str = Field(
        default="https://resume-builder.cantileverlabs.com/api/resume-parsing/extract-text",
        validation_alias=AliasChoices("RESUME_PARSER_URL", "resume_parser_url"),
    )

    extraction_version: str = "cantilever-parser-v1"
    analysis_version: str = "llm-v1"
    scoring_version: str = "ats-scoring-v2"

    sbert_model: str = "0xnbk/nbk-ats-semantic-v1-en"


settings = Settings()
