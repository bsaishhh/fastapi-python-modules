from fastapi import APIRouter, File, Form, UploadFile

from app.core.exceptions import ValidationError
from modules.resume_ats.schemas.api import DomainDetectResponse
from modules.resume_ats.services.resume_service import ResumeAnalysisService

router = APIRouter()
analysis_service = ResumeAnalysisService()


@router.post("/domain-detect", response_model=DomainDetectResponse)
async def domain_detect(
    resume_file: UploadFile = File(...),
    role: str | None = Form(default=None),
    department: str | None = Form(default=None),
    bearer_token: str | None = Form(default=None),
) -> DomainDetectResponse:
    if not resume_file.filename or not resume_file.filename.lower().endswith(".pdf"):
        raise ValidationError("Only PDF resume files are supported")
    file_bytes = await resume_file.read()
    if not file_bytes:
        raise ValidationError("Empty file uploaded")

    result = await analysis_service.domain_detect(
        file_bytes,
        role=role,
        department=department,
        bearer_token=bearer_token,
    )
    return DomainDetectResponse(**result)
