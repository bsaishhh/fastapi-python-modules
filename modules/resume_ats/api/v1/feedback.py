from fastapi import APIRouter, File, Form, UploadFile

from app.core.exceptions import ValidationError
from modules.resume_ats.schemas.api import ATSAnalysisResponse
from modules.resume_ats.services.resume_service import ResumeAnalysisService

router = APIRouter()
analysis_service = ResumeAnalysisService()


@router.post("/feedback", response_model=ATSAnalysisResponse)
async def feedback(
    resume_file: UploadFile = File(...),
    role: str = Form(..., examples=["AI_ML_ENGINEER"]),
    bearer_token: str | None = Form(default=None),
) -> ATSAnalysisResponse:
    if not resume_file.filename or not resume_file.filename.lower().endswith(".pdf"):
        raise ValidationError("Only PDF resume files are supported")
    file_bytes = await resume_file.read()
    if not file_bytes:
        raise ValidationError("Empty file uploaded")

    result = await analysis_service.analyze_from_file(
        file_bytes=file_bytes,
        filename=resume_file.filename,
        role=role,
        bearer_token=bearer_token,
    )
    return ATSAnalysisResponse(**result)
