from fastapi import APIRouter, File, Form, UploadFile

from app.core.exceptions import ValidationError
from modules.resume_ats.schemas.api import ExtractResponse
from modules.resume_ats.services.resume_service import ExtractService

router = APIRouter()
extract_service = ExtractService()


@router.post("/extract", response_model=ExtractResponse)
async def extract_resume(
    resume_file: UploadFile = File(...),
    role: str | None = Form(default=None),
    bearer_token: str | None = Form(default=None),
) -> ExtractResponse:
    if not resume_file.filename or not resume_file.filename.lower().endswith(".pdf"):
        raise ValidationError("Only PDF resume files are supported")
    file_bytes = await resume_file.read()
    if not file_bytes:
        raise ValidationError("Empty file uploaded")

    result = await extract_service.extract_and_save(
        file_bytes=file_bytes,
        filename=resume_file.filename,
        role=role,
        bearer_token=bearer_token,
    )
    return ExtractResponse(
        resume_id=result["resume_id"],
        extraction_id=result["extraction_id"],
        extracted_text=result["extracted_text"],
        extraction_version=result["extraction_version"],
    )
