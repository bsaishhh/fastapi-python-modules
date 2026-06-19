from fastapi import APIRouter

from modules.resume_ats.schemas.api import ATSAnalysisResponse, ResumeAnalysisRequest
from modules.resume_ats.services.resume_service import ResumeAnalysisService

router = APIRouter()
analysis_service = ResumeAnalysisService()


@router.post("/analyze", response_model=ATSAnalysisResponse)
async def analyze_resume(request: ResumeAnalysisRequest) -> ATSAnalysisResponse:
    result = await analysis_service.analyze_from_structured(
        role=request.role,
        structured_resume=request.extracted_resume,
        resume_id=request.resume_id,
        extraction_id=request.extraction_id,
    )
    return ATSAnalysisResponse(**result)
