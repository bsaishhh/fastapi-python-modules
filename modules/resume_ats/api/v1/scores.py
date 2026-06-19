from fastapi import APIRouter

from modules.resume_ats.schemas.api import ATSScoreBreakdown, ResumeAnalysisRequest
from modules.resume_ats.services.resume_service import ResumeAnalysisService

router = APIRouter()
analysis_service = ResumeAnalysisService()


@router.post("/scores", response_model=ATSScoreBreakdown)
async def scores_only(request: ResumeAnalysisRequest) -> ATSScoreBreakdown:
    """Deterministic ATS scores only (no LLM, no MongoDB writes)."""
    result = await analysis_service.score_only(request.extracted_resume, request.role)
    return ATSScoreBreakdown(**result)
