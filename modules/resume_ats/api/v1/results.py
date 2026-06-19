from fastapi import APIRouter

from modules.resume_ats.db.repository import ResumeRepository

router = APIRouter()


@router.get("/results")
async def list_results(limit: int = 20):
    repo = ResumeRepository()
    results = await repo.list_recent(limit=limit)
    return {"success": True, "count": len(results), "results": results}
