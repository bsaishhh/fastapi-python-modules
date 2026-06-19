from fastapi import APIRouter

from modules.resume_ats.api.v1 import analyze, domain_detect, extract, feedback, results, score, scores, roles

PREFIX = "/resume"
TAGS = ["resume-ats"]

router = APIRouter()
router.include_router(extract.router)
router.include_router(score.router)
router.include_router(scores.router)
router.include_router(analyze.router)
router.include_router(domain_detect.router)
router.include_router(feedback.router)
router.include_router(results.router)
router.include_router(roles.router)
