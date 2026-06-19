from fastapi import APIRouter, Query, Body
from typing import Optional
import json
from pathlib import Path

from modules.resume_ats.data.jd_loader import list_roles as _list_jds

router = APIRouter(prefix="/roles", tags=["resume-ats"])

# Path to roles.json from api/v1/roles.py → ../../data/roles.json
ROLES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "roles.json"


def load_roles():
    with open(ROLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/", summary="List available roles and sub-roles")
def list_roles(job_description: Optional[str] = Query(None, description="Optional job description to prefer relevant roles")):
    roles = load_roles()
    # If job_description provided, return same list but mark best-effort matches (simple keyword match)
    if job_description:
        jd = job_description.lower()
        for r in roles:
            r["match_score"] = 1 if r["role"].lower() in jd or any(sr.lower() in jd for sr in r.get("sub_roles", [])) else 0
    return {"roles": roles}


@router.post("/select", summary="Select a role (accepts optional job description)")
def select_role(role: str = Body(..., embed=True), job_description: Optional[str] = Body(None)):
    roles = load_roles()
    for r in roles:
        if r["role"].lower() == role.lower() or any(sr.lower() == role.lower() for sr in r.get("sub_roles", [])):
            return {"selected": r, "job_description": job_description}
    return {"error": "role not found"}


@router.get("/jds", summary="List available JD role keys (for frontend)")
def list_jd_roles():
    """Fetch all available JD role keys from backend."""
    try:
        jds = _list_jds()
        # Map file stems to uppercase role keys
        return {"jds": sorted([s.upper() for s in jds])}
    except Exception as e:
        return {"jds": [], "error": str(e)}


@router.get("/all", summary="List all roles and sub-roles")
def list_all_roles():
    """Fetch all roles with sub-roles from roles.json."""
    try:
        roles = load_roles()
        return {"roles": roles, "count": len(roles)}
    except Exception as e:
        return {"roles": [], "count": 0, "error": str(e)}
