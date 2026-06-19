from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.core.exceptions import NotFoundError
from modules.resume_ats.contracts import RoleJD

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "jds"

ROLE_ALIASES = {
    "AI_ML_ENGINEER": "ai_ml_engineer",
    "SOFTWARE_ENGINEER": "software_engineer",
    "DATA_ENGINEER": "data_engineer",
    "QUANT_FINANCE": "quant_finance",
    "CONSULTING_STRATEGY": "consulting_strategy",
    "PRODUCT_DESIGN": "product_design",
    "MECHANICAL_MANUFACTURING": "mechanical_manufacturing",
    "ELECTRICAL_ELECTRONICS": "electrical_electronics",
    "AEROSPACE_DEFENCE": "aerospace_defence",
    "CORE_SCIENCE_RND": "core_science_rnd",
    "CIVIL_INFRASTRUCTURE": "civil_infrastructure",
    "ROBOTICS_AUTONOMOUS": "robotics_autonomous",
    "FOUNDERS_OFFICE": "founders_office",
    "EDUCATION_EDTECH": "education_edtech",
    "GAMING_GRAPHICS": "gaming_graphics",
    "SUPPLY_CHAIN_OPERATIONS": "supply_chain_operations",
}


@lru_cache(maxsize=512)
def load_jd(role: str) -> RoleJD:
    normalized = ROLE_ALIASES.get(role.upper(), role.lower())
    path = DATA_DIR / f"{normalized}.json"
    if not path.exists():
        raise NotFoundError(f"JD not found for role: {role}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data  # type: ignore[return-value]


def list_roles() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.json"))
