from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "iit" / "department_role_map.json"


@lru_cache(maxsize=1)
def load_department_map() -> dict[str, list[str]]:
    with DATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def get_roles_for_department(department: str) -> list[str]:
    mapping = load_department_map()
    return mapping.get(department, [])
