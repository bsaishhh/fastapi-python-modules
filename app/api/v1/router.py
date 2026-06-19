from fastapi import APIRouter

from app.core.module_registry import register_modules

api_v1_router = APIRouter()
register_modules(api_v1_router)
