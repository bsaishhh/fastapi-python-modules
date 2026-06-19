import importlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_REGISTERED_MODULES: list[str] = []


def register_modules(api_router: APIRouter) -> list[str]:
    module_paths = ["modules.resume_ats"]
    registered: list[str] = []
    for module_path in module_paths:
        mod = importlib.import_module(f"{module_path}.router")
        prefix: str = getattr(mod, "PREFIX", "")
        tags: list[str] = getattr(mod, "TAGS", [module_path.split(".")[-1]])
        api_router.include_router(mod.router, prefix=prefix, tags=tags)
        registered.append(module_path)
    global _REGISTERED_MODULES
    _REGISTERED_MODULES = registered
    return registered


def create_app() -> FastAPI:
    from app.api.v1.router import api_v1_router
    from app.core.config import settings
    from app.core.exceptions import AppError, app_error_handler
    from app.core.logging import setup_logging
    from app.infrastructure.mongodb import close_mongo_client

    setup_logging(settings.debug)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Eagerly load/cache the embedding model during startup (loaded once, reused forever)
        try:
            from modules.resume_ats.scoring.semantic_scorer import _load_auto_model, get_ensemble_mappers
            _load_auto_model()       # warms up AutoTokenizer + AutoModel
            get_ensemble_mappers()   # warms up Ridge/MLP/Poly ensemble weights
        except Exception:
            pass
        yield
        await close_mongo_client()

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.add_exception_handler(AppError, app_error_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_v1_router, prefix=settings.api_v1_prefix)

    project_root = Path(__file__).resolve().parent.parent.parent
    static_dir = project_root / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        async def test_ui():
            return FileResponse(static_dir / "index.html")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "modules": _REGISTERED_MODULES or ["resume_ats"],
            "mongo_configured": bool(settings.mongo_uri),
            "openrouter_configured": bool(settings.openrouter_api_key),
            "scoring": {
                "version": settings.scoring_version,
                "embedding_model": settings.sbert_model,
                "library": "transformers (AutoModel + mean pooling)",
            },
        }

    return app
