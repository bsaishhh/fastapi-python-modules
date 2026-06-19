from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo.errors import PyMongoError

from app.core.config import settings
from app.core.exceptions import AppError
from app.infrastructure.mongodb import get_database

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ResumeRepository:
    def __init__(self) -> None:
        self._db = None

    @property
    def db(self):
        if self._db is None:
            self._db = get_database()
        return self._db

    @property
    def resumes(self):
        return self.db[settings.collection_resume]

    @property
    def extractions(self):
        return self.db[settings.collection_extraction]

    @property
    def analyses(self):
        return self.db[settings.collection_domain_analysis]

    async def save_resume_upload(
        self,
        filename: str,
        content_hash: str,
        role: str | None = None,
        extracted_text: str | None = None,
    ) -> str:
        doc = {
            "filename": filename,
            "content_hash": content_hash,
            "role": role,
            "resume_text": extracted_text or "",
            "extracted_text": extracted_text or "",
            "created_at": _utcnow(),
        }
        try:
            result = await self.resumes.insert_one(doc)
        except PyMongoError as exc:
            logger.error("MongoDB save_resume_upload failed: %s", exc)
            raise AppError(f"Database error: {exc}", 503)
        return str(result.inserted_id)

    async def save_extraction(
        self,
        resume_id: str,
        extracted_text: str,
    ) -> str:
        doc = {
            "resume_id": resume_id,
            "extracted_text": extracted_text,
            "extraction_version": settings.extraction_version,
            "created_at": _utcnow(),
        }
        try:
            result = await self.extractions.insert_one(doc)
        except PyMongoError as exc:
            logger.error("MongoDB save_extraction failed: %s", exc)
            raise AppError(f"Database error: {exc}", 503)
        return str(result.inserted_id)

    async def save_domain_analysis(
        self,
        resume_id: str,
        extraction_id: str,
        role: str,
        master_jd: dict[str, Any],
        analysis: dict[str, Any],
        scores: dict[str, Any] | None = None,
    ) -> str:
        doc = {
            "resume_id": resume_id,
            "extraction_id": extraction_id,
            "role": role.upper(),
            "master_jd": master_jd,
            "scores": scores or {},
            "analysis": analysis,
            "analysis_version": settings.analysis_version,
            "scoring_version": (scores or {}).get("scoring_version", settings.scoring_version),
            "created_at": _utcnow(),
        }
        try:
            result = await self.analyses.insert_one(doc)
        except PyMongoError as exc:
            logger.error("MongoDB save_domain_analysis failed: %s", exc)
            raise AppError(f"Database error: {exc}", 503)
        return str(result.inserted_id)

    async def get_extraction(self, extraction_id: str) -> dict[str, Any] | None:
        if not ObjectId.is_valid(extraction_id):
            return None
        try:
            return await self.extractions.find_one({"_id": ObjectId(extraction_id)})
        except PyMongoError as exc:
            logger.error("MongoDB get_extraction failed: %s", exc)
            raise AppError(f"Database error: {exc}", 503)

    async def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        try:
            cursor = self.analyses.find().sort("created_at", -1).limit(limit)
            results = []
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                results.append(doc)
            return results
        except PyMongoError as exc:
            logger.error("MongoDB list_recent failed: %s", exc)
            raise AppError(f"Database error: {exc}", 503)
