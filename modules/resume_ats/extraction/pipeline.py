from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import AppError

logger = logging.getLogger(__name__)

# Timeout for external parser API call (seconds)
_PARSER_TIMEOUT = 300.0
# Retry configuration for transient 502/503/504 errors
_MAX_RETRIES = 2
_RETRY_BACKOFF = 5.0  # seconds between retries (doubles each attempt)
class ExtractionPipeline:
    """Calls the Cantilever Labs extract-text API and returns raw resume text."""

    def __init__(self) -> None:
        self.parser_url = settings.resume_extraction_api_url or settings.resume_parser_url

    async def run(
        self,
        file_bytes: bytes,
        filename: str = "resume.pdf",
        bearer_token: str | None = None,
    ) -> str:
        """Upload PDF to the external extractor and return full resume text."""
        api_response = await self._call_parser(file_bytes, filename, bearer_token=bearer_token)
        extracted_text = (((api_response.get("data") or {}).get("text")) or "").strip()
        if not extracted_text:
            raise AppError("Resume extraction API returned empty text", 502)
        return extracted_text

    # ---------------------------------------------------------------------------
    # External API call
    # ---------------------------------------------------------------------------

    async def _call_parser(
        self,
        file_bytes: bytes,
        filename: str,
        bearer_token: str | None = None,
    ) -> dict[str, Any]:
        """POST the PDF file to the extract-text endpoint with bearer auth."""
        headers = {}
        token = (bearer_token or settings.bearer_token).strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # Deployed extractor uses upload.single('jdFile')
        files = {"jdFile": (filename, file_bytes, "application/pdf")}
        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 2):  # 1 initial + _MAX_RETRIES retries
            async with httpx.AsyncClient(timeout=_PARSER_TIMEOUT) as client:
                try:
                    resp = await client.post(self.parser_url, files=files, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    logger.debug("Extraction API returned keys: %s", list(data.keys()))
                    return data
                except httpx.TimeoutException:
                    logger.error(
                        "Resume extraction API timed out after %ss (attempt %d/%d)",
                        _PARSER_TIMEOUT, attempt, _MAX_RETRIES + 1,
                    )
                    last_error = AppError("Resume extraction service timed out", 504)
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    logger.error(
                        "Resume extraction API HTTP %s (attempt %d/%d): %s",
                        status, attempt, _MAX_RETRIES + 1,
                        exc.response.text[:500],
                    )
                    # Retry only on transient gateway errors (502, 503, 504)
                    if status in (502, 503, 504) and attempt <= _MAX_RETRIES:
                        last_error = AppError(f"Resume extraction API returned HTTP {status}", 502)
                    else:
                        raise AppError(f"Resume extraction API returned HTTP {status}", 502)
                except AppError:
                    raise
                except Exception as exc:
                    logger.error("Resume extraction API error: %s", exc)
                    raise AppError(f"Resume extraction API error: {exc}", 502)

            # Backoff before retry (5s, 10s, ...)
            if attempt <= _MAX_RETRIES:
                wait = _RETRY_BACKOFF * attempt
                logger.info("Retrying extractor in %ss (attempt %d/%d)...", wait, attempt + 1, _MAX_RETRIES + 1)
                await asyncio.sleep(wait)

        # All retries exhausted
        raise last_error or AppError("Resume extraction API failed after retries", 502)
