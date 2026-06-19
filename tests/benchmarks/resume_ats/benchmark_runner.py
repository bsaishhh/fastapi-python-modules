"""Benchmarking framework for resume ATS analysis latency."""

from __future__ import annotations

import time
from pathlib import Path

from modules.resume_ats.services.resume_service import ResumeAnalysisService

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_resumes"


class BenchmarkRunner:
    def __init__(self) -> None:
        self.service = ResumeAnalysisService()

    async def run(self, pdf_path: Path, role: str) -> dict:
        file_bytes = pdf_path.read_bytes()
        start = time.perf_counter()
        result = await self.service.analyze_from_file(file_bytes, pdf_path.name, role)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"latency_ms": round(elapsed_ms, 2), **result}
