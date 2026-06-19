import pytest
from fastapi.testclient import TestClient

from app.main import app
from modules.resume_ats.analysis.llm_analyzer import ATSAnalysis, LLMAnalyzer
from modules.resume_ats.db.repository import ResumeRepository
from modules.resume_ats.extraction.pipeline import ExtractionPipeline


SAMPLE_STRUCTURED = {
    "profile": {"name": "John Doe", "email": "john@email.com", "summary": "AI ML engineer"},
    "education": [{"degree": "B.Tech CS", "school": "IIT"}],
    "experience": [{"company": "TechCorp", "title": "ML Engineer", "start_date": "2020", "end_date": "2024"}],
    "projects": [{"name": "Classifier", "technologies": ["PyTorch"]}],
    "skills": ["Python", "PyTorch", "FastAPI", "Docker", "AWS"],
    "certifications": [],
    "publications": [],
    "achievements": [],
}

SAMPLE_EXTRACTED_TEXT = """
JOHN DOE
john@email.com
SUMMARY
AI ML engineer with Python, PyTorch, FastAPI, Docker, and AWS experience.
EXPERIENCE
ML Engineer at TechCorp from 2020 to 2024
Built production ML APIs with FastAPI and improved latency by 40%.
PROJECTS
Classifier project using PyTorch and FastAPI
SKILLS
Python, PyTorch, FastAPI, Docker, AWS
EDUCATION
B.Tech CS, IIT
""".strip()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_mongo_llm_and_extraction(monkeypatch):
    # Global mock to prevent downloading large embedding models during tests
    from app.core.config import settings
    monkeypatch.setattr(settings, "sbert_model", "BAAI/bge-large-en-v1.5")
    from modules.resume_ats.scoring.semantic_scorer import get_embedding_model
    get_embedding_model.cache_clear()
    monkeypatch.setattr(
        "modules.resume_ats.scoring.semantic_scorer.get_embedding_model",
        lambda: None,
    )

    class FakeInsertResult:
        inserted_id = "674abc123def456789012345"

    class FakeCollection:
        async def insert_one(self, doc):
            return FakeInsertResult()

        def find(self):
            return self

        def sort(self, *args, **kwargs):
            return self

        def limit(self, n):
            return self

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def find_one(self, query):
            return None

    class FakeDB:
        def __getitem__(self, name):
            return FakeCollection()

    monkeypatch.setattr("modules.resume_ats.db.repository.get_database", lambda: FakeDB())

    async def fake_list_recent(self, limit=20):
        return []

    monkeypatch.setattr(ResumeRepository, "list_recent", fake_list_recent)

    def fake_analyze(self, role, resume_text, entities, master_jd):
        return ATSAnalysis(
            best_fit_subroles=["Generalist"],
            strengths=["Strong Python background"],
            areas_of_improvement=["Missing RAG experience"],
            critical_missing_keywords=["RAG", "LangChain"],
            action_plan=[{"priority": "High", "action": "Add a RAG project", "impact": "Improves role match", "example": "Describe a retrieval project"}],
        )

    monkeypatch.setattr(LLMAnalyzer, "analyze", fake_analyze)

    async def fake_extract_run(self, file_bytes, filename="resume.pdf"):
        return SAMPLE_EXTRACTED_TEXT

    monkeypatch.setattr(ExtractionPipeline, "run", fake_extract_run)


def _minimal_pdf_with_text(text_lines: list[str]) -> bytes:
    content_stream = "BT\n/F1 10 Tf\n"
    y = 750
    for line in text_lines:
        safe = line.replace("(", "\\(").replace(")", "\\)")
        content_stream += f"50 {y} Td\n({safe}) Tj\n0 -14 Td\n"
        y -= 14
    content_stream += "ET"

    objects = []
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objects.append(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R "
        "/MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )
    objects.append(
        f"4 0 obj\n<< /Length {len(content_stream)} >>\nstream\n{content_stream}\nendstream\nendobj\n"
    )
    objects.append("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    pdf = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj

    xref_pos = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n"
    pdf += "0000000000 65535 f \n"
    for off in offsets[1:]:
        pdf += f"{off:010d} 00000 n \n"
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
    pdf += f"startxref\n{xref_pos}\n%%EOF"
    return pdf.encode("latin-1")


@pytest.fixture
def sample_resume_pdf() -> bytes:
    return _minimal_pdf_with_text([
        "JOHN DOE",
        "john.doe@email.com",
        "SUMMARY",
        "AI ML engineer with Python PyTorch FastAPI Docker",
        "EXPERIENCE",
        "ML Engineer TechCorp 2022 - Present",
        "Built production ML APIs with FastAPI",
        "SKILLS",
        "Python, PyTorch, TensorFlow, FastAPI, Docker, AWS",
        "EDUCATION",
        "B.Tech Computer Science IIT 2018 - 2022",
    ])
