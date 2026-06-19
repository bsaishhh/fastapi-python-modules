from modules.resume_ats.contracts import StructuredResume
from modules.resume_ats.entities.extractor import EntityExtractor
from modules.resume_ats.extraction.line_reconstruction.line_grouper import LineGrouper
from modules.resume_ats.extraction.models.text_block import TextBlock


def test_line_grouper_merges_blocks_on_same_y():
    blocks = [
        TextBlock(text="Hello", x=10, y=100, width=30, height=12, fontname="Helvetica", bold=False),
        TextBlock(text="World", x=45, y=100, width=30, height=12, fontname="Helvetica", bold=False),
    ]
    lines = LineGrouper().group_lines(blocks)
    assert len(lines) == 1
    assert "Hello" in lines[0].text and "World" in lines[0].text


def test_entity_extractor_finds_python_and_pytorch():
    resume: StructuredResume = {
        "profile": {"name": "Test", "summary": "Python developer"},
        "education": [],
        "experience": [{"company": "Co", "title": "ML Engineer", "start_date": "2020", "end_date": "2024"}],
        "projects": [{"name": "NLP Project", "technologies": ["PyTorch"]}],
        "skills": ["Python", "FastAPI", "Docker"],
        "certifications": [],
        "publications": [],
        "achievements": [],
    }
    entities = EntityExtractor().extract(resume)
    assert "python" in entities["languages"]
    assert "pytorch" in entities["frameworks"]
    assert entities["experience_years"] > 0
