import pytest

from modules.resume_ats.contracts import ResumeEntities, StructuredResume
from modules.resume_ats.data.jd_loader import load_jd
from modules.resume_ats.domain_classifier.classifier import DomainClassifier
from modules.resume_ats.entities.extractor import EntityExtractor
from modules.resume_ats.scoring.orchestrator import ScoringOrchestrator
from modules.resume_ats.services.resume_service import _build_score_breakdown, _build_weighted_overall_score




def test_scoring_orchestrator_returns_all_scores():
    resume: StructuredResume = {
        "profile": {"summary": "Built retrieval augmented generation systems with Python and PyTorch"},
        "education": [{"degree": "B.Tech CS"}],
        "experience": [{"company": "Tech", "title": "ML Engineer", "start_date": "2020", "end_date": "2024"}],
        "projects": [{"name": "RAG Chatbot", "technologies": ["PyTorch", "LangChain"]}],
        "skills": ["Python", "PyTorch", "FastAPI", "Docker", "AWS"],
        "certifications": ["AWS Certified"],
        "publications": [],
        "achievements": ["Hackathon Winner"],
    }
    entities: ResumeEntities = {
        "skills": ["Python", "PyTorch", "FastAPI", "Docker", "AWS", "LangChain"],
        "tools": ["docker", "aws"],
        "frameworks": ["pytorch", "fastapi", "langchain"],
        "languages": ["python"],
        "degrees": ["B.Tech"],
        "companies": ["Tech"],
        "projects": ["RAG Chatbot"],
        "research": [],
        "certifications": ["AWS Certified"],
        "experience_years": 4.0,
    }
    jd = load_jd("AI_ML_ENGINEER")
    result = ScoringOrchestrator().score(resume, entities, jd)

    assert 0 <= result["overall_score"] <= 100
    assert 0 <= result["semantic_score"] <= 100
    assert 0 <= result["keyword_score"] <= 100
    assert 0 <= result["bm25_score"] <= 100
    assert 0 <= result["jaccard_score"] <= 100
    assert 0 <= result["exact_keyword_score"] <= 100
    assert 0 <= result["domain_score"] <= 100
    assert 0 <= result["experience_score"] <= 100
    assert 0 <= result["skill_depth_score"] <= 100
    assert 0 <= result["resume_quality_score"] <= 100
    assert "missing_keyword_severity" in result
    assert "critical" in result["missing_keyword_severity"]
    assert "important" in result["missing_keyword_severity"]
    assert "optional" in result["missing_keyword_severity"]
    assert result["scoring_version"] == "ats-scoring-v2"
    assert result["embedding_model"] == "BAAI/bge-large-en-v1.5"


def test_semantic_scorer_runs_without_model():
    """When Sentence Transformers is unavailable, semantic falls back to TF-IDF."""
    from modules.resume_ats.scoring.semantic_scorer import SemanticScorer

    jd = load_jd("AI_ML_ENGINEER")
    score = SemanticScorer().score(
        "Python PyTorch machine learning engineer",
        jd,
    )
    assert 0 <= score <= 100


def test_missing_keyword_severity_classification():
    """Verify CRITICAL/IMPORTANT/OPTIONAL severity tiers for missing keywords."""
    resume: StructuredResume = {
        "profile": {"summary": "Python developer"},
        "education": [{"degree": "B.Tech CS"}],
        "experience": [],
        "projects": [],
        "skills": ["Python"],
        "certifications": [],
        "publications": [],
        "achievements": [],
    }
    entities: ResumeEntities = {
        "skills": ["Python"],
        "tools": [],
        "frameworks": [],
        "languages": ["python"],
        "degrees": ["B.Tech"],
        "companies": [],
        "projects": [],
        "research": [],
        "certifications": [],
        "experience_years": 0.0,
    }
    jd = load_jd("AI_ML_ENGINEER")
    result = ScoringOrchestrator().score(resume, entities, jd)

    severity = result["missing_keyword_severity"]
    # PyTorch and TensorFlow are required but missing → should be in critical
    assert len(severity["critical"]) > 0
    # There should be some missing across all tiers given a minimal resume
    total_missing = len(severity["critical"]) + len(severity["important"]) + len(severity["optional"])
    assert total_missing > 0


def test_domain_boost_applied_for_matching_role():
    """When classifier detects the correct domain, domain_boost should be positive."""
    resume: StructuredResume = {
        "profile": {"summary": "AI ML engineer with RAG and PyTorch"},
        "education": [{"degree": "B.Tech CS"}],
        "experience": [{"company": "AI Corp", "title": "ML Engineer", "start_date": "2021", "end_date": "2025"}],
        "projects": [{"name": "RAG Pipeline", "technologies": ["PyTorch", "LangChain", "Qdrant"]}],
        "skills": ["Python", "PyTorch", "TensorFlow", "LangChain", "RAG", "FastAPI", "Docker", "AWS", "Transformers"],
        "certifications": [],
        "publications": [],
        "achievements": [],
    }
    entities: ResumeEntities = {
        "skills": ["Python", "PyTorch", "TensorFlow", "LangChain", "RAG", "FastAPI", "Docker", "AWS", "Transformers"],
        "tools": ["docker", "aws"],
        "frameworks": ["pytorch", "tensorflow", "fastapi", "langchain"],
        "languages": ["python"],
        "degrees": ["B.Tech"],
        "companies": ["AI Corp"],
        "projects": ["RAG Pipeline"],
        "research": [],
        "certifications": [],
        "experience_years": 4.0,
    }
    jd = load_jd("AI_ML_ENGINEER")
    result = ScoringOrchestrator().score(resume, entities, jd)

    # With a strong AI/ML profile, domain classifier should detect AI_ML_ENGINEER
    # and apply a domain boost
    assert result["detected_domain"] == "AI_ML_ENGINEER"
    assert result["domain_boost"] >= 0.0


def test_resume_quality_score_independent_of_jd():
    """Resume quality score should evaluate structure, not JD match."""
    resume_high_quality: StructuredResume = {
        "profile": {
            "name": "Jane Doe",
            "email": "jane@email.com",
            "summary": "Experienced ML engineer with 5 years building production ML systems at scale.",
        },
        "education": [{"degree": "M.Tech CS", "school": "IIT Delhi"}],
        "experience": [{
            "company": "TechCorp",
            "title": "Senior ML Engineer",
            "start_date": "2019",
            "end_date": "2024",
            "bullets": [
                "Built production RAG pipeline serving 1M users, reducing latency by 40%",
                "Deployed ML models on AWS reducing inference cost by $500K annually",
                "Led team of 5 engineers to deliver GenAI platform",
            ],
        }],
        "projects": [{"name": "RAG System", "description": "Built enterprise RAG with 95% accuracy"}],
        "skills": ["Python", "PyTorch", "AWS", "Docker", "FastAPI", "LangChain"],
        "certifications": ["AWS Solutions Architect"],
        "publications": [],
        "achievements": ["Best Paper Award"],
    }

    resume_low_quality: StructuredResume = {
        "profile": {},
        "education": [],
        "experience": [],
        "projects": [],
        "skills": ["python"],
        "certifications": [],
        "publications": [],
        "achievements": [],
    }

    from modules.resume_ats.entities.extractor import EntityExtractor
    from modules.resume_ats.scoring.resume_quality_scorer import ResumeQualityScorer

    extractor = EntityExtractor()
    scorer = ResumeQualityScorer()

    entities_hq = extractor.extract(resume_high_quality)
    entities_lq = extractor.extract(resume_low_quality)

    hq_score = scorer.score(resume_high_quality, entities_hq)
    lq_score = scorer.score(resume_low_quality, entities_lq)

    assert hq_score > lq_score
    assert hq_score >= 50
    assert lq_score < 50


def test_consulting_resume_detects_synonyms_and_required_skills():
    resume: StructuredResume = {
        "profile": {
            "summary": "Business Analyst with strategy and manufacturing optimization experience. "
                       "Built an AI classification project and digital twin strategy recommendations."
        },
        "education": [{"degree": "B.Tech", "school": "IIT"}],
        "experience": [
            {
                "company": "Accenture",
                "title": "Business Analyst",
                "start_date": "2022",
                "end_date": "2024",
                "bullets": [
                    "Created market analysis and strategy recommendations for client engagements",
                    "Managed 200+ people during public events and led embassy negotiations",
                    "Delivered client presentations using MS Office and mentored junior students",
                    "Drove cost reduction and manufacturing optimization initiatives",
                ],
            }
        ],
        "projects": [
            {
                "name": "AI Classification Project",
                "description": "Built a Python classification workflow and digital twin strategy for manufacturing optimization",
                "technologies": ["Python", "AWS"],
            }
        ],
        "skills": ["Python", "AWS", "MATLAB", "MS Office", "Strategy", "Leadership"],
        "achievements": ["Head of Proshows", "Debate award winner", "Mentor for student teams"],
        "certifications": [],
        "publications": [],
    }

    extractor = EntityExtractor()
    entities = extractor.extract(resume)
    jd = load_jd("CONSULTING_STRATEGY")
    result = ScoringOrchestrator().score(resume, entities, jd)
    best_role = DomainClassifier().suggest_best_role(entities, resume=resume)["best_role"]

    assert best_role == "CONSULTING_STRATEGY"
    assert result["detected_domain"] == "CONSULTING_STRATEGY"
    assert result["domain_score"] >= 80
    assert result["skill_depth_score"] >= 40

    critical_missing = set(result["missing_keyword_severity"]["critical"])
    assert "Business Analysis" not in critical_missing
    assert "Problem Solving" not in critical_missing
    assert "Communication" not in critical_missing
    assert "Excel" not in critical_missing
    assert "PowerPoint" not in critical_missing

    optional_missing = set(result["missing_keyword_severity"]["optional"])
    assert "SQL" not in optional_missing


def test_weighted_overall_score_follows_visible_breakdown():
    scores = {
        "keyword_score": 20,
        "exact_keyword_score": 20,
        "domain_score": 20,
        "resume_quality_score": 100,
        "semantic_score": 80,
        "experience_score": 80,
        "skill_depth_score": 80,
    }

    breakdown = _build_score_breakdown(scores)
    overall = _build_weighted_overall_score(breakdown)

    assert breakdown["keywordMatch"]["score"] == 2
    assert breakdown["formatting"]["score"] == 10
    assert overall < 75
