def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "scoring" in data
    assert data["scoring"]["embedding_model"] == "BAAI/bge-large-en-v1.5"


def test_extract_endpoint(client, sample_resume_pdf):
    response = client.post(
        "/api/v1/resume/extract",
        files={"resume_file": ("resume.pdf", sample_resume_pdf, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "extracted_text" in data
    assert isinstance(data["extracted_text"], str)
    assert "resume_id" in data
    assert "extraction_id" in data


def test_score_endpoint(client, sample_resume_pdf):
    response = client.post(
        "/api/v1/resume/score",
        files={"resume_file": ("resume.pdf", sample_resume_pdf, "application/pdf")},
        data={"role": "AI_ML_ENGINEER"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "AI_ML_ENGINEER"
    assert "overall_score" in data
    assert "semantic_score" in data
    assert "keyword_score" in data
    assert "bm25_score" in data
    assert "jaccard_score" in data
    assert "exact_keyword_score" in data
    assert "domain_score" in data
    assert "strengths" in data
    assert "analysis_id" in data
    assert "analysis" in data
    assert "data" in data
    assert "areasOfImprovement" in data["analysis"]
    assert "criticalMissingKeywords" in data["data"]


def test_scores_only_endpoint(client):
    response = client.post(
        "/api/v1/resume/scores",
        json={
            "role": "AI_ML_ENGINEER",
            "extracted_resume": {
                "skills": ["Python", "PyTorch", "FastAPI"],
                "projects": [{"name": "RAG Chatbot", "technologies": ["LangChain"]}],
                "profile": {"summary": "Built retrieval augmented generation systems"},
                "education": [],
                "experience": [],
                "certifications": [],
                "publications": [],
                "achievements": [],
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["overall_score"] <= 100
    assert 0 <= data["semantic_score"] <= 100
    assert 0 <= data["jaccard_score"] <= 100
    assert data["embedding_model"] == "BAAI/bge-large-en-v1.5"


def test_feedback_endpoint(client, sample_resume_pdf):
    response = client.post(
        "/api/v1/resume/feedback",
        files={"resume_file": ("resume.pdf", sample_resume_pdf, "application/pdf")},
        data={"role": "AI_ML_ENGINEER"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "strengths" in data
    assert "overall_score" in data
    assert data["critical_missing_keywords"]


def test_analyze_json_endpoint(client):
    response = client.post(
        "/api/v1/resume/analyze",
        json={
            "role": "AI_ML_ENGINEER",
            "extracted_resume": "Python PyTorch FastAPI ATS Resume Analyzer",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["semantic_score"] >= 0
    assert "analysis" in data
    assert "data" in data
    assert "areasOfImprovement" in data["analysis"]


def test_domain_detect_endpoint(client, sample_resume_pdf):
    response = client.post(
        "/api/v1/resume/domain-detect",
        files={"resume_file": ("resume.pdf", sample_resume_pdf, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "best_role" in data or "domain_score" in data or "suggested_roles" in data


def test_results_endpoint(client):
    response = client.get("/api/v1/resume/results")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
