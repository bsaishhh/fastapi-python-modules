import pytest
from modules.resume_ats.contracts import ResumeEntities, StructuredResume
from modules.resume_ats.domain_classifier.classifier import DomainClassifier
from modules.resume_ats.data.jd_loader import load_jd

def test_recommendation_engine_score_caps_and_trajectory():
    classifier = DomainClassifier()
    entities: ResumeEntities = {
        "skills": ["Python", "Pandas", "SQL", "Excel"],
        "tools": ["excel"],
        "frameworks": ["pandas"],
        "languages": ["python"],
        "degrees": ["B.Tech CS"],
        "companies": [],
        "projects": [],
        "research": [],
        "certifications": [],
        "experience_years": 1.0,
    }

    # Case 1: No direct evidence for Aerospace, only Python skill
    # Aerospace should be capped at 50 (only transferable skills) or 55
    resume_no_evidence: StructuredResume = {
        "profile": {"summary": "Analytical developer with Python and SQL"},
        "education": [{"degree": "B.Tech CS", "school": "IIT"}],
        "experience": [{"company": "DataCorp", "title": "Data Analyst", "start_date": "2023", "end_date": "2024"}],
        "projects": [],
        "skills": ["Python", "Pandas", "SQL", "Excel"],
    }
    
    result_aerospace = classifier.suggest_best_role(entities, resume=resume_no_evidence)
    rankings_dict = {r["role_key"]: r for r in result_aerospace["top_10_best_fit_roles"]}
    
    # Verify cap for Aerospace is <= 50 (since no direct signals like CFD, aerodynamics exist)
    if "AEROSPACE_DEFENCE" in rankings_dict:
        assert rankings_dict["AEROSPACE_DEFENCE"]["fit_score"] <= 50
        assert rankings_dict["AEROSPACE_DEFENCE"]["confidence_level"] == "Low"

    # Case 2: Weak Direct Evidence (1 minor internship with direct titles/signals)
    # Consulting should be capped at 70
    resume_weak_evidence: StructuredResume = {
        "profile": {"summary": "Junior analyst"},
        "education": [{"degree": "B.Tech", "school": "IIT"}],
        "experience": [{"company": "ConsultingCorp", "title": "Business Analyst Intern", "start_date": "2023", "end_date": "2023"}],
        "projects": [],
        "skills": ["Excel", "PowerPoint", "Problem Solving"],
    }
    
    result_weak = classifier.suggest_best_role(entities, resume=resume_weak_evidence)
    rankings_weak = {r["role_key"]: r for r in result_weak["top_10_best_fit_roles"]}
    
    if "CONSULTING_STRATEGY" in rankings_weak:
        assert rankings_weak["CONSULTING_STRATEGY"]["fit_score"] <= 70
        assert rankings_weak["CONSULTING_STRATEGY"]["confidence_level"] == "Medium"

    # Case 3: Strong Direct Evidence (1+ full-time professional experience matching title directly)
    # AI_ML should be allowed to go above 70
    resume_strong_evidence: StructuredResume = {
        "profile": {"summary": "ML Engineer"},
        "education": [{"degree": "B.Tech CS", "school": "IIT"}],
        "experience": [{"company": "AICorp", "title": "Machine Learning Engineer", "start_date": "2021", "end_date": "2024"}],
        "projects": [{"name": "RAG Chatbot", "description": "Built RAG systems with PyTorch and LangChain", "technologies": ["PyTorch", "LangChain"]}],
        "skills": ["Python", "PyTorch", "TensorFlow", "LangChain", "Docker"],
    }
    
    entities_strong = {
        **entities,
        "skills": ["Python", "PyTorch", "TensorFlow", "LangChain", "Docker"],
        "frameworks": ["pytorch", "tensorflow", "langchain"],
    }
    
    result_strong = classifier.suggest_best_role(entities_strong, resume=resume_strong_evidence)
    rankings_strong = {r["role_key"]: r for r in result_strong["top_10_best_fit_roles"]}
    
    assert result_strong["best_role"] == "AI_ML_ENGINEER"
    assert rankings_strong["AI_ML_ENGINEER"]["fit_score"] >= 70
    assert rankings_strong["AI_ML_ENGINEER"]["confidence_level"] == "High"
    assert len(rankings_strong["AI_ML_ENGINEER"]["matching_evidence"]) > 0


def test_career_trajectory_boost():
    classifier = DomainClassifier()
    entities: ResumeEntities = {
        "skills": ["Figma", "Roadmap", "Wireframing", "Excel"],
        "tools": ["figma"],
        "frameworks": [],
        "languages": [],
        "degrees": [],
        "companies": [],
        "projects": [],
        "research": [],
        "certifications": [],
        "experience_years": 2.0,
    }
    
    # Candidate whose latest experience was "Associate Product Manager"
    resume_pm: StructuredResume = {
        "profile": {"summary": "Product person"},
        "education": [],
        "experience": [
            {"company": "Startup", "title": "Associate Product Manager", "start_date": "2022", "end_date": "2024"},
            {"company": "Corp", "title": "Business Analyst", "start_date": "2020", "end_date": "2022"},
        ],
        "projects": [],
        "skills": ["Figma", "Roadmap", "Wireframing", "Excel"],
    }
    
    result = classifier.suggest_best_role(entities, resume=resume_pm)
    assert result["best_role"] == "PRODUCT_DESIGN" or result["best_role"] == "CONSULTING_STRATEGY"
    
    # Verify that details like career_track_summary and top_10_best_fit_roles are returned
    assert "career_track_summary" in result
    assert len(result["top_10_best_fit_roles"]) == 10
    
    # Ensure evidence details are present in the response
    first_role = result["top_10_best_fit_roles"][0]
    assert "matching_evidence" in first_role
    assert "missing_evidence" in first_role
    assert "confidence_level" in first_role
