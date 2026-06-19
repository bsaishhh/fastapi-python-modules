# ATS Resume Intelligence Engine — DEPLOYMENT.md

---

## System Overview

The ATS engine accepts a resume PDF and a target role, then returns:
- **Deterministic scores** (Semantic, TF-IDF, BM25, Jaccard, Domain, Experience, Skill Depth)
- **LLM qualitative feedback** (Strengths, Weaknesses, Critical Missing Keywords, Action Items)
- **Best-fit subroles** within the selected role cluster
- **MongoDB persistence** of all results

---

## Full Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│  Client (PDF upload + role)                                          │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 1 — Extraction Layer                                           │
│  modules/resume_ats/extraction/pipeline.py                          │
│                                                                      │
│  PDF bytes → POST https://resume-builder.cantileverlabs.com/api/    │
│              parse-resume (Cantilever external API)                  │
│           → Returns StructuredResume JSON:                           │
│              {                                                        │
│                profile: { name, email, summary, ... }               │
│                experience: [{ title, company, bullets[], ... }]     │
│                projects: [{ name, description, technologies[] }]    │
│                skills: ["Python", "LangChain", "Vector DBs", ...]   │
│                education: [{ degree, school, field }]               │
│                certifications: [...]                                 │
│                achievements: [...]                                   │
│              }                                                        │
│                                                                      │
│  Saved to MongoDB collection: extraction                             │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ StructuredResume JSON
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 2 — Entity Extraction                                          │
│  modules/resume_ats/entities/extractor.py                           │
│                                                                      │
│  Scans the full resume JSON using regex + vocabulary dictionaries.   │
│  Produces a ResumeEntities dict:                                     │
│    languages:  ["python", "javascript", "sql"]                      │
│    frameworks: ["fastapi", "langchain", "pytorch"]                  │
│    tools:      ["docker", "git", "aws", "qdrant"]                   │
│    skills:     (all of the above merged + raw skills list)          │
│    degrees:    ["B.E.", "Computer Science"]                          │
│    experience_years: float                                           │
│                                                                      │
│  WHY: The Cantilever parser splits keywords across multiple arrays.  │
│  This step unifies them into one searchable entity set used by both  │
│  the deterministic scorers and the LLM for accurate keyword matching.│
└──────────────────────────┬───────────────────────────────────────────┘
                           │ StructuredResume + ResumeEntities
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 3 — JD Loading                                                 │
│  modules/resume_ats/data/jd_loader.py                               │
│  modules/resume_ats/data/jds/<role>.json                            │
│                                                                      │
│  Loads the pre-defined benchmark Job Description for the chosen role │
│  from a local JSON file. Each JD contains:                           │
│    role_key, family, sub_roles[]                                     │
│    required_skills[], preferred_skills[]                             │
│    tools[], frameworks[], responsibilities[], keywords[]             │
│                                                                      │
│  16 supported role clusters:                                         │
│    AI_ML_ENGINEER, SOFTWARE_ENGINEERING, DATA_ENGINEERING,          │
│    QUANT_FINANCE, CONSULTING_STRATEGY, PRODUCT_MANAGER,             │
│    MECHANICAL_ENGINEERING, ELECTRICAL_ELECTRONICS,                   │
│    AEROSPACE_DEFENCE, CORE_SCIENCE_RND, CIVIL_INFRASTRUCTURE,       │
│    ROBOTICS_AUTONOMOUS, FOUNDERS_OFFICE, EDUCATION_EDTECH,          │
│    GAMING_GRAPHICS, SUPPLY_CHAIN_OPERATIONS                         │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ RoleJD
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 4 — Text Formatting (resume_to_text)                           │
│  modules/resume_ats/scoring/utils.py :: resume_to_text()            │
│                                                                      │
│  Converts StructuredResume JSON → clean, section-labeled plain text: │
│                                                                      │
│    Summary:                                                          │
│    <profile summary>                                                 │
│                                                                      │
│    Experience:                                                       │
│    <title> at <company>                                              │
│    <bullet points>                                                   │
│                                                                      │
│    Projects:                                                         │
│    <project name> — <description>                                    │
│    Technologies: <tech list>                                         │
│                                                                      │
│    Technical Competencies:                                           │
│    Skills: <skills array>                                            │
│    Languages: <languages array>     ← pulled from EntityExtractor   │
│    Frameworks: <frameworks array>   ← pulled from EntityExtractor   │
│    Tools: <tools array>             ← pulled from EntityExtractor   │
│                                                                      │
│    Education: ...                                                    │
│    Certifications: ...                                               │
│                                                                      │
│  WHY: Raw JSON confuses embedding models and TF-IDF. Clean text is   │
│  what ATS systems process. The Technical Competencies block ensures  │
│  tools like Git, Docker, and Vector DBs are never missed.            │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ resume_text (str)  +  jd_text (str)
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 5 — Deterministic ATS Scoring                                  │
│  modules/resume_ats/scoring/orchestrator.py                         │
│                                                                      │
│  Seven independent scorers run in sequence:                          │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ SemanticScorer (weight: 40%)                                │    │
│  │ semantic_scorer.py                                          │    │
│  │                                                             │    │
│  │ Loads 0xnbk/nbk-ats-semantic-v1-en via sentence-transformers│    │
│  │ with manually constructed Pooling layer (bypasses missing   │    │
│  │ 1_Pooling/config.json). Falls back to all-MiniLM-L6-v2 if  │    │
│  │ the 0xnbk model has architecture errors.                    │    │
│  │                                                             │    │
│  │ Process:                                                    │    │
│  │   resume_text + jd_text → encode() → cosine similarity     │    │
│  │   → Ridge + MLPRegressor ensemble (from 0xnbk repo weights)│    │
│  │   → calibrated 0–100 score                                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ BM25Scorer (weight: 15%)                                    │    │
│  │ bm25_scorer.py                                              │    │
│  │ rank-bm25 lexical retrieval; treats JD keywords as query,   │    │
│  │ resume text as the document corpus.                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ TF-IDF Scorer (weight: 10%)                                 │    │
│  │ tfidf_scorer.py                                             │    │
│  │ scikit-learn TfidfVectorizer + cosine similarity between    │    │
│  │ resume text and full JD text.                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ DomainScorer (weight: 20%)                                  │    │
│  │ domain_scorer.py                                            │    │
│  │ Ratio of required_skills[] present in resume entities.      │    │
│  │ Includes alias map: Vector DBs = Vector Databases,          │    │
│  │ sklearn = scikit-learn, huggingface = Hugging Face, etc.    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ ExperienceScorer (weight: 10%)                              │    │
│  │ experience_scorer.py                                        │    │
│  │ Evaluates past job titles, impact verbs, seniority          │    │
│  │ signals, duration, and quantified achievements.             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ SkillDepthScorer (weight: 5%)                               │    │
│  │ skill_depth_scorer.py                                       │    │
│  │ Contextual usage depth — did the resume use the skill in    │    │
│  │ a project/job or just list it in a skills section?          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Diagnostic Scores (not in formula)                          │    │
│  │ JaccardScorer        — token set overlap ratio              │    │
│  │ ExactKeywordScorer   — hard keyword coverage                │    │
│  │ ResumeQualityScorer  — JD-independent resume quality        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Formula:                                                            │
│    overall = (semantic×0.40) + (bm25×0.15) + (tfidf×0.10)          │
│            + (domain×0.20) + (experience×0.10) + (skill_depth×0.05)│
│            + domain_boost (0–5 if classifier agrees with target)    │
│                                                                      │
│  Missing keywords classified into CRITICAL / IMPORTANT / OPTIONAL   │
│  using the same alias map (Git, Vector DBs, etc. all recognized).   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ scores dict
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 6 — LLM Qualitative Analysis                                   │
│  modules/resume_ats/analysis/llm_analyzer.py                        │
│                                                                      │
│  Input to LLM:                                                       │
│    System: ATS_SYSTEM_PROMPT (enhanced with precise evaluation      │
│             rules, domain-specific expectations, subrole mapping)   │
│    User:   Role: AI_ML_ENGINEER                                      │
│            Candidate Resume Text: <formatted resume_to_text()>      │
│                                                                      │
│            --- EXTRACTED TECHNICAL ENTITIES ---                      │
│            Languages: python, javascript                             │
│            Frameworks: fastapi, langchain, pytorch                   │
│            Tools: docker, git, qdrant, aws                           │
│                                                                      │
│            Master JD Text: <jd required/preferred/tools/keywords>   │
│                                                                      │
│            --- VALID SUBROLES ---                                    │
│            AI Engineer, ML Engineer, LLM Engineer, GenAI Engineer,  │
│            Research Scientist                                        │
│                                                                      │
│  Output (via Instructor structured extraction):                      │
│    best_fit_subroles: ["GenAI Engineer"]                            │
│    strengths: [...]  (evidence-based, domain-specific)              │
│    weaknesses: [...]  (focused on critical gaps)                    │
│    critical_missing_keywords: [...]  (only truly absent core skills)│
│    action_items: [...]  (specific, actionable improvements)         │
│                                                                      │
│  LLM Accuracy Enhancements:                                          │
│    • Evidence-based evaluation only (no hallucinations)             │
│    • Synonym-aware matching (ML ≡ Machine Learning)                │
│    • Depth assessment (years + complexity vs. surface knowledge)    │
│    • Domain-specific criteria per role (e.g., PyTorch for ML)       │
│    • Critical vs. Important vs. Optional keyword classification     │
│    • Precise subrole mapping to candidate's experience trajectory   │
│                                                                      │
│  LLM: OpenRouter (configurable model, default gpt-4o-mini)          │
│  Fallback: Rule-based analysis if OPENROUTER_API_KEY not set        │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ analysis dict
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 7 — MongoDB Persistence                                        │
│  modules/resume_ats/db/repository.py                                │
│                                                                      │
│  Collections:                                                        │
│    resume-new       → upload metadata (filename, hash, role)        │
│    extraction       → full StructuredResume JSON                    │
│    domain-analysis  → scores + LLM analysis + master JD snapshot   │
│                                                                      │
│  Returns: resume_id, extraction_id, analysis_id                     │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 8 — API Response                                               │
│                                                                      │
│  {                                                                   │
│    "overall_score": 74,          ← /100                             │
│    "semantic_score": 81,         ← /100                             │
│    "bm25_score": 68,             ← /100                             │
│    "keyword_score": 72,          ← /100 (TF-IDF)                   │
│    "jaccard_score": 61,          ← /100 (diagnostic only)          │
│    "domain_score": 80,           ← /100                             │
│    "experience_score": 65,       ← /100                             │
│    "skill_depth_score": 70,      ← /100                             │
│    "resume_quality_score": 78,   ← /100 (diagnostic only)          │
│    "domain_boost": 3.5,                                             │
│    "detected_domain": "AI_ML_ENGINEER",                             │
│    "best_fit_subroles": ["GenAI Engineer"],                         │
│    "strengths": [...],                                               │
│    "weaknesses": [...],                                              │
│    "critical_missing_keywords": [...],                               │
│    "action_items": [...],                                            │
│    "missing_keyword_severity": {                                     │
│      "critical": [...],                                              │
│      "important": [...],                                             │
│      "optional": [...]                                               │
│    },                                                                │
│    "scoring_version": "ats-scoring-v2",                             │
│    "embedding_model": "0xnbk/nbk-ats-semantic-v1-en"               │
│  }                                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Scoring Weights

| Scorer | Weight | Out of |
|--------|--------|--------|
| Semantic (embedding cosine) | 40% | 100 |
| Domain (required skill coverage) | 20% | 100 |
| BM25 (lexical retrieval) | 15% | 100 |
| TF-IDF (keyword cosine) | 10% | 100 |
| Experience (title/impact/duration) | 10% | 100 |
| Skill Depth (contextual usage) | 5% | 100 |
| Domain Boost | 0–5 bonus pts | — |
| Jaccard / Exact KW / Resume Quality | diagnostic | not in score |

---

## Embedding Model

| Property | Value |
|----------|-------|
| **Primary model** | `0xnbk/nbk-ats-semantic-v1-en` |
| **Loader** | `sentence-transformers` with manually constructed `Pooling` layer |
| **Why manual Pooling** | The HuggingFace repo has no `1_Pooling/config.json`; `SentenceTransformer(model_id)` crashes without it |
| **Fallback model** | `sentence-transformers/all-MiniLM-L6-v2` (if 0xnbk has architecture errors) |
| **Score calibration** | Raw cosine similarity → Ridge + MLPRegressor ensemble (weights from `ridge_weights.json`, `neural_weights.json` in the 0xnbk repo) |
| **Startup preload** | Both model and ensemble weights loaded eagerly in FastAPI `lifespan` |

---

## Keyword Synonym Map

The deterministic scorer recognizes these as equivalent:

| Resume says | JD says |
|-------------|---------|
| `Vector DBs` / `vectordb` | `Vector Databases` |
| `sklearn` | `scikit-learn` |
| `HuggingFace` / `hugging face` | `Hugging Face Transformers` |
| `ML` | `Machine Learning` |
| `DL` | `Deep Learning` |
| `AI` | `Artificial Intelligence` |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/resume/extract` | PDF → StructuredResume JSON → MongoDB |
| POST | `/api/v1/resume/feedback` | Full pipeline: scores + LLM feedback + global role-fit |
| POST | `/api/v1/resume/score` | Same as feedback |
| POST | `/api/v1/resume/scores` | Deterministic scores only (JSON body, no LLM, no DB write) |
| POST | `/api/v1/resume/analyze` | Analyze from pre-extracted JSON |
| POST | `/api/v1/resume/domain-detect` | Rule-based domain/role hints |
| GET | `/api/v1/resume/results` | Recent MongoDB analyses |
| **GET** | **`/api/v1/resume/roles/jds`** | **List all available JD role keys (AI_ML_ENGINEER, SOFTWARE_ENGINEER, etc.)** |
| **GET** | **`/api/v1/resume/roles/all`** | **List all roles with sub-roles from roles.json** |
| GET | `/` | HTML test UI |
| GET | `/health` | Health + config status |

---

## Environment Variables

```env
MONGO_URI=mongodb+srv://...
MONGO_DB_NAME=fulcrum-staging
OPENROUTER_API_KEY=sk-...
OPENROUTER_MODEL=openai/gpt-4o-mini
SBERT_MODEL=0xnbk/nbk-ats-semantic-v1-en
SCORING_VERSION=ats-scoring-v2
```

---

## Run Locally

```bash
pip install -r requirements.txt
python scripts/seed_jds.py
set PYTHONPATH=.
python -m uvicorn app.main:app --reload
```

Open **http://localhost:8000/** for the test UI.

---

## Global Role-Fit Evaluation (New Feature)

### Overview

Instead of only evaluating against the supplied role, the system now:

1. **Evaluates the candidate's resume against ALL 16 role families** in the system
2. **Ranks the top 5 best-fit roles** globally
3. **Assesses the supplied role** separately, with explicit feedback if it's not the optimal match
4. **Recommends primary and secondary roles** based on global ranking
5. **Persists all results** to MongoDB within the analysis document

### Output Format

Each `/feedback`, `/score`, or `/analyze` response now includes:

```json
{
  "supplied_role_fit_score": 65,
  "supplied_role_assessment": "Supplied role is not the optimal match; candidate aligns better with other domains.",
  "recommended_primary_role": "DATA_ENGINEERING",
  "recommended_secondary_roles": ["AI_ML_ENGINEER"],
  "top_5_best_fit_roles": [
    {
      "role_key": "DATA_ENGINEERING",
      "role_family": "Data Engineering & Analytics",
      "sub_roles": ["Data Engineer", "Analytics Engineer", "BI Developer"],
      "fit_score": 82,
      "reasoning": "Matched required: 8; Matched preferred: 5; Domain score: 82"
    },
    {
      "role_key": "AI_ML_ENGINEER",
      "role_family": "AI / ML / Data Science",
      "sub_roles": ["ML Engineer", "LLM Engineer", "GenAI Engineer"],
      "fit_score": 78,
      "reasoning": "Matched required: 7; Matched preferred: 4; Domain score: 78"
    },
    ...
  ]
}
```

### How It Works

**modules/resume_ats/services/resume_service.py** → `analyze_from_structured()`

1. Extract entities from resume (languages, frameworks, tools, skills)
2. Load the supplied JD
3. Run deterministic scorers + LLM analysis for supplied role
4. **NEW**: Invoke `domain_classifier.suggest_best_role(entities)` to score against ALL JDs
5. Generate top-5 ranked roles with reasoning
6. Compute `supplied_role_fit_score` vs. top scorer
7. Decide if supplied role is suboptimal (> 5 point gap in top-1 vs. supplied)
8. Merge all global-fit fields into analysis payload
9. **Persist to MongoDB domain-analysis collection**

### Accuracy & Precision

The global ranking uses:

- **Deterministic Domain Scoring**: Required skills (50%), Preferred skills (30%), Keywords (20%)
- **Synonym-aware matching**: Recognizes common aliases (ML ≡ Machine Learning, HuggingFace ≡ Hugging Face Transformers)
- **Exact and fuzzy matching**: Partial matches (e.g., "PyTorch" matches "torch") + exact matches
- **Evidence-based reasoning**: Reports matched required/preferred counts and raw domain score

### Frontend Display

The HTML test UI (`static/index.html`) displays:

1. **Supplied Role Assessment** — Pass/fail with fit score
2. **Recommended Primary Role** — Best global fit
3. **Recommended Secondary Roles** — Top 2-3 alternatives
4. **Top 5 Best-Fit Roles** — Full ranked list with reasoning

---

## Role Taxonomy (16 Clusters)

| Cluster Key | Subroles / Coverage |
|-------------|---------------------|
| `AI_ML_ENGINEER` | AI Engineer, ML Engineer, LLM Engineer, GenAI Engineer, Research Scientist |
| `SOFTWARE_ENGINEERING` | SDE, Backend, Frontend, Full-Stack, Mobile, Platform — all seniorities |
| `DATA_ENGINEERING` | Data Engineer, Analytics Engineer, BI Developer, Analytics Consultant |
| `QUANT_FINANCE` | Quant Researcher, Quant Developer, IB Analyst, Trader |
| `CONSULTING_STRATEGY` | Business Analyst, Associate Consultant, M&A, Knowledge Associate |
| `PRODUCT_MANAGER` | APM, PM, Product Engineer, UX/UI Designer |
| `MECHANICAL_ENGINEERING` | Design, CFD, FEA, Thermal, Process, Materials |
| `ELECTRICAL_ELECTRONICS` | VLSI/ASIC, Embedded, RF, Power Electronics, Firmware |
| `AEROSPACE_DEFENCE` | Avionics, Propulsion, GNC, Structures, Flight Test |
| `CORE_SCIENCE_RND` | Research Scientist, Research Engineer, Operations Research |
| `CIVIL_INFRASTRUCTURE` | Structural Design, Geotech, BIM |
| `ROBOTICS_AUTONOMOUS` | Robotics Software, Perception, Drone, Automation |
| `FOUNDERS_OFFICE` | Strategy Ops, Growth, CEO Office, Program Management |
| `EDUCATION_EDTECH` | Faculty, SME, JEE/NEET Trainer, Instructional Designer |
| `GAMING_GRAPHICS` | Game Developer, 3D Artist, Graphics Engineer |
| `SUPPLY_CHAIN_OPERATIONS` | SCM, Logistics, Procurement, Operations Management |

---

## API Usage Examples

### 1. Fetch All Available JD Roles (for frontend population)

```bash
curl -X GET http://localhost:8000/api/v1/resume/roles/jds
```

**Response:**
```json
{
  "jds": [
    "AI_ML_ENGINEER",
    "SOFTWARE_ENGINEER",
    "DATA_ENGINEERING",
    "QUANT_FINANCE",
    "CONSULTING_STRATEGY",
    "PRODUCT_MANAGER",
    "MECHANICAL_MANUFACTURING",
    "ELECTRICAL_ELECTRONICS",
    "AEROSPACE_DEFENCE",
    "CORE_SCIENCE_RND",
    "CIVIL_INFRASTRUCTURE",
    "ROBOTICS_AUTONOMOUS",
    "FOUNDERS_OFFICE",
    "EDUCATION_EDTECH",
    "GAMING_GRAPHICS",
    "SUPPLY_CHAIN_OPERATIONS"
  ]
}
```

### 2. Fetch All Roles with Sub-Roles

```bash
curl -X GET http://localhost:8000/api/v1/resume/roles/all
```

**Response:**
```json
{
  "roles": [
    {
      "role": "Software Development Engineer",
      "sub_roles": ["SDE I", "SDE II", "SDE III"]
    },
    {
      "role": "Backend Engineer",
      "sub_roles": []
    },
    ...
  ],
  "count": 250
}
```

### 3. Full Pipeline Analysis with Global Role-Fit

```bash
curl -X POST http://localhost:8000/api/v1/resume/feedback \
  -F "resume_file=@sample.pdf" \
  -F "role=AI_ML_ENGINEER"
```

**Response includes:**
```json
{
  "overall_score": 82,
  "supplied_role_fit_score": 82,
  "supplied_role_assessment": "Supplied role is a reasonable match.",
  "recommended_primary_role": "AI_ML_ENGINEER",
  "recommended_secondary_roles": ["DATA_ENGINEERING"],
  "top_5_best_fit_roles": [
    {
      "role_key": "AI_ML_ENGINEER",
      "role_family": "AI / ML / Data Science",
      "sub_roles": ["AI Engineer", "ML Engineer", "LLM Engineer"],
      "fit_score": 82,
      "reasoning": "Matched required: 9; Matched preferred: 6; Domain score: 82"
    },
    {
      "role_key": "DATA_ENGINEERING",
      "role_family": "Data Engineering & Analytics",
      "sub_roles": ["Data Engineer", "Analytics Engineer"],
      "fit_score": 71,
      "reasoning": "Matched required: 7; Matched preferred: 4; Domain score: 71"
    },
    ...
  ],
  "strengths": [
    "5+ years Python experience with ML production projects",
    "Strong PyTorch and LangChain expertise",
    "Docker and AWS deployment experience"
  ],
  "weaknesses": [
    "No mentioned Kubernetes orchestration experience",
    "Limited distributed ML training background"
  ],
  "critical_missing_keywords": [],
  "best_fit_subroles": ["ML Engineer", "GenAI Engineer"],
  "action_items": [
    "Consider adding Kubernetes experience (e.g., CKA certification)",
    "Explore distributed training frameworks (Horovod, Ray)"
  ]
}
```

---

## MongoDB Collections

| Collection | Contents |
|------------|----------|
| `resume-new` | Upload metadata: filename, content hash, role, timestamp |
| `extraction` | Full StructuredResume JSON from Cantilever parser |
| `domain-analysis` | Scores dict + LLM analysis dict + master JD snapshot |
