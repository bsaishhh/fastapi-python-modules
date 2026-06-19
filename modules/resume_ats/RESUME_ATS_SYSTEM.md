# Resume ATS System — Architecture & Flow

> **Module:** `modules/resume_ats/`  
> **Base URL:** `POST /api/v1/resume/...`  
> **Parser Backend:** Cantilever Labs Resume Builder (`resume-builder.cantileverlabs.com`)

---

## 1. High-Level Architecture

```
┌──────────────┐     PDF      ┌───────────────────┐   multipart    ┌──────────────────────────────┐
│   Client     │─────────────▶│  FastAPI Routes    │──────────────▶│  resume-builder.cantilever   │
│  (Frontend)  │              │  /api/v1/resume/*  │   + JWT Auth  │  labs.com/api/parse-resume   │
└──────────────┘              └────────┬───────────┘               └──────────────┬───────────────┘
       ▲                               │                                          │
       │  JSON response                │  async/await                   rawResumeData JSON
       │                               ▼                                          │
       │                      ┌───────────────────┐                               │
       │                      │  ExtractService    │◀──────────────────────────────┘
       │                      │  + Pipeline        │
       │                      └────────┬───────────┘
       │                               │
       │                               ▼
       │                      ┌───────────────────┐
       │                      │  MongoDB Atlas     │
       │                      │  (motor async)     │
       │                      └───────────────────┘
       │                               │
       └───────────────────────────────┘
```

---

## 2. API Endpoints

| Method | Path | Description | Runs Extraction? | Runs LLM? |
|--------|------|-------------|:---:|:---:|
| `POST` | `/api/v1/resume/extract` | Upload PDF → structured JSON | ✅ | ❌ |
| `POST` | `/api/v1/resume/score` | Upload PDF → ATS analysis | ✅ | ✅ |
| `POST` | `/api/v1/resume/feedback` | Upload PDF → improvement feedback | ✅ | ✅ |
| `POST` | `/api/v1/resume/analyze` | Pre-extracted JSON → ATS analysis | ❌ | ✅ |
| `POST` | `/api/v1/resume/domain-detect` | Upload PDF → detect best role | ✅ | ❌ |
| `GET`  | `/api/v1/resume/results` | List recent analyses | ❌ | ❌ |

---

## 3. Complete Request Flow (Extract Endpoint)

### Step-by-step for `POST /api/v1/resume/extract`

```
Client                      extract.py                ExtractService           ExtractionPipeline         External API           MongoDB
  │                            │                           │                         │                       │                    │
  │  POST /extract             │                           │                         │                       │                    │
  │  (PDF file)                │                           │                         │                       │                    │
  │───────────────────────────▶│                           │                         │                       │                    │
  │                            │  validate PDF             │                         │                       │                    │
  │                            │  read file bytes          │                         │                       │                    │
  │                            │                           │                         │                       │                    │
  │                            │  extract_and_save()       │                         │                       │                    │
  │                            │──────────────────────────▶│                         │                       │                    │
  │                            │                           │                         │                       │                    │
  │                            │                           │  pipeline.run(bytes)    │                       │                    │
  │                            │                           │────────────────────────▶│                       │                    │
  │                            │                           │                         │                       │                    │
  │                            │                           │                         │  1. Generate JWT      │                    │
  │                            │                           │                         │  (HMAC-SHA256)        │                    │
  │                            │                           │                         │                       │                    │
  │                            │                           │                         │  2. POST /api/parse   │                    │
  │                            │                           │                         │  resume (PDF + JWT)   │                    │
  │                            │                           │                         │──────────────────────▶│                    │
  │                            │                           │                         │                       │                    │
  │                            │                           │                         │                       │ 3. LlamaParse      │
  │                            │                           │                         │                       │    (serverless)    │
  │                            │                           │                         │                       │ 4. AI Structuring  │
  │                            │                           │                         │                       │    pipeline        │
  │                            │                           │                         │                       │                    │
  │                            │                           │                         │  rawResumeData JSON   │                    │
  │                            │                           │                         │◀──────────────────────│                    │
  │                            │                           │                         │                       │                    │
  │                            │                           │                         │  5. Map to            │                    │
  │                            │                           │                         │  StructuredResume     │                    │
  │                            │                           │                         │                       │                    │
  │                            │                           │  StructuredResume       │                       │                    │
  │                            │                           │◀────────────────────────│                       │                    │
  │                            │                           │                         │                       │                    │
  │                            │                           │  6. SHA-256 hash        │                       │                    │
  │                            │                           │  7. save_resume_upload  │                       │                    │
  │                            │                           │────────────────────────────────────────────────────────────────────▶│
  │                            │                           │                         │                       │    resume_id       │
  │                            │                           │◀────────────────────────────────────────────────────────────────────│
  │                            │                           │                         │                       │                    │
  │                            │                           │  8. save_extraction     │                       │                    │
  │                            │                           │────────────────────────────────────────────────────────────────────▶│
  │                            │                           │                         │                       │  extraction_id     │
  │                            │                           │◀────────────────────────────────────────────────────────────────────│
  │                            │                           │                         │                       │                    │
  │                            │  result dict              │                         │                       │                    │
  │                            │◀──────────────────────────│                         │                       │                    │
  │                            │                           │                         │                       │                    │
  │  ExtractResponse JSON      │                           │                         │                       │                    │
  │◀───────────────────────────│                           │                         │                       │                    │
```

---

## 4. Extraction Pipeline — Detailed Stages

The pipeline (`extraction/pipeline.py`) is the core of resume parsing. It delegates to the **Cantilever Labs Resume Builder API** and then maps the response.

### Stage A: JWT Authentication

```python
# Generates a JWT token using Python stdlib (no PyJWT needed)
token = _make_jwt(user_id, jwt_secret, expires_in=86400)
# → eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI2OTI2...
```

- Uses **HMAC-SHA256** signing
- Token is **cached** on the pipeline instance (generated once, reused)
- Payload: `{ userId, iat, exp }`

### Stage B: External API Call

```python
POST https://resume-builder.cantileverlabs.com/api/parse-resume
Headers:  Authorization: Bearer <jwt>
Body:     multipart/form-data  { resume: <pdf_file> }
```

**What happens inside the Resume Builder API:**
1. **File validation** — checks PDF/Word mime type
2. **Cache lookup** — SHA-256 hash check, returns cached result if seen before
3. **Primary parse** — LlamaParse serverless Lambda (cloud-hosted)
4. **AI structuring** — converts raw text → structured JSON using AI
5. **Thumbnail generation** — S3 screenshot of page 1
6. **MongoDB persistence** — saves resume record
7. **Response** — returns `rawResumeData` + `enhancedResumeData` + metadata

**Timeout:** 300 seconds (5 minutes)

### Stage C: Response Mapping

The API returns `rawResumeData` in this format:

```json
{
  "personalDetails": { "name": "...", "email": "...", "phoneNumber": "...", "address": "..." },
  "jobRole": "Backend Developer",
  "areasOfInterest": "...",
  "skills": { "languages": "...", "frameworks": "...", "developerTools": "..." },
  "experience": [{ "company": "...", "role": "...", "dates": "...", "description": "..." }],
  "educationDetails": [{ "degree": "...", "college": "...", "duration": "...", "grades": "..." }],
  "projects": [{ "title": "...", "technologies": "...", "duration": "...", "description": "..." }],
  "accomplishments": [{ "title": "...", "organization": "...", "description": "..." }],
  "certifications": [],
  "publications": [],
  "extracurriculars": [{ "title": "...", "organization": "...", "date": "..." }]
}
```

The pipeline maps this to our internal **StructuredResume** format:

| Source Field | → | Target Field | Transform |
|---|---|---|---|
| `personalDetails.name` | → | `profile.name` | direct |
| `personalDetails.email` | → | `profile.email` | direct |
| `personalDetails.phoneNumber` | → | `profile.phone` | direct |
| `personalDetails.address` | → | `profile.location` | direct |
| `areasOfInterest` / `jobRole` | → | `profile.summary` | fallback chain |
| `educationDetails[].college` | → | `education[].school` | direct |
| `educationDetails[].duration` | → | `education[].start_date / end_date` | split on ` - ` |
| `educationDetails[].grades` | → | `education[].gpa` | direct |
| `experience[].company` | → | `experience[].company` | direct |
| `experience[].role` | → | `experience[].title` | direct |
| `experience[].dates` | → | `experience[].start_date / end_date` | split on ` - ` |
| `experience[].description` | → | `experience[].bullets` | sentence-split or bullet-split |
| `projects[].title` | → | `projects[].name` | direct |
| `projects[].technologies` | → | `projects[].technologies` | split on `, ; \|` → list |
| `skills` (dict) | → | `skills` (flat list) | flatten all values, split on `,` |
| `accomplishments[]` | → | `achievements[]` | join title — org — desc |
| `extracurriculars[]` | → | `achievements[]` | join title — org — date |
| `certifications[]` | → | `certifications[]` | flatten |
| `publications[]` | → | `publications[]` | flatten |

---

## 5. Internal Data Model (StructuredResume)

Defined in `contracts.py` as TypedDicts:

```python
class Profile:
    name, email, phone, location, summary, linkedin, github: str

class EducationEntry:
    school, degree, field, start_date, end_date, gpa, description: str

class ExperienceEntry:
    company, title, location, start_date, end_date, description: str
    bullets: list[str]

class ProjectEntry:
    name, description: str
    technologies: list[str]
    url: str

class StructuredResume:
    profile: Profile
    education: list[EducationEntry]
    experience: list[ExperienceEntry]
    projects: list[ProjectEntry]
    skills: list[str]
    certifications: list[str]
    publications: list[str]
    achievements: list[str]
```

---

## 6. Service Layer

### ExtractService (`services/resume_service.py`)

Used by `/extract`, `/score`, `/feedback`:

```
extract_and_save(file_bytes, filename, role?)
  ├── pipeline.run(file_bytes)           → StructuredResume
  ├── SHA-256(file_bytes)                → content_hash
  ├── repo.save_resume_upload(...)       → resume_id     (MongoDB)
  └── repo.save_extraction(...)          → extraction_id  (MongoDB)
```

### ResumeAnalysisService (`services/resume_service.py`)

Used by `/score`, `/feedback`, `/analyze`, `/domain-detect`:

```
analyze_from_file(file_bytes, filename, role)
  ├── ExtractService.extract_and_save()  → structured resume
  ├── load_jd(role)                      → master JD JSON
  ├── LLMAnalyzer.analyze()              → ATSAnalysis (strengths, weaknesses, etc.)
  └── repo.save_domain_analysis()        → analysis_id

analyze_from_structured(role, structured_resume, ...)
  ├── load_jd(role)                      → master JD
  ├── LLMAnalyzer.analyze()              → ATSAnalysis
  └── repo.save_domain_analysis()        → analysis_id

domain_detect(file_bytes, role?, department?)
  ├── pipeline.run()                     → StructuredResume
  ├── EntityExtractor.extract()          → ResumeEntities
  └── DomainClassifier                   → role suggestion / classification
```

---

## 7. LLM Analysis Pipeline

### LLMAnalyzer (`analysis/llm_analyzer.py`)

- Uses **OpenRouter** API (OpenAI-compatible) via `instructor` for structured output
- Model: configurable (default `openai/gpt-4o-mini`)
- Output schema: `ATSAnalysis` (Pydantic) with `strengths`, `weaknesses`, `critical_missing_keywords`, `action_items`
- Fallback: keyword-matching analysis if API key is missing

### EntityExtractor (`entities/extractor.py`)

NER-style extraction from structured resume:
- Tokenizes all resume text
- Classifies tokens against known sets: `PROGRAMMING_LANGUAGES`, `FRAMEWORKS`, `TOOLS`
- Regex patterns for degree detection
- Estimates total experience years from date ranges

### DomainClassifier (`domain_classifier/classifier.py`)

- Computes domain relevance score (0-100) against role benchmark JDs
- Score breakdown: required skills (50%) + preferred skills (30%) + keywords (20%)
- Can suggest best-fit role across all known roles
- Supports department → role mapping (IIT department names)

---

## 8. Database Layer

### MongoDB Collections (Atlas)

| Collection | Config Key | Purpose |
|---|---|---|
| `resume-new` | `COLLECTION_RESUME` | Upload metadata (filename, hash, role) |
| `extraction` | `COLLECTION_EXTRACTION` | Structured resume JSON |
| `domain-analysis` | `COLLECTION_DOMAIN_ANALYSIS` | LLM analysis results |

### Repository (`db/repository.py`)

All methods are **async** (motor driver) with `PyMongoError` handling → `AppError(503)`:

| Method | Collection | Returns |
|---|---|---|
| `save_resume_upload()` | `resume-new` | `resume_id` |
| `save_extraction()` | `extraction` | `extraction_id` |
| `save_domain_analysis()` | `domain-analysis` | `analysis_id` |
| `get_extraction()` | `extraction` | document |
| `list_recent()` | `domain-analysis` | last N docs |

---

## 9. Configuration

All config via `.env` file (Pydantic Settings):

```env
# ── MongoDB ──────────────────────────────────────
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/db
MONGO_DB_NAME=fulcrum-staging

# ── Resume Parser (External API) ────────────────
RESUME_PARSER_URL=https://resume-builder.cantileverlabs.com/api/parse-resume
RESUME_PARSER_USER_ID=69269673bcb6764c9f0f1770
RESUME_PARSER_JWT_SECRET=wienrceiworcm8eicewomc392c248394n8c493

# ── LLM Analysis ────────────────────────────────
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openai/gpt-4o-mini

# ── Versioning ──────────────────────────────────
EXTRACTION_VERSION=cantilever-parser-v1
ANALYSIS_VERSION=llm-v1
```

---

## 10. Replacing Hardcoded User Values

The JWT auth for the external parser currently uses hardcoded values. Here are the options to make them dynamic:

### Option A: Per-User Token Passthrough (Recommended)

Pass the user's own JWT from the frontend through a header:

```python
# In extract.py route handler:
@router.post("/extract", response_model=ExtractResponse)
async def extract_resume(
    resume_file: UploadFile = File(...),
    authorization: str = Header(None),          # ← accept user's token
) -> ExtractResponse:
    ...
    result = await extract_service.extract_and_save(
        file_bytes=file_bytes,
        filename=resume_file.filename,
        auth_token=authorization,               # ← pass to service
    )
```

Then in `pipeline.py`:

```python
async def run(self, file_bytes: bytes, filename: str, auth_token: str | None = None) -> StructuredResume:
    token = auth_token or self._get_auth_token()   # user token > service token
    ...
```

### Option B: Service Account with Login API

Instead of generating JWT manually, call the login API to get a fresh token:

```python
async def _login_and_get_token(self) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://resume-builder.cantileverlabs.com/api/auth/login",
            json={"email": settings.resume_parser_email, "password": settings.resume_parser_password}
        )
        return resp.json()["token"]
```

Add to `.env`:
```env
RESUME_PARSER_EMAIL=service@cantileverlabs.com
RESUME_PARSER_PASSWORD=...
```

### Option C: API Key Auth (If Backend Supports It)

Add a static API key middleware to the resume-builder backend, bypassing JWT entirely:

```python
headers = {"X-API-Key": settings.resume_parser_api_key}
```

---

## 11. Error Handling

| Scenario | HTTP Code | Response |
|---|---|---|
| Non-PDF file | 422 | `{"detail": "Only PDF resume files are supported"}` |
| Empty file | 422 | `{"detail": "Empty file uploaded"}` |
| Parser API timeout (>300s) | 504 | `{"detail": "Resume parser service timed out"}` |
| Parser API error | 502 | `{"detail": "Resume parser returned HTTP ..."}` |
| MongoDB connection failure | 503 | `{"detail": "Database error: ..."}` |

All errors use the `AppError` exception class with the global error handler in `app/core/exceptions.py`.

---

## 12. File Map

```
modules/resume_ats/
├── router.py                              # Aggregates all sub-routers under /resume
├── contracts.py                           # TypedDict data contracts (StructuredResume, etc.)
├── schemas/api.py                         # Pydantic request/response schemas
│
├── api/v1/                                # Route handlers
│   ├── extract.py                         # POST /extract
│   ├── score.py                           # POST /score
│   ├── analyze.py                         # POST /analyze
│   ├── feedback.py                        # POST /feedback
│   ├── domain_detect.py                   # POST /domain-detect
│   └── results.py                         # GET  /results
│
├── services/
│   └── resume_service.py                  # ExtractService + ResumeAnalysisService
│
├── extraction/
│   └── pipeline.py                        # External API call + response mapping
│
├── analysis/
│   └── llm_analyzer.py                    # LLM-based ATS analysis (OpenRouter)
│
├── entities/
│   └── extractor.py                       # NER entity extraction from structured resume
│
├── domain_classifier/
│   ├── classifier.py                      # Role/domain classification + scoring
│   └── iit_department_map.py              # IIT department → role mapping
│
├── data/
│   └── jd_loader.py                       # Loads benchmark job descriptions
│
└── db/
    └── repository.py                      # MongoDB async CRUD (motor)
```
