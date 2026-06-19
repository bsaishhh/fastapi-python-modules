from __future__ import annotations

import re
from modules.resume_ats.contracts import ResumeEntities, RoleJD, StructuredResume
from modules.resume_ats.data.jd_loader import list_roles, load_jd
from modules.resume_ats.domain_classifier.iit_department_map import get_roles_for_department
from modules.resume_ats.scoring.skill_synonyms import collection_matches_skill, synonym_match, text_matches_skill

# High-fidelity Direct Signals & Target Keywords for all 16 Role Families
DIRECT_SIGNALS = {
    "AI_ML_ENGINEER": {
        "titles": [
            "machine learning", "ml engineer", "ml intern", "deep learning", "ai engineer", 
            "ai research", "nlp engineer", "computer vision", "llm engineer", "genai engineer",
            "applied scientist", "research scientist", "speech ai", "recommendation systems engineer"
        ],
        "keywords": [
            "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "transformers", "hugging face",
            "huggingface", "llm", "llms", "large language model", "large language models", "nlp", "natural language processing",
            "computer vision", "opencv", "cnn", "rnn", "lstm", "bert", "gpt", "rag", "retrieval augmented generation",
            "langchain", "llamaIndex", "vector database", "vector databases", "pinecone", "qdrant", "weaviate",
            "chromadb", "mlops", "mlflow", "kubeflow", "model deployment", "model training", "fine-tuning",
            "transfer learning", "deep learning", "reinforcement learning", "neural network", "neural networks"
        ]
    },
    "SOFTWARE_ENGINEER": {
        "titles": [
            "software development engineer", "software engineer", "sde", "backend developer", 
            "backend engineer", "frontend developer", "frontend engineer", "full stack developer", 
            "full stack engineer", "systems engineer", "platform engineer", "site reliability engineer", 
            "sre", "devops engineer", "cloud engineer", "infrastructure engineer", "security engineer",
            "application engineer", "qa engineer", "automation test engineer", "solutions engineer",
            "ios developer", "android developer", "mobile developer"
        ],
        "keywords": [
            "java", "c++", "golang", "rust", "javascript", "typescript", "python", "c#", "ruby", "scala",
            "spring boot", "django", "fastapi", "node.js", "nodejs", "react", "angular", "vue.js", "vue",
            "microservices", "system design", "rest api", "restful api", "graphql", "sql", "postgresql", "mysql",
            "mongodb", "redis", "docker", "kubernetes", "aws", "gcp", "azure", "ci/cd", "jenkins", "git",
            "data structures", "algorithms"
        ]
    },
    "DATA_ENGINEER": {
        "titles": [
            "data engineer", "analytics engineer", "etl developer", "data architect", 
            "data platform engineer", "data warehouse engineer"
        ],
        "keywords": [
            "apache spark", "spark", "hadoop", "apache airflow", "airflow", "dbt", "databricks", "snowflake",
            "bigquery", "redshift", "hive", "kafka", "etl", "elt", "data pipeline", "data pipelines",
            "data warehousing", "data warehouse", "data lake", "data lakes", "sql", "pyspark", "scala"
        ]
    },
    "QUANT_FINANCE": {
        "titles": [
            "quantitative researcher", "quantitative analyst", "quant researcher", "quant analyst",
            "quant developer", "quant trader", "algorithmic trader", "trading strategist", "financial engineer"
        ],
        "keywords": [
            "stochastic calculus", "quantitative finance", "algorithmic trading", "arbitrage", "derivatives pricing",
            "stochastic modeling", "time series analysis", "monte carlo", "options pricing", "black-scholes",
            "quantitative modeling", "statistical arbitrage", "high-frequency trading", "hft", "risk management",
            "portfolio optimization", "c++", "python", "r", "matlab", "math", "statistics", "probability"
        ]
    },
    "CONSULTING_STRATEGY": {
        "titles": [
            "consultant", "associate consultant", "management consultant", "strategy consultant",
            "technology consultant", "operations consultant", "business analyst", "strategy associate",
            "corporate strategy", "pmo analyst", "program manager", "transformation consultant"
        ],
        "keywords": [
            "problem-solving", "case interviews", "market research", "business analysis", "strategy formulation",
            "stakeholder management", "business case", "financial modeling", "excel", "powerpoint",
            "slide design", "process optimization", "operations improvement", "due diligence", "competitor analysis",
            "market entry", "growth strategy", "pmo", "project management"
        ]
    },
    "PRODUCT_DESIGN": {
        "titles": [
            "ux designer", "ui designer", "product designer", "interaction designer", 
            "ux researcher", "design system designer", "customer experience designer",
            "apm", "associate product manager", "product manager", "product owner", "product strategy"
        ],
        "keywords": [
            "figma", "sketch", "adobe xd", "wireframing", "wireframe", "wireframes", "prototyping", "prototype",
            "user research", "user journeys", "information architecture", "usability testing", "ui design",
            "ux design", "interaction design", "heuristic evaluation", "personas", "design systems",
            "product strategy", "feature prioritization", "roadmap", "product analytics"
        ]
    },
    "MECHANICAL_MANUFACTURING": {
        "titles": [
            "mechanical engineer", "mechanical design engineer", "product design engineer", "cad engineer",
            "cfd engineer", "fea engineer", "thermal engineer", "manufacturing engineer", "process engineer",
            "production engineer", "tool design engineer", "industrial engineer", "r&d engineer"
        ],
        "keywords": [
            "cad", "solidworks", "catia", "autocad", "fea", "finite element analysis", "cfd",
            "computational fluid dynamics", "ansys", "mechanical design", "thermodynamics", "heat transfer",
            "manufacturing processes", "cnc", "dfm", "design for manufacturing", "gd&t", "mechatronics"
        ]
    },
    "ELECTRICAL_ELECTRONICS": {
        "titles": [
            "electrical engineer", "electronics engineer", "embedded systems engineer", "firmware engineer",
            "vlsi engineer", "asic design engineer", "verification engineer", "fpga engineer", "rf engineer",
            "signal integrity engineer", "pcb design engineer", "power electronics engineer", "control systems engineer"
        ],
        "keywords": [
            "embedded systems", "firmware", "vlsi", "asic", "fpga", "pcb design", "altium", "microcontrollers",
            "verilog", "vhdl", "systemverilog", "analog design", "digital design", "rtos", "c", "assembly",
            "circuit design", "signal processing", "power electronics", "controls"
        ]
    },
    "AEROSPACE_DEFENCE": {
        "titles": [
            "aerospace engineer", "avionics engineer", "flight software engineer", "propulsion engineer",
            "gnc engineer", "aerodynamics engineer", "structures engineer", "flight test engineer",
            "satellite systems engineer", "spacecraft engineer", "defence systems engineer", "mission design engineer"
        ],
        "keywords": [
            "aerodynamics", "cfd", "flight dynamics", "gnc", "guidance navigation and control", "avionics",
            "propulsion", "rocket engine", "aerospace", "spacecraft", "satellite", "orbital mechanics",
            "flight testing", "matlab", "simulink", "defence systems"
        ]
    },
    "CORE_SCIENCE_RND": {
        "titles": [
            "research scientist", "computational scientist", "physicist", "chemist", 
            "materials scientist", "biotechnologist"
        ],
        "keywords": [
            "scientific research", "r&d", "rnd", "molecular dynamics", "quantum mechanics", "biotechnology",
            "materials science", "computational chemistry", "bioinformatics", "genomics", "crystallography",
            "spectroscopy", "data analysis", "numerical simulation"
        ]
    },
    "CIVIL_INFRASTRUCTURE": {
        "titles": [
            "civil engineer", "structural engineer", "geotechnical engineer", "transportation engineer",
            "construction engineer", "bim engineer", "bim modeler", "site engineer", "quantity surveyor"
        ],
        "keywords": [
            "structural design", "concrete", "steel structures", "geotechnical", "foundation engineering",
            "bim", "building information modeling", "autocad", "revit", "staad pro", "etabs", "construction management",
            "surveying", "infrastructure"
        ]
    },
    "ROBOTICS_AUTONOMOUS": {
        "titles": [
            "robotics engineer", "robotics software engineer", "perception engineer", "slam engineer",
            "autonomous systems engineer", "controls engineer", "motion planning engineer", "mechatronics engineer",
            "drone engineer", "automation engineer", "industrial robotics engineer", "robot simulation engineer"
        ],
        "keywords": [
            "ros", "robot operating system", "slam", "simultaneous localization and mapping", "lidar",
            "perception", "motion planning", "autonomous vehicles", "path planning", "computer vision",
            "robotics", "drones", "control systems", "pid controller", "c++", "python"
        ]
    },
    "FOUNDERS_OFFICE": {
        "titles": [
            "founder's office", "founders office", "chief of staff", "strategy & operations",
            "business operations associate", "growth associate", "growth manager", "venture analyst",
            "startup generalist", "ceo office", "partnerships associate", "expansion associate"
        ],
        "keywords": [
            "startup operations", "growth strategy", "business development", "fundraising", "pitch deck",
            "investor relations", "market expansion", "cross-functional leadership", "strategic initiatives",
            "gtm", "go-to-market", "operations management", "business metrics"
        ]
    },
    "EDUCATION_EDTECH": {
        "titles": [
            "faculty", "assistant professor", "lecturer", "subject matter expert", "sme",
            "curriculum designer", "instructional designer", "academic content developer", "jee faculty",
            "neet faculty", "coding instructor", "learning experience designer", "education consultant"
        ],
        "keywords": [
            "teaching", "curriculum development", "instructional design", "pedagogy", "academic writing",
            "e-learning", "lms", "student engagement", "tutoring", "education technology", "edtech"
        ]
    },
    "GAMING_GRAPHICS": {
        "titles": [
            "game developer", "gameplay programmer", "game engine engineer", "graphics engineer",
            "rendering engineer", "technical artist", "3d artist", "environment artist", "character artist",
            "game designer", "level designer", "ar/vr developer", "simulation engineer"
        ],
        "keywords": [
            "unity", "unreal engine", "c++", "c#", "opengl", "directx", "vulkan", "shader", "shaders",
            "graphics rendering", "ray tracing", "technical art", "3d modeling", "blender", "maya",
            "game physics", "ar/vr", "augmented reality", "virtual reality"
        ]
    },
    "SUPPLY_CHAIN_OPERATIONS": {
        "titles": [
            "supply chain analyst", "supply chain manager", "logistics analyst", "logistics coordinator",
            "procurement analyst", "procurement specialist", "operations analyst", "operations manager",
            "inventory planner", "demand planner", "warehouse operations manager", "manufacturing operations",
            "operations management trainee", "omt", "vendor management specialist"
        ],
        "keywords": [
            "supply chain", "supply chain management", "scm", "logistics", "procurement", "sourcing",
            "inventory management", "inventory control", "demand forecasting", "warehouse operations",
            "sap", "erp", "operations analytics", "six sigma", "lean manufacturing"
        ]
    }
}

GENERIC_DOMAIN_KEYWORDS = {
    "python", "java", "javascript", "typescript", "c++", "c", "r", "matlab",
    "sql", "aws", "gcp", "azure", "docker", "kubernetes", "git",
    "math", "statistics", "probability",
}


class DomainClassifier:
    """Classify resume domain relevance against role benchmark JDs."""

    def classify(self, entities: ResumeEntities, role: str, resume: StructuredResume | None = None) -> dict:
        jd = load_jd(role)
        if resume:
            evaluation = self._evaluate_career_fit(resume, role.upper(), jd, entities)
            domain_score = evaluation["fit_score"]
        else:
            domain_score = self._compute_domain_score(entities, jd)

        return {
            "role": role,
            "family": jd.get("family", ""),
            "domain_score": domain_score,
            "matched_required": self._matched(entities, jd.get("required_skills", [])),
            "matched_preferred": self._matched(entities, jd.get("preferred_skills", [])),
            "missing_critical": self._missing(entities, jd.get("keywords", [])),
        }

    def detect_from_department(self, department: str) -> dict:
        roles = get_roles_for_department(department)
        return {"department": department, "suggested_roles": roles}

    def suggest_best_role(self, entities: ResumeEntities, resume: StructuredResume | None = None) -> dict:
        if not resume:
            # Fallback to simple keyword match if resume is not provided
            scores: list[tuple[str, int]] = []
            for role_key in list_roles():
                jd = load_jd(role_key.upper())
                score = self._compute_domain_score(entities, jd)
                scores.append((role_key, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            best_role, best_score = scores[0] if scores else ("", 0)
            return {"best_role": best_role.upper(), "domain_score": best_score, "rankings": scores[:5]}

        rankings = []
        for role_key in list_roles():
            role_upper = role_key.upper()
            try:
                jd = load_jd(role_upper)
            except Exception:
                continue

            evaluation = self._evaluate_career_fit(resume, role_upper, jd, entities)
            rankings.append({
                "role_key": role_upper,
                "role_family": jd.get("family", role_upper),
                "sub_roles": jd.get("sub_roles", []),
                "fit_score": evaluation["fit_score"],
                "confidence_level": evaluation["confidence_level"],
                "matching_evidence": evaluation["matching_evidence"],
                "missing_evidence": evaluation["missing_evidence"],
                "reasoning": evaluation["reasoning"],
            })

        # Sort by actual career fit score descending
        rankings.sort(key=lambda x: x["fit_score"], reverse=True)

        best_role = rankings[0]["role_key"] if rankings else ""
        best_score = rankings[0]["fit_score"] if rankings else 0

        # Form rankings tuples for backward compatibility (list of (role_key, score))
        rankings_tuples = [(r["role_key"], r["fit_score"]) for r in rankings]

        # Generate Career Track Summary
        career_track_summary = self._generate_track_summary(rankings, resume)

        return {
            "best_role": best_role.upper(),
            "domain_score": best_score,
            "rankings": rankings_tuples,
            "top_10_best_fit_roles": rankings[:10],
            "career_track_summary": career_track_summary,
        }

    def _evaluate_career_fit(
        self,
        resume: StructuredResume,
        family_key: str,
        jd: RoleJD,
        entities: ResumeEntities
    ) -> dict:
        signals = DIRECT_SIGNALS.get(family_key.upper(), {
            "titles": [family_key.lower().replace("_", " ")],
            "keywords": [k.lower() for k in jd.get("keywords", [])]
        })

        target_titles = signals["titles"]
        target_keywords = signals["keywords"]
        specific_keywords = [kw for kw in target_keywords if kw.lower() not in GENERIC_DOMAIN_KEYWORDS]

        def contains_phrase(text: str, phrase: str) -> bool:
            return text_matches_skill(text, phrase)

        def contains_any_phrase(text: str, phrases: list[str]) -> bool:
            return any(text_matches_skill(text, p) for p in phrases)

        matching_evidence = []
        missing_evidence = []

        prof_exp_direct = []
        intern_direct = []
        project_direct = []
        bullets_direct = []
        skills_direct = []

        # Parse Professional Experience and Internships
        experiences = resume.get("experience", []) or []
        for exp in experiences:
            title = exp.get("title", "")
            company = exp.get("company", "")
            bullets = exp.get("bullets", []) or []
            desc = exp.get("description", "") or ""
            
            bullets_text = " ".join(bullets) + " " + desc
            
            is_internship = contains_any_phrase(title, ["intern", "trainee", "apprenticeship", "student", "fellow", "assistant"])
            title_match = contains_any_phrase(title, target_titles)
            matched_kws = [kw for kw in target_keywords if contains_phrase(bullets_text, kw)]
            specific_matched_kws = [kw for kw in specific_keywords if contains_phrase(bullets_text, kw)]

            if is_internship:
                if title_match:
                    intern_direct.append(exp)
                    matching_evidence.append(f"Internship as '{title}' at {company} directly aligns with {family_key}")
                elif specific_matched_kws:
                    matching_evidence.append(f"Internship as '{title}' at {company} involved relevant skills: {', '.join(specific_matched_kws[:3])}")
            else:
                if title_match:
                    prof_exp_direct.append(exp)
                    matching_evidence.append(f"Professional Experience as '{title}' at {company} directly aligns with {family_key}")
                elif specific_matched_kws:
                    matching_evidence.append(f"Professional Experience as '{title}' at {company} involved relevant skills: {', '.join(specific_matched_kws[:3])}")

            if specific_matched_kws:
                bullets_direct.extend(specific_matched_kws)

        # Parse Projects
        projects = resume.get("projects", []) or []
        for proj in projects:
            name = proj.get("name", "")
            desc = proj.get("description", "") or ""
            techs = proj.get("technologies", []) or []
            
            proj_text = name + " " + desc + " " + " ".join(techs)
            name_match = contains_any_phrase(name, target_titles)
            matched_kws = [kw for kw in target_keywords if contains_phrase(proj_text, kw)]
            specific_matched_kws = [kw for kw in specific_keywords if contains_phrase(proj_text, kw)]

            if name_match:
                project_direct.append(proj)
                matching_evidence.append(f"Project '{name}' directly aligns with {family_key}")
            elif specific_matched_kws:
                project_direct.append(proj)
                matching_evidence.append(f"Project '{name}' utilized relevant technologies/skills: {', '.join(specific_matched_kws[:3])}")

        # Parse Skills
        skills = resume.get("skills", []) or []
        matched_skills = [s for s in skills if contains_any_phrase(s, target_keywords) or contains_any_phrase(s, target_titles)]
        specific_matched_skills = [s for s in matched_skills if s.lower() not in GENERIC_DOMAIN_KEYWORDS]
        if matched_skills:
            skills_direct.extend(specific_matched_skills or matched_skills[:2])
            matching_evidence.append(f"Listed Skills: {', '.join((specific_matched_skills or matched_skills)[:5])}")

        # Parse Education
        education = resume.get("education", []) or []
        for edu in education:
            degree = edu.get("degree", "") or ""
            field = edu.get("field", "") or ""
            school = edu.get("school", "") or ""
            desc = edu.get("description", "") or ""
            
            edu_text = degree + " " + field + " " + desc
            matched_edu_kws = [kw for kw in target_keywords if contains_phrase(edu_text, kw)]
            if matched_edu_kws:
                matching_evidence.append(f"Relevant education/coursework in '{field or degree}' at {school}")

        # Classify Signal Strength
        has_strong_direct = False
        has_weak_direct = False
        has_transferable = False

        if len(prof_exp_direct) >= 1:
            has_strong_direct = True
        elif len(intern_direct) >= 2:
            has_strong_direct = True
        elif len(intern_direct) == 1 and (len(project_direct) >= 1 or len(bullets_direct) >= 2):
            has_strong_direct = True
        elif len(project_direct) >= 2 and len(skills_direct) >= 3:
            has_strong_direct = True
        elif len(intern_direct) >= 1 or len(project_direct) >= 1 or len(bullets_direct) >= 2:
            has_weak_direct = True
        elif len(skills_direct) >= 1 or len(bullets_direct) >= 1:
            has_transferable = True

        # Base scoring
        skill_score = min(18, len(skills_direct) * 4)
        project_score = min(24, len(project_direct) * 12)
        intern_score = min(24, len(intern_direct) * 16)
        exp_score = min(52, len(prof_exp_direct) * 40)

        # Trajectory Boost
        trajectory_boost = 0
        if experiences:
            latest_title = experiences[0].get("title", "").lower()
            if contains_any_phrase(latest_title, target_titles):
                trajectory_boost = 18

        base_fit_score = skill_score + project_score + intern_score + exp_score + trajectory_boost

        # Apply Caps strictly
        if not has_strong_direct and not has_weak_direct and not has_transferable:
            fit_score = min(55, int(base_fit_score))
            confidence_level = "Low"
        elif has_transferable and not has_strong_direct and not has_weak_direct:
            fit_score = min(50, int(base_fit_score))
            confidence_level = "Low"
        elif has_weak_direct and not has_strong_direct:
            fit_score = min(70, max(51, int(base_fit_score)))
            confidence_level = "Medium"
        else:
            fit_score = min(100, max(70, int(base_fit_score)))
            confidence_level = "High"

        # Check for missing critical skills
        required_skills = jd.get("required_skills", [])
        resume_text_corpus = self._resume_to_text_internal(resume)
        for rs in required_skills:
            if not text_matches_skill(resume_text_corpus, rs):
                missing_evidence.append(rs)

        # Build reasoning string
        if fit_score >= 70:
            reasoning = f"{', '.join(matching_evidence[:3])} strongly align with {family_key} roles."
        elif fit_score >= 50:
            reasoning = f"Transferable skills and moderate evidence ({', '.join(matching_evidence[:2])}) suggest alignment with {family_key} roles."
        else:
            reasoning = f"Limited evidence or only basic skill matches ({', '.join(matching_evidence[:2])}) for {family_key} roles."

        return {
            "fit_score": fit_score,
            "confidence_level": confidence_level,
            "matching_evidence": matching_evidence,
            "missing_evidence": missing_evidence,
            "reasoning": reasoning,
        }

    def _generate_track_summary(self, rankings: list[dict], resume: StructuredResume) -> str:
        if not rankings:
            return "No career matches identified from the candidate's resume."
            
        primary = rankings[0]
        secondaries = rankings[1:3]
        
        skills = ", ".join(resume.get("skills", [])[:5])
        
        # Calculate experience years
        exp_entries = resume.get("experience", []) or []
        
        latest_title = exp_entries[0].get("title", "candidate") if exp_entries else "candidate"
        latest_company = exp_entries[0].get("company", "") if exp_entries else ""
        at_company = f" at {latest_company}" if latest_company else ""

        summary = (
            f"The candidate is strongest in the {primary['role_family']} track (Fit Score: {primary['fit_score']}), "
            f"evidenced by their latest role as {latest_title}{at_company}. They list core competencies including {skills}. "
        )
        
        if secondaries:
            sec_names = [s["role_family"] for s in secondaries]
            summary += f"Furthermore, they show viable alternative career tracks in {' and '.join(sec_names)} based on transferable capabilities."
            
        return summary

    def _resume_to_text_internal(self, resume: StructuredResume) -> str:
        parts: list[str] = []
        profile = resume.get("profile", {})
        for key in ("name", "summary", "email"):
            if profile.get(key):
                parts.append(str(profile[key]))
        for section in ("education", "experience", "projects"):
            for entry in resume.get(section, []):
                parts.extend(str(v) for v in entry.values() if v)
        parts.extend(resume.get("skills", []))
        parts.extend(resume.get("certifications", []))
        parts.extend(resume.get("publications", []))
        parts.extend(resume.get("achievements", []))
        return " ".join(parts)

    def _compute_domain_score(self, entities: ResumeEntities, jd: RoleJD) -> int:
        resume_terms = {t.lower() for t in (
            entities["skills"] + entities["tools"] + entities["frameworks"] + entities["languages"]
        )}
        required = [k.lower() for k in jd.get("required_skills", [])]
        preferred = [k.lower() for k in jd.get("preferred_skills", [])]
        keywords = [k.lower() for k in jd.get("keywords", [])]

        req_hits = sum(1 for r in required if any(r in term or term in r for term in resume_terms))
        pref_hits = sum(1 for p in preferred if any(p in term or term in p for term in resume_terms))
        kw_hits = sum(1 for k in keywords if any(k in term or term in k for term in resume_terms))

        req_score = (req_hits / max(len(required), 1)) * 50
        pref_score = (pref_hits / max(len(preferred), 1)) * 30
        kw_score = (kw_hits / max(len(keywords), 1)) * 20
        return int(min(100, round(req_score + pref_score + kw_score)))

    def _matched(self, entities: ResumeEntities, terms: list[str]) -> list[str]:
        resume_terms = {t.lower() for t in (
            entities["skills"] + entities["tools"] + entities["frameworks"]
        )}
        return [
            t for t in terms
            if collection_matches_skill(resume_terms, t)
        ]

    def _missing(self, entities: ResumeEntities, terms: list[str]) -> list[str]:
        resume_terms = {t.lower() for t in (
            entities["skills"] + entities["tools"] + entities["frameworks"] + entities["languages"]
        )}
        missing = []
        for term in terms:
            if not collection_matches_skill(resume_terms, term):
                missing.append(term)
        return missing
