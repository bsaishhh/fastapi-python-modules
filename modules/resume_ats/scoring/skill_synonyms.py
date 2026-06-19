"""Skill synonym graph for ATS keyword expansion.

Maps canonical skill names to common aliases, tools, and related concepts
so that e.g. "Qdrant" contributes toward "Vector Database" score.
"""

from __future__ import annotations

from functools import lru_cache

# Canonical → [synonyms/aliases/related-tools]
# Keys must be lowercase; all lookups are case-insensitive.
SYNONYM_GRAPH: dict[str, list[str]] = {
    # ── AI / ML ─────────────────────────────────────────────────────
    "rag": [
        "retrieval augmented generation",
        "retrieval-augmented generation",
        "hybrid search",
        "vector search",
        "grounded generation",
    ],
    "llm": [
        "large language model",
        "llama", "gpt", "claude", "gemini", "mistral",
        "openai", "anthropic", "chatgpt", "copilot",
        "foundation model", "generative ai", "genai",
    ],
    "vector database": [
        "qdrant", "pinecone", "weaviate", "milvus", "chroma",
        "chromadb", "pgvector", "faiss", "annoy",
        "vector store", "vector db", "vector index",
    ],
    "deep learning": [
        "pytorch", "tensorflow", "keras", "jax",
        "neural network", "cnn", "rnn", "transformer",
        "dl", "deep neural",
    ],
    "machine learning": [
        "ml", "scikit-learn", "sklearn", "xgboost", "lightgbm",
        "random forest", "svm", "gradient boosting",
        "supervised learning", "unsupervised learning",
    ],
    "mlops": [
        "mlflow", "kubeflow", "weights and biases", "wandb",
        "dvc", "model registry", "model monitoring",
        "ml pipeline", "ml platform",
    ],
    "nlp": [
        "natural language processing", "spacy", "nltk", "stanfordnlp",
        "text mining", "text classification", "ner",
        "named entity recognition", "sentiment analysis",
    ],
    "computer vision": [
        "opencv", "yolo", "resnet", "object detection",
        "image segmentation", "cnn", "vision transformer", "vit",
    ],
    "langchain": [
        "langchain", "langflow", "langgraph",
        "llm orchestration", "chain framework",
    ],
    "transformers": [
        "hugging face", "huggingface", "bert", "gpt",
        "attention mechanism", "encoder decoder",
    ],
    "reinforcement learning": [
        "rl", "q-learning", "ppo", "openai gym", "stable baselines",
    ],
    # ── Cloud / DevOps ──────────────────────────────────────────────
    "aws": [
        "amazon web services", "ec2", "s3", "lambda", "sagemaker",
        "ecs", "eks", "cloudformation", "boto3",
    ],
    "gcp": [
        "google cloud", "google cloud platform", "bigquery",
        "vertex ai", "gke", "cloud run",
    ],
    "azure": [
        "microsoft azure", "azure ml", "azure functions", "aks",
    ],
    "kubernetes": [
        "k8s", "helm", "eks", "gke", "aks",
        "container orchestration", "kubectl",
    ],
    "docker": [
        "container", "containerization", "dockerfile",
        "docker-compose", "containerized",
    ],
    "ci/cd": [
        "ci cd", "continuous integration", "continuous deployment",
        "jenkins", "github actions", "gitlab ci", "circleci",
        "azure devops", "argocd",
    ],
    "terraform": [
        "iac", "infrastructure as code", "cloudformation",
        "pulumi", "ansible",
    ],
    # ── Data Engineering ────────────────────────────────────────────
    "etl": [
        "extract transform load", "data pipeline", "data ingestion",
        "elt", "data integration",
    ],
    "apache spark": [
        "spark", "pyspark", "databricks", "spark sql",
    ],
    "kafka": [
        "apache kafka", "confluent", "event streaming",
        "message queue", "pub sub",
    ],
    "airflow": [
        "apache airflow", "dag", "workflow orchestration",
        "prefect", "dagster",
    ],
    "data warehouse": [
        "snowflake", "redshift", "bigquery", "databricks",
        "data lake", "warehouse", "dwh",
    ],
    "dbt": [
        "data build tool", "analytics engineering",
        "sql transformation",
    ],
    # ── Databases ───────────────────────────────────────────────────
    "postgresql": [
        "postgres", "psql", "pg", "postgresql",
    ],
    "mongodb": [
        "mongo", "nosql", "document database", "mongoose",
    ],
    "redis": [
        "memcached", "in-memory cache", "caching",
    ],
    "elasticsearch": [
        "elastic search", "opensearch", "full text search",
        "search engine", "lucene",
    ],
    # ── Web / API ───────────────────────────────────────────────────
    "rest api": [
        "rest", "restful", "api", "http api", "web service",
        "flask api", "fastapi", "express",
    ],
    "graphql": [
        "gql", "apollo", "graph query",
    ],
    "microservices": [
        "microservice", "service mesh", "distributed system",
        "api gateway", "grpc",
    ],
    # ── Frontend ────────────────────────────────────────────────────
    "react": [
        "reactjs", "react.js", "nextjs", "next.js",
        "jsx", "react native",
    ],
    "vue": [
        "vuejs", "vue.js", "nuxt", "nuxtjs",
    ],
    "angular": [
        "angularjs", "ng", "rxjs",
    ],
    "typescript": [
        "ts", "tsx", "typed javascript",
    ],
    # ── Quantitative / Finance ──────────────────────────────────────
    "quantitative finance": [
        "quant", "financial modeling", "derivatives",
        "risk management", "stochastic calculus",
    ],
    # ── General ─────────────────────────────────────────────────────
    "git": [
        "version control", "github", "gitlab", "bitbucket",
    ],
    "agile": [
        "scrum", "kanban", "sprint", "jira", "agile methodology",
    ],
    "system design": [
        "distributed systems", "scalability", "load balancing",
        "caching", "database design", "architecture",
    ],
}


@lru_cache(maxsize=1)
def _build_reverse_index() -> dict[str, set[str]]:
    """Build reverse lookup: synonym → {canonical_skills}."""
    reverse: dict[str, set[str]] = {}
    for canonical, synonyms in SYNONYM_GRAPH.items():
        reverse.setdefault(canonical, set()).add(canonical)
        for syn in synonyms:
            reverse.setdefault(syn.lower(), set()).add(canonical)
    return reverse


def expand_skill(skill: str) -> set[str]:
    """Return the set of canonical skills that *skill* maps to (including itself)."""
    reverse = _build_reverse_index()
    lower = skill.lower()
    matched = reverse.get(lower, set())
    # Also check partial containment for multi-word synonyms
    if not matched:
        for syn_key, canonicals in reverse.items():
            if syn_key in lower or lower in syn_key:
                matched = matched | canonicals
    return matched | {lower}


def synonym_match(resume_term: str, jd_term: str) -> bool:
    """Return True if resume_term and jd_term share any canonical skill."""
    resume_canonicals = expand_skill(resume_term)
    jd_canonicals = expand_skill(jd_term)
    return bool(resume_canonicals & jd_canonicals)


def expand_skill_list(skills: list[str]) -> set[str]:
    """Expand a list of skills into all canonical forms."""
    expanded: set[str] = set()
    for skill in skills:
        expanded |= expand_skill(skill)
    return expanded
