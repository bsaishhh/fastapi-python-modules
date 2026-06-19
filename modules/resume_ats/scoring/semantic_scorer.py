from __future__ import annotations

import logging
import sys
import types
import os
import glob
import json
from functools import lru_cache

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings
from modules.resume_ats.contracts import RoleJD, StructuredResume
from modules.resume_ats.scoring.tfidf_scorer import TfidfScorer
from modules.resume_ats.scoring.utils import (
    clamp_score,
    jd_to_text,
    jd_responsibilities_text,
    jd_skills_text,
    jd_degree_requirements_text,
    resume_experience_text,
    resume_project_text,
    resume_skills_text,
    resume_education_text,
)

logger = logging.getLogger(__name__)

# Section weights: how much each section-pair contributes to semantic score
SECTION_WEIGHTS = {
    "experience": 0.45,
    "projects":   0.25,
    "skills":     0.20,
    "education":  0.10,
}


def patch_transformers_compatibility():
    """Apply runtime patches to transformers/torch libraries to support older Jina-based architectures on newer versions."""
    # 1. Inject transformers.onnx mock module if missing (removed in newer transformers)
    try:
        import transformers.onnx
    except ModuleNotFoundError:
        logger.info("Injecting transformers.onnx mock module")
        onnx_mock = types.ModuleType("transformers.onnx")
        onnx_mock.OnnxConfig = object
        sys.modules["transformers.onnx"] = onnx_mock

    # 2. Inject find_pruneable_heads_and_indices in transformers.pytorch_utils if missing
    try:
        import torch
        import transformers.pytorch_utils
        if not hasattr(transformers.pytorch_utils, "find_pruneable_heads_and_indices"):
            logger.info("Patching find_pruneable_heads_and_indices helper in transformers.pytorch_utils")
            def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                prune_heads = set(heads) - already_pruned_heads
                indices = torch.tensor(
                    [i for h in prune_heads for i in range(h * head_size, (h + 1) * head_size)],
                    dtype=torch.long
                )
                return prune_heads, indices
            transformers.pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    except Exception as e:
        logger.warning("Failed to patch find_pruneable_heads_and_indices: %s", e)

    # 3. Patch PretrainedConfig to return defaults for missing properties
    try:
        from transformers import PretrainedConfig
        if not hasattr(PretrainedConfig, "_ats_patched"):
            logger.info("Patching PretrainedConfig attribute access for JinaBert compatibility")
            original_getattribute = PretrainedConfig.__getattribute__
            def safe_getattribute(self, key):
                try:
                    return original_getattribute(self, key)
                except AttributeError:
                    defaults = {
                        "is_decoder": False,
                        "add_cross_attention": False,
                        "chunk_size_feed_forward": 0,
                    }
                    if key in defaults:
                        return defaults[key]
                    raise
            PretrainedConfig.__getattribute__ = safe_getattribute
            PretrainedConfig._ats_patched = True
    except Exception as e:
        logger.warning("Failed to patch PretrainedConfig: %s", e)

    # 4. Patch modeling_bert.py on-disk to fix the device mismatch error (cpu vs meta device)
    try:
        base_cache = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
        cache_patterns = [
            os.path.join(base_cache, "*/*/modeling_bert.py"),
            os.path.join(base_cache, "*/*/*/modeling_bert.py")
        ]
        paths_to_check = []
        for p in cache_patterns:
            paths_to_check.extend(glob.glob(p))
            
        for path in paths_to_check:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            target = "alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position"
            replacement = "alibi = slopes.unsqueeze(1).unsqueeze(1).to(relative_position.device) * relative_position"
            if target in content:
                logger.info("Applying device compatibility patch to HuggingFace cached file: %s", path)
                content = content.replace(target, replacement)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
    except Exception as e:
        logger.warning("Failed to patch cached modeling_bert.py: %s", e)


# Apply compatibility patches immediately upon module import
patch_transformers_compatibility()


@lru_cache(maxsize=1)
def _load_embedding_model():
    """
    Load a sentence embedding model with multi-level fallback:
      1. Try 0xnbk/nbk-ats-semantic-v1-en via sentence-transformers with
         manually constructed Pooling (avoids 1_Pooling/config.json 404).
      2. Fall back to all-MiniLM-L6-v2 — stable, fast, always works.
    Returns a SentenceTransformer instance or None.
    """
    from sentence_transformers import SentenceTransformer, models

    model_id = settings.sbert_model

    # ── Attempt 1: 0xnbk model with manual Pooling ───────────────────────────
    try:
        logger.info("Attempting to load %s with manual Pooling", model_id)
        from transformers import AutoConfig

        # Determine embedding dimension from config
        try:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        except Exception:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        embedding_dim = getattr(cfg, "hidden_size", 768)

        # Build SentenceTransformer manually: Transformer + explicit Pooling
        try:
            word_embedding = models.Transformer(model_id, model_args={"trust_remote_code": True}, tokenizer_args={"trust_remote_code": True}, cache_dir=None)
        except Exception:
            word_embedding = models.Transformer(model_id, model_args={"trust_remote_code": True}, tokenizer_args={"trust_remote_code": True})

        pooling = models.Pooling(word_embedding.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
        st_model = SentenceTransformer(modules=[word_embedding, pooling])

        # Smoke test
        st_model.encode(["test"], show_progress_bar=False)
        logger.info("Loaded %s via manual Pooling construction", model_id)
        return st_model
    except Exception as e1:
        logger.warning("0xnbk manual Pooling load failed (%s); trying fallback model", e1)

    # ── Attempt 2: all-MiniLM-L6-v2 stable fallback ──────────────────────────
    fallback = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        logger.info("Loading fallback embedding model: %s", fallback)
        try:
            st_model = SentenceTransformer(fallback, local_files_only=True)
        except Exception:
            st_model = SentenceTransformer(fallback)
        logger.info("Fallback model loaded: %s", fallback)
        return st_model
    except Exception as e2:
        logger.warning("All embedding models failed (%s); semantic score uses TF-IDF", e2)
        return None


def _encode_texts(texts: list[str]) -> np.ndarray:
    """Encode texts into normalized numpy embeddings."""
    model = _load_embedding_model()
    if model is None:
        return np.array([])
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings


@lru_cache(maxsize=1)
def get_embedding_model():
    """Compatibility shim — returns the loaded model or None."""
    return _load_embedding_model()


@lru_cache(maxsize=1)
def get_ensemble_mappers():
    """Download and reconstruct the Ridge, MLPRegressor, and PolynomialFeatures models from HuggingFace JSON."""
    try:
        from huggingface_hub import hf_hub_download
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import PolynomialFeatures

        logger.info("Downloading/loading ensemble weights from HuggingFace repo: %s", settings.sbert_model)

        try:
            ridge_path = hf_hub_download(repo_id=settings.sbert_model, filename="ridge_weights.json", local_files_only=True)
            neural_path = hf_hub_download(repo_id=settings.sbert_model, filename="neural_weights.json", local_files_only=True)
            poly_path = hf_hub_download(repo_id=settings.sbert_model, filename="poly_features.json", local_files_only=True)
        except Exception:
            ridge_path = hf_hub_download(repo_id=settings.sbert_model, filename="ridge_weights.json")
            neural_path = hf_hub_download(repo_id=settings.sbert_model, filename="neural_weights.json")
            poly_path = hf_hub_download(repo_id=settings.sbert_model, filename="poly_features.json")

        with open(ridge_path, "r", encoding="utf-8") as f:
            ridge_data = json.load(f)
        with open(neural_path, "r", encoding="utf-8") as f:
            neural_data = json.load(f)
        with open(poly_path, "r", encoding="utf-8") as f:
            poly_data = json.load(f)

        score_mapper = Ridge(alpha=ridge_data["alpha"])
        score_mapper.coef_ = np.array(ridge_data["coefficients"])
        score_mapper.intercept_ = ridge_data["intercept"]
        score_mapper.n_features_in_ = ridge_data["n_features_in"]

        neural_mapper = MLPRegressor(
            hidden_layer_sizes=tuple(neural_data["hidden_layer_sizes"]),
            activation=neural_data["activation"]
        )
        neural_mapper.coefs_ = [np.array(c) for c in neural_data["coefs"]]
        neural_mapper.intercepts_ = [np.array(i) for i in neural_data["intercepts"]]
        neural_mapper.n_features_in_ = neural_data["n_features_in"]

        poly_features = PolynomialFeatures(
            degree=poly_data["degree"],
            include_bias=poly_data["include_bias"]
        )
        poly_features.n_features_in_ = poly_data["n_features_in"]
        poly_features.n_output_features_ = poly_data["n_output_features"]

        return score_mapper, neural_mapper, poly_features
    except Exception as exc:
        logger.warning("Failed to load ensemble mappers (%s); using linear fallback scaling", exc)
        return None


def map_similarity_to_score(similarity: float, mappers) -> float:
    """Map raw cosine similarity to calibrated 0–100 score using the ensemble regressor."""
    if mappers is None:
        return max(0.0, min(100.0, similarity * 100.0))

    score_mapper, neural_mapper, poly_features = mappers
    try:
        features = poly_features.transform([[similarity]])
        ridge_pred = score_mapper.predict(features)[0]
        neural_pred = neural_mapper.predict(features)[0]
        final_score = (ridge_pred * 0.5 + neural_pred * 0.5)
        return float(np.clip(final_score, 0.0, 100.0))
    except Exception as exc:
        logger.warning("Ensemble prediction failed (%s); falling back to linear scaling", exc)
        return max(0.0, min(100.0, similarity * 100.0))


class SemanticScorer:
    """Full-document semantic similarity using direct AutoModel loading.

    Encodes the full resume text and full JD text into embeddings via
    mean-pooled transformers output, then computes cosine similarity
    and maps through the nbk ensemble regressor to a calibrated 0–100 score.
    """

    def score(self, resume_text: str, jd: RoleJD, resume: StructuredResume | None = None) -> int:
        jd_text = jd_to_text(jd)
        if not resume_text.strip() or not jd_text.strip():
            return 0

        embeddings = _encode_texts([resume_text, jd_text])
        if embeddings is None or len(embeddings) == 0:
            logger.warning("Encoding failed; falling back to TF-IDF for semantic score")
            return TfidfScorer().score(resume_text, jd_text)

        sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        mappers = get_ensemble_mappers()
        return clamp_score(map_similarity_to_score(sim, mappers))
