from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer
    cache = "/root/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    return SentenceTransformer(cache)


def _word_embeddings(text: str) -> np.ndarray:
    words = text.lower().split()
    if not words:
        return np.zeros((1, 384))
    embedder = _get_embedder()
    return embedder.encode(words, normalize_embeddings=False)


def wasserstein_minimality(original: str, changed: str) -> float:
    """Wasserstein-1 distance between word embedding distributions, mapped to [0,1] (higher = more minimal)."""
    from scipy.stats import wasserstein_distance
    emb_o = _word_embeddings(original)
    emb_c = _word_embeddings(changed)
    # project to 1D via PCA-like mean direction for efficiency
    all_emb = np.vstack([emb_o, emb_c])
    direction = np.linalg.svd(all_emb - all_emb.mean(0), full_matrices=False)[2][0]
    proj_o = emb_o @ direction
    proj_c = emb_c @ direction
    dist = wasserstein_distance(proj_o, proj_c)
    # normalise: dist=0 → score=1, dist≥2 → score=0
    return float(max(0.0, 1.0 - dist / 2.0))


def edit_similarity(original: str, changed: str) -> float:
    """Higher is better; 1 means almost unchanged."""
    return float(SequenceMatcher(a=original, b=changed).ratio())


def minimality_score(original: str, changed: str, use_wasserstein: bool = False) -> float:
    if use_wasserstein:
        return wasserstein_minimality(original, changed)
    return edit_similarity(original, changed)
