from __future__ import annotations

import os
from functools import lru_cache

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@lru_cache(maxsize=1)
def _get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def semantic_similarity(text_a: str, text_b: str) -> float:
    model = _get_embedder()
    emb = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

