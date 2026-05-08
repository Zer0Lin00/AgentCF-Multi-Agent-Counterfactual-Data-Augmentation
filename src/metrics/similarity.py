from __future__ import annotations

import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_LOCAL_MINILM = "/root/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


@lru_cache(maxsize=1)
def _get_embedder(model_name: str = _LOCAL_MINILM) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")


def semantic_similarity(text_a: str, text_b: str) -> float:
    model = _get_embedder()
    emb = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])
