from __future__ import annotations

import random

import pandas as pd

REPLACE_MAP = {
    "movie": "film",
    "film": "movie",
    "good": "nice",
    "bad": "poor",
    "great": "excellent",
    "terrible": "awful",
    "story": "plot",
}


def synonym_replacement(text: str) -> str:
    out = text
    keys = list(REPLACE_MAP.keys())
    random.shuffle(keys)
    for k in keys[:3]:
        if k in out.lower():
            out = out.replace(k, REPLACE_MAP[k]).replace(k.capitalize(), REPLACE_MAP[k].capitalize())
    return out


def build_standard_aug(df: pd.DataFrame, ratio: float = 1.0) -> pd.DataFrame:
    n = int(len(df) * ratio)
    sampled = df.sample(n=min(n, len(df)), random_state=42).copy()
    sampled["text"] = sampled["text"].map(synonym_replacement)
    sampled["source"] = "standard_aug"
    return sampled[["id", "text", "label", "source"]]

