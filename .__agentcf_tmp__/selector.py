from __future__ import annotations

import re
from typing import Any

import pandas as pd

STRONG_SENTIMENT = {
    "good",
    "great",
    "excellent",
    "wonderful",
    "amazing",
    "love",
    "bad",
    "terrible",
    "awful",
    "boring",
    "disappointing",
    "hate",
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z']+", text.lower()))


class SelectorAgent:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.thresholds = config["thresholds"]
        self.top_k = int(config["augmentation"]["keep_top_k"])

    def select(self, sample: dict[str, Any], verified: list[dict[str, Any]], target_label: int) -> list[dict[str, Any]]:
        if not verified:
            return []
        df = pd.DataFrame(verified)
        df = df[
            (df["label_score"] >= self.thresholds["label_score"])
            & (df["semantic_score"] >= self.thresholds["semantic_score"])
            & (df["final_score"] >= self.thresholds["final_score"])
        ].copy()
        if df.empty:
            return []

        if self.config.get("ood_mode", False):
            original_tokens = _tokens(str(sample["text"]))
            df["shortcut_penalty"] = [
                self._shortcut_penalty(original_tokens, str(text)) for text in df["candidate_text"]
            ]
            df["selector_ood_score"] = (
                0.45 * df["final_score"]
                + 0.20 * df.get("domain_invariant_score", df["label_score"])
                + 0.20 * df.get("style_robustness_score", df["semantic_score"])
                + 0.15 * df["semantic_score"]
                - df["shortcut_penalty"]
            )
            df = df[df["selector_ood_score"] >= self.thresholds.get("selector_ood_score", 0.0)].copy()
            if df.empty:
                return []

        if self.thresholds.get("filtering_mode") == "dynamic_percentile" and len(df) > self.top_k:
            keep = float(self.thresholds.get("percentile_keep", 0.3))
            quantile = max(0.0, min(1.0, 1.0 - keep))
            score_col = "selector_ood_score" if "selector_ood_score" in df else "final_score"
            cut = float(df[score_col].quantile(quantile))
            df = df[df[score_col] >= cut].copy()
            if df.empty:
                return []

        score_col = "selector_ood_score" if "selector_ood_score" in df else "final_score"
        picked = df.sort_values(score_col, ascending=False).head(self.top_k)
        return [
            {
                "id": sample["id"],
                "text": row["candidate_text"],
                "label": target_label,
                "source": "agentcf",
                "candidate_id": row["candidate_id"],
                "final_score": float(row["final_score"]),
                "selector_ood_score": float(row.get("selector_ood_score", row["final_score"])),
            }
            for _, row in picked.iterrows()
        ]

    @staticmethod
    def _shortcut_penalty(original_tokens: set[str], candidate_text: str) -> float:
        candidate_tokens = _tokens(candidate_text)
        changed = original_tokens.symmetric_difference(candidate_tokens)
        if len(changed) <= 4 and changed & STRONG_SENTIMENT:
            return 0.18
        if len(candidate_tokens) < 6:
            return 0.12
        return 0.0
