from __future__ import annotations

from typing import Any

import pandas as pd


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

        if self.thresholds.get("filtering_mode") == "dynamic_percentile" and len(df) > self.top_k:
            keep = float(self.thresholds.get("percentile_keep", 0.3))
            quantile = max(0.0, min(1.0, 1.0 - keep))
            cut = float(df["final_score"].quantile(quantile))
            df = df[df["final_score"] >= cut].copy()
            if df.empty:
                return []

        picked = df.sort_values("final_score", ascending=False).head(self.top_k)
        return [
            {
                "id": sample["id"],
                "text": row["candidate_text"],
                "label": target_label,
                "source": "agentcf",
                "candidate_id": row["candidate_id"],
                "final_score": float(row["final_score"]),
            }
            for _, row in picked.iterrows()
        ]
