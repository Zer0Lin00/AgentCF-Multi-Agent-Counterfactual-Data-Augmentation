from __future__ import annotations


def final_quality_score(
    label_score: float,
    semantic_score: float,
    minimality_score: float,
    consistency_score: float,
    weights: dict[str, float],
) -> float:
    return (
        weights["label_score"] * label_score
        + weights["semantic_score"] * semantic_score
        + weights["minimality_score"] * minimality_score
        + weights["consistency_score"] * consistency_score
    )

