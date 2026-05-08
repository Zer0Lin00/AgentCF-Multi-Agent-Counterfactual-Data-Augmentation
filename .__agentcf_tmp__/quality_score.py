from __future__ import annotations


def final_quality_score(components: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(float(w) for w in weights.values())
    if total_weight <= 0:
        return 0.0
    return sum(float(weights.get(name, 0.0)) * float(components.get(name, 0.0)) for name in weights) / total_weight
