from __future__ import annotations

from itertools import combinations

from .minimality import edit_similarity


def diversity_score(candidates: list[str]) -> float:
    if len(candidates) <= 1:
        return 0.0
    scores = []
    for a, b in combinations(candidates, 2):
        scores.append(1.0 - edit_similarity(a, b))
    return float(sum(scores) / len(scores))

