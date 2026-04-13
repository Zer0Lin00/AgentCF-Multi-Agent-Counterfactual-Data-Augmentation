from __future__ import annotations

import re
from typing import Any

from src.metrics.minimality import edit_similarity
from src.metrics.quality_score import final_quality_score
from src.metrics.similarity import semantic_similarity

POS_WORDS = {
    "good",
    "great",
    "excellent",
    "wonderful",
    "touching",
    "amazing",
    "fun",
    "love",
    "moving",
}
NEG_WORDS = {
    "bad",
    "terrible",
    "awful",
    "dull",
    "flat",
    "boring",
    "disappointing",
    "hate",
    "lifeless",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


class VerifierAgent:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.thresholds = config["thresholds"]
        self.weights = config["weights"]

    def verify(
        self,
        sample: dict[str, Any],
        target_label: int,
        plan: dict[str, Any],
        candidate: dict[str, str],
    ) -> dict[str, Any]:
        original = sample["text"]
        cand_text = candidate["text"]

        label_score = self._label_score(cand_text, target_label)
        sem_score = semantic_similarity(original, cand_text)
        min_score = edit_similarity(original, cand_text)
        consistency = self._consistency_score(original, cand_text, plan)
        score = final_quality_score(label_score, sem_score, min_score, consistency, self.weights)

        hard_ok = (
            label_score >= self.thresholds["label_score"]
            and sem_score >= self.thresholds["semantic_score"]
            and min_score >= self.thresholds["minimality_score"]
            and score >= self.thresholds["final_score"]
        )
        return {
            "id": sample["id"],
            "candidate_id": candidate["candidate_id"],
            "candidate_text": cand_text,
            "label_score": round(label_score, 4),
            "semantic_score": round(sem_score, 4),
            "minimality_score": round(min_score, 4),
            "consistency_score": round(consistency, 4),
            "final_score": round(score, 4),
            "status": "pass" if hard_ok else "reject",
            "critique": "" if hard_ok else self._critique(label_score, sem_score, min_score),
        }

    def _label_score(self, text: str, target_label: int) -> float:
        toks = _tokenize(text)
        pos = sum(t in POS_WORDS for t in toks)
        neg = sum(t in NEG_WORDS for t in toks)
        if target_label == 1:
            raw = (pos + 1) / (pos + neg + 2)
        else:
            raw = (neg + 1) / (pos + neg + 2)
        return float(min(max(raw, 0.0), 1.0))

    def _consistency_score(self, original: str, candidate: str, plan: dict[str, Any]) -> float:
        preserve = [str(w).lower() for w in plan.get("elements_to_preserve", []) if w]
        if not preserve:
            return 0.8
        o = original.lower()
        c = candidate.lower()
        preserved = 0
        for token in preserve:
            token = token.split()[0]
            if token in o and token in c:
                preserved += 1
        return max(0.0, min(1.0, preserved / max(len(preserve), 1)))

    @staticmethod
    def _critique(label_score: float, semantic_score: float, min_score: float) -> str:
        issues = []
        if label_score < 0.8:
            issues.append("sentiment flip is insufficient")
        if semantic_score < 0.8:
            issues.append("semantic drift is too large")
        if min_score < 0.7:
            issues.append("edits are too large")
        return "; ".join(issues) if issues else "improve fluency and constraints"

