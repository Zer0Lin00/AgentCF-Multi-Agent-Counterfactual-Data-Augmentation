from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


@lru_cache(maxsize=2)
def _load_label_model(model_name: str) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to("cpu")
    return tokenizer, model


class VerifierAgent:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.thresholds = config["thresholds"]
        self.weights = config["weights"]
        self.label_model_name = str(
            config.get("verification", {}).get("label_model_name", "distilbert-base-uncased-finetuned-sst-2-english")
        )

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
        tokenizer, model = _load_label_model(self.label_model_name)
        with torch.no_grad():
            batch = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()
        if len(probs) < 2:
            return 0.0
        return float(probs[int(target_label)])

    def _consistency_score(self, original: str, candidate: str, plan: dict[str, Any]) -> float:
        raw_preserve = plan.get("elements_to_preserve", [])
        preserve: list[str] = []
        if isinstance(raw_preserve, list):
            for item in raw_preserve:
                if isinstance(item, str):
                    preserve.append(item.lower())
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("token") or item.get("name")
                    if isinstance(text, str):
                        preserve.append(text.lower())
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
        if label_score < 0.75:
            issues.append("sentiment flip is insufficient")
        if semantic_score < 0.8:
            issues.append("semantic drift is too large")
        if min_score < 0.7:
            issues.append("edits are too large")
        return "; ".join(issues) if issues else "improve fluency and constraints"
