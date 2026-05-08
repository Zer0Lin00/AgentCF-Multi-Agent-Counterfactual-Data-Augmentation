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


@lru_cache(maxsize=4)
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
            config.get("verification", {}).get(
                "sst2_label_model_name",
                config.get("verification", {}).get("label_model_name", "distilbert-base-uncased-finetuned-sst-2-english"),
            )
        )
        self.general_model_name = str(
            config.get("verification", {}).get("general_sentiment_model_name", self.label_model_name)
        )
        self.ood_mode = bool(config.get("ood_mode", False) or config.get("verification", {}).get("ood_mode", False))

    def verify(
        self,
        sample: dict[str, Any],
        target_label: int,
        plan: dict[str, Any],
        candidate: dict[str, str],
    ) -> dict[str, Any]:
        original = sample["text"]
        cand_text = candidate["text"]

        label_score = self._label_score(cand_text, target_label, self.label_model_name)
        general_score = (
            self._label_score(cand_text, target_label, self.general_model_name) if self.ood_mode else label_score
        )
        domain_invariant = min(label_score, general_score)
        sem_score = semantic_similarity(original, cand_text)
        min_score = edit_similarity(original, cand_text)
        consistency = self._consistency_score(original, cand_text, plan)
        style_robustness = self._style_robustness_score(original, cand_text)
        score = final_quality_score(
            {
                "label_score": label_score,
                "general_sentiment_score": general_score,
                "domain_invariant_score": domain_invariant,
                "semantic_score": sem_score,
                "minimality_score": min_score,
                "consistency_score": consistency,
                "style_robustness_score": style_robustness,
            },
            self.weights,
        )

        hard_ok = (
            label_score >= self.thresholds["label_score"]
            and general_score >= self.thresholds.get("general_sentiment_score", self.thresholds["label_score"])
            and domain_invariant >= self.thresholds.get("domain_invariant_score", 0.0)
            and sem_score >= self.thresholds["semantic_score"]
            and min_score >= self.thresholds["minimality_score"]
            and score >= self.thresholds["final_score"]
        )
        return {
            "id": sample["id"],
            "candidate_id": candidate["candidate_id"],
            "candidate_text": cand_text,
            "label_score": round(label_score, 4),
            "general_sentiment_score": round(general_score, 4),
            "domain_invariant_score": round(domain_invariant, 4),
            "semantic_score": round(sem_score, 4),
            "minimality_score": round(min_score, 4),
            "consistency_score": round(consistency, 4),
            "style_robustness_score": round(style_robustness, 4),
            "final_score": round(score, 4),
            "status": "pass" if hard_ok else "reject",
            "critique": "" if hard_ok else self._critique(label_score, general_score, sem_score, min_score),
        }

    def _label_score(self, text: str, target_label: int, model_name: str) -> float:
        tokenizer, model = _load_label_model(model_name)
        with torch.no_grad():
            batch = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()
        if len(probs) == 3:
            return float(probs[2] if int(target_label) == 1 else probs[0])
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
            parts = token.split()
            if not parts:
                continue
            token = parts[0]
            if token in o and token in c:
                preserved += 1
        return max(0.0, min(1.0, preserved / max(len(preserve), 1)))

    @staticmethod
    def _style_robustness_score(original: str, candidate: str) -> float:
        o_tokens = set(_tokenize(original))
        c_tokens = set(_tokenize(candidate))
        changed = o_tokens.symmetric_difference(c_tokens)
        sentiment_changed = len(changed & (POS_WORDS | NEG_WORDS))
        lexical_shortcut_penalty = 0.30 if sentiment_changed <= 2 and len(changed) <= 4 else 0.0
        short_penalty = 0.20 if len(candidate.split()) < 8 else 0.0
        template_penalty = 0.15 if len(changed) == 1 and sentiment_changed == 1 else 0.0
        length_ratio = min(len(candidate.split()), len(original.split())) / max(len(candidate.split()), len(original.split()), 1)
        review_markers = {"because", "although", "while", "but", "however", "overall", "yet", "still", "despite", "though"}
        context_bonus = 0.14 if c_tokens & review_markers else 0.0
        score = 0.70 * length_ratio + context_bonus - lexical_shortcut_penalty - short_penalty - template_penalty
        return max(0.0, min(1.0, score))

    @staticmethod
    def _critique(label_score: float, general_score: float, semantic_score: float, min_score: float) -> str:
        issues = []
        if label_score < 0.75:
            issues.append("SST-2 sentiment flip is insufficient")
        if general_score < 0.70:
            issues.append("domain-general sentiment is insufficient")
        if semantic_score < 0.8:
            issues.append("semantic drift is too large")
        if min_score < 0.7:
            issues.append("edits are too large")
        return "; ".join(issues) if issues else "improve fluency and constraints"
