from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from src.utils.llm import LLMClient

POS_TO_NEG = {
    "good": "bad",
    "great": "terrible",
    "excellent": "awful",
    "wonderful": "dull",
    "touching": "flat",
    "fun": "boring",
    "amazing": "disappointing",
    "love": "hate",
}
NEG_TO_POS = {v: k for k, v in POS_TO_NEG.items()}
NEG_TO_POS.update(
    {
        "bad": "good",
        "terrible": "great",
        "awful": "excellent",
        "boring": "fun",
        "disappointing": "amazing",
        "hate": "love",
    }
)


class GeneratorAgent:
    def __init__(self, config: dict[str, Any], prompt_path: str = "prompts/generator_prompt.txt") -> None:
        self.config = config
        self.task_type = str(config.get("task_type", config.get("data", {}).get("task_type", "sentiment"))).lower()
        self.prompt_template = Path(self._resolve_prompt_path(prompt_path)).read_text(encoding="utf-8")
        self.llm = LLMClient(config)

    def _resolve_prompt_path(self, default_path: str) -> str:
        if self.task_type == "nli":
            candidate = Path(default_path).with_name("generator_nli_prompt.txt")
            if candidate.exists():
                return str(candidate)
        return default_path

    async def generate(
        self,
        sample: dict[str, Any],
        plan: dict[str, Any],
        target_label: int,
        round_idx: int = 1,
        critique: str = "",
    ) -> list[dict[str, str]]:
        num_candidates = int(self.config["augmentation"]["num_candidates"])
        if not self.llm.enabled:
            return self._rule_candidates(sample["text"], target_label, num_candidates)

        prompt = self.prompt_template.format(
            id=sample["id"],
            text=sample["text"],
            plan_json=plan,
            critique_or_empty=critique or "",
            num_candidates=num_candidates,
            target_label=target_label,
        )
        try:
            payload = await self.llm.json_completion(stage="generator", prompt=prompt, max_retries=3)
            return self._normalize_candidates(payload)
        except Exception as exc:
            raise RuntimeError(f"Generator LLM call failed; rule fallback is disabled: {exc}") from exc

    def _rule_candidates(self, text: str, target_label: int, n: int) -> list[dict[str, str]]:
        if self.task_type == "nli":
            return self._nli_rule_candidates(text, target_label, n)
        mapping = POS_TO_NEG if target_label == 0 else NEG_TO_POS
        outputs = []
        for i in range(n):
            changed = text
            for src, dst in random.sample(list(mapping.items()), k=min(4, len(mapping))):
                changed = changed.replace(src, dst).replace(src.capitalize(), dst.capitalize())
            if changed == text:
                changed = f"{text} but it feels {'worse' if target_label == 0 else 'better'} overall."
            outputs.append({"candidate_id": f"c{i+1}", "text": changed})
        return outputs

    def _nli_rule_candidates(self, text: str, target_label: int, n: int) -> list[dict[str, str]]:
        premise, hypothesis = self._split_nli_pair(text)
        outputs: list[dict[str, str]] = []
        for i in range(n):
            if target_label == 1:
                candidate_hypothesis = self._make_entailment_hypothesis(premise, hypothesis, i)
            else:
                candidate_hypothesis = self._make_non_entailment_hypothesis(premise, hypothesis, i)
            outputs.append(
                {
                    "candidate_id": f"c{i+1}",
                    "text": f"Premise: {premise}\nHypothesis: {candidate_hypothesis}",
                }
            )
        return outputs

    @staticmethod
    def _split_nli_pair(text: str) -> tuple[str, str]:
        premise = text
        hypothesis = text
        if "Premise:" in text and "Hypothesis:" in text:
            premise_part, hypothesis_part = text.split("Hypothesis:", 1)
            premise = premise_part.split("Premise:", 1)[-1].strip()
            hypothesis = hypothesis_part.strip()
        return premise.strip(), hypothesis.strip()

    @staticmethod
    def _make_entailment_hypothesis(premise: str, hypothesis: str, idx: int) -> str:
        premise_sentences = [part.strip() for part in premise.split(".") if part.strip()]
        if premise_sentences:
            base = premise_sentences[min(idx, len(premise_sentences) - 1)]
            return base if base.endswith(".") else f"{base}."
        if hypothesis.strip():
            return hypothesis.strip() if hypothesis.strip().endswith(".") else f"{hypothesis.strip()}."
        return "This hypothesis is supported by the premise."

    @staticmethod
    def _make_non_entailment_hypothesis(premise: str, hypothesis: str, idx: int) -> str:
        templates = [
            "The premise does not guarantee this statement.",
            "This claim introduces information not stated in the premise.",
            "The premise fails to support this hypothesis.",
        ]
        base = templates[idx % len(templates)]
        if hypothesis.strip():
            return f"{base} {hypothesis.strip()}"
        return base

    @staticmethod
    def _normalize_candidates(payload: dict[str, Any]) -> list[dict[str, str]]:
        raw = payload.get("candidates", [])
        out: list[dict[str, str]] = []
        if isinstance(raw, list):
            for i, item in enumerate(raw, start=1):
                if isinstance(item, dict):
                    txt = item.get("text") or item.get("candidate") or item.get("output")
                    cid = item.get("candidate_id") or f"c{i}"
                    if isinstance(txt, str) and txt.strip():
                        out.append({"candidate_id": str(cid), "text": txt.strip()})
                elif isinstance(item, str) and item.strip():
                    out.append({"candidate_id": f"c{i}", "text": item.strip()})
        if not out:
            pred = payload.get("prediction")
            if isinstance(pred, str) and pred.strip():
                out.append({"candidate_id": "c1", "text": pred.strip()})
        return out
