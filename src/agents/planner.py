from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.llm import LLMClient


class PlannerAgent:
    def __init__(self, config: dict[str, Any], prompt_path: str = "prompts/planner_prompt.txt") -> None:
        self.config = config
        self.task_type = str(config.get("task_type", config.get("data", {}).get("task_type", "sentiment"))).lower()
        self.prompt_template = Path(self._resolve_prompt_path(prompt_path)).read_text(encoding="utf-8")
        self.llm = LLMClient(config)

    def _resolve_prompt_path(self, default_path: str) -> str:
        if self.task_type == "nli":
            candidate = Path(default_path).with_name("planner_nli_prompt.txt")
            if candidate.exists():
                return str(candidate)
        return default_path

    async def plan(self, sample: dict[str, Any], target_label: int) -> dict[str, Any]:
        if not self.llm.enabled:
            return self._rule_plan(sample, target_label)
        prompt = self.prompt_template.format(
            id=sample["id"],
            text=sample["text"],
            label=sample["label"],
            target_label=target_label,
        )
        try:
            return await self.llm.json_completion(stage="planner", prompt=prompt, max_retries=3)
        except Exception as exc:
            return self._rule_plan(sample, target_label)

    def _rule_plan(self, sample: dict[str, Any], target_label: int) -> dict[str, Any]:
        words = sample["text"].split()
        if self.task_type == "nli":
            return {
                "id": sample["id"],
                "target_label": target_label,
                "causal_features": ["premise-hypothesis entailment relation", "shared entities", "logical implication"],
                "spurious_features": ["surface overlap", "lexical shortcuts", "artifact words"],
                "elements_to_change": ["hypothesis wording", "entailment cues"],
                "elements_to_preserve": ["premise meaning", "named entities", "topic"],
                "editing_constraints": [
                    "keep premise intact",
                    "rewrite hypothesis to flip entailment relation",
                    "keep output fluent and pair-formatted",
                ],
            }
        candidates = [w.strip(".,!?;:") for w in words if len(w) > 5][:4]
        return {
            "id": sample["id"],
            "target_label": target_label,
            "causal_features": ["movie/topic entity", "main event"],
            "spurious_features": ["sentiment adjectives", "emotional tone"],
            "elements_to_change": candidates,
            "elements_to_preserve": ["named entities", "main event structure"],
            "editing_constraints": [
                "minimal edits",
                "do not change named entities unless necessary",
                "keep fluent and natural",
            ],
        }
