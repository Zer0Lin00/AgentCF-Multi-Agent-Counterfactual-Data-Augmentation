from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.llm import LLMClient


class PlannerAgent:
    def __init__(self, config: dict[str, Any], prompt_path: str = "prompts/planner_prompt.txt") -> None:
        self.config = config
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")
        self.llm = LLMClient(config)

    async def plan(self, sample: dict[str, Any], target_label: int) -> dict[str, Any]:
        if not self.llm.enabled:
            raise RuntimeError("Planner LLM is disabled; rule fallback is not allowed for this run")
        prompt = self.prompt_template.format(
            id=sample["id"],
            text=sample["text"],
            label=sample["label"],
            target_label=target_label,
        )
        try:
            return await self.llm.json_completion(stage="planner", prompt=prompt, max_retries=3)
        except Exception as exc:
            raise RuntimeError(f"Planner LLM call failed; rule fallback is disabled: {exc}") from exc

    def _rule_plan(self, sample: dict[str, Any], target_label: int) -> dict[str, Any]:
        words = sample["text"].split()
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
