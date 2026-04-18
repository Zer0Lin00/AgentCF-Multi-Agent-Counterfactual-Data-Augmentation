from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.generator import GeneratorAgent
from src.agents.verifier import VerifierAgent
from src.utils.llm import LLMClient


def _extract_candidate_text(payload: dict[str, Any]) -> str:
    for key in ["text", "counterfactual", "candidate", "prediction", "output"]:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    candidates = payload.get("candidates", [])
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            txt = first.get("text") or first.get("candidate") or first.get("output")
            if isinstance(txt, str):
                return txt.strip()
    return ""


async def build_single_agent_cf(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt_template = Path("prompts/single_agent_prompt.txt").read_text(encoding="utf-8")
    llm = LLMClient(config)
    generator_fallback = GeneratorAgent(config)
    verifier = VerifierAgent(config)

    rows: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []

    for sample in df.to_dict(orient="records"):
        target_label = 1 - int(sample["label"])
        cand_text = ""
        if llm.enabled:
            prompt = prompt_template.format(
                text=sample["text"],
                label=sample["label"],
                target_label=target_label,
            )
            try:
                payload = await llm.json_completion(stage="generator", prompt=prompt, max_retries=3)
                cand_text = _extract_candidate_text(payload)
            except Exception:
                cand_text = ""
        if not cand_text:
            fallback = await generator_fallback.generate(
                sample=sample,
                plan={"elements_to_preserve": ["movie", "event"]},
                target_label=target_label,
                round_idx=1,
                critique="",
            )
            if fallback:
                cand_text = fallback[0]["text"]
        if not cand_text:
            continue
        cand = {"candidate_id": "c1", "text": cand_text}
        result = verifier.verify(sample, target_label, {"elements_to_preserve": ["movie"]}, cand)
        verifications.append(result)
        if result["status"] == "pass":
            rows.append(
                {
                    "id": sample["id"],
                    "text": cand_text,
                    "label": target_label,
                    "source": "single_agent",
                }
            )

    aug_df = pd.DataFrame(rows, columns=["id", "text", "label", "source"])
    ver_df = pd.DataFrame(verifications)
    return aug_df, ver_df

