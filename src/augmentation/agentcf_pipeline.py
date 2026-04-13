from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.generator import GeneratorAgent
from src.agents.planner import PlannerAgent
from src.agents.selector import SelectorAgent
from src.agents.verifier import VerifierAgent
from src.utils.io import append_jsonl


async def _process_one(
    sample: dict[str, Any],
    planner: PlannerAgent,
    generator: GeneratorAgent,
    verifier: VerifierAgent,
    selector: SelectorAgent,
    max_retry_rounds: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    target_label = 1 - int(sample["label"])
    ablation = planner.config.get("ablation", {})
    if ablation.get("w_o_planner", False):
        plan = {
            "id": sample["id"],
            "target_label": target_label,
            "elements_to_preserve": ["movie", "event"],
            "elements_to_change": [],
        }
    else:
        plan = await planner.plan(sample, target_label)
    all_candidates: list[dict] = []
    all_verifications: list[dict] = []
    critique = ""
    chosen: list[dict] = []
    retry_budget = 0 if ablation.get("w_o_verifier_feedback", False) else max_retry_rounds
    for round_idx in range(1, retry_budget + 2):
        cands = await generator.generate(
            sample=sample,
            plan=plan,
            target_label=target_label,
            round_idx=round_idx,
            critique=critique,
        )
        for c in cands:
            rec = {"id": sample["id"], "round": round_idx, "target_label": target_label, **c}
            all_candidates.append(rec)
            ver = verifier.verify(sample, target_label, plan, c)
            ver["round"] = round_idx
            all_verifications.append(ver)
        current_round = [v for v in all_verifications if v["round"] == round_idx]
        if ablation.get("w_o_selector", False):
            chosen = [
                {
                    "id": sample["id"],
                    "text": v["candidate_text"],
                    "label": target_label,
                    "source": "agentcf",
                    "candidate_id": v["candidate_id"],
                    "final_score": float(v["final_score"]),
                }
                for v in current_round
                if v["status"] == "pass"
            ]
        else:
            chosen = selector.select(sample, current_round, target_label)
        if chosen:
            break
        rejected = [v for v in all_verifications if v["round"] == round_idx and v["status"] == "reject"]
        critique = rejected[0]["critique"] if rejected else "improve label flip and minimal edits"
    return chosen, [plan], all_candidates, all_verifications


async def build_agentcf_aug(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    planner = PlannerAgent(config)
    generator = GeneratorAgent(config)
    verifier = VerifierAgent(config)
    selector = SelectorAgent(config)
    max_retry_rounds = int(config["augmentation"]["max_retry_rounds"])
    checkpoint_every = int(config["runtime"]["checkpoint_every_n_samples"])
    output_root = Path(config.get("output_root", "outputs"))
    checkpoints_dir = output_root / "checkpoints"
    candidates_dir = output_root / "generated_candidates"
    selected_dir = output_root / "selected_counterfactuals"
    selected_rows: list[dict] = []
    selected_buffer: list[dict] = []
    plans_buffer: list[dict] = []
    candidates_buffer: list[dict] = []
    verifications_buffer: list[dict] = []
    total_candidates = 0
    total_verifications = 0

    sample_records = df.to_dict(orient="records")
    for idx, sample in enumerate(sample_records, start=1):
        selected, plans, cands, vers = await _process_one(
            sample, planner, generator, verifier, selector, max_retry_rounds
        )
        selected_rows.extend(selected)
        selected_buffer.extend(selected)
        plans_buffer.extend(plans)
        candidates_buffer.extend(cands)
        verifications_buffer.extend(vers)
        total_candidates += len(cands)
        total_verifications += len(vers)

        if idx % checkpoint_every == 0 or idx == len(sample_records):
            append_jsonl(checkpoints_dir / "plans.jsonl", plans_buffer)
            append_jsonl(candidates_dir / "candidates.jsonl", candidates_buffer)
            append_jsonl(checkpoints_dir / "verifications.jsonl", verifications_buffer)
            append_jsonl(selected_dir / "selected.jsonl", selected_buffer)
            plans_buffer.clear()
            candidates_buffer.clear()
            verifications_buffer.clear()
            selected_buffer.clear()
            print(f"[AgentCF] checkpoint: {idx}/{len(sample_records)} samples processed", flush=True)

    aug_df = pd.DataFrame(selected_rows, columns=["id", "text", "label", "source", "candidate_id", "final_score"])
    stats = {
        "input_samples": len(df),
        "generated_candidates": total_candidates,
        "verified_candidates": total_verifications,
        "selected_samples": len(aug_df),
        "dropped_samples": max(0, len(df) - len(aug_df)),
    }
    return aug_df, stats
