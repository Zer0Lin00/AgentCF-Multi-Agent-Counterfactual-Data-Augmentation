from __future__ import annotations

import asyncio
import pandas as pd

from src.agents.generator import GeneratorAgent
from src.agents.verifier import VerifierAgent


async def build_single_cf(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator = GeneratorAgent(config)
    verifier = VerifierAgent(config)
    rows: list[dict] = []
    verifications: list[dict] = []

    sample_records = df.to_dict(orient="records")
    concurrency = int(config.get("runtime", {}).get("concurrency", 10))
    sem = asyncio.Semaphore(concurrency)

    async def _process_one(sample: dict) -> tuple[list[dict], list[dict]]:
        local_rows: list[dict] = []
        local_vers: list[dict] = []
        target_label = 1 - int(sample["label"])
        cands = await generator.generate(
            sample=sample,
            plan={"elements_to_preserve": ["movie", "event"]},
            target_label=target_label,
            round_idx=1,
            critique="",
        )
        for cand in cands:
            result = verifier.verify(sample, target_label, {"elements_to_preserve": ["movie"]}, cand)
            local_vers.append(result)
            if result["status"] == "pass":
                local_rows.append(
                    {
                        "id": sample["id"],
                        "text": cand["text"],
                        "label": target_label,
                        "source": "single_cf",
                    }
                )
                break
        return local_rows, local_vers

    async def _process_with_sem(sample: dict):
        async with sem:
            return await _process_one(sample)

    tasks = [_process_with_sem(s) for s in sample_records]
    for coro in asyncio.as_completed(tasks):
        local_rows, local_vers = await coro
        rows.extend(local_rows)
        verifications.extend(local_vers)

    aug = pd.DataFrame(rows, columns=["id", "text", "label", "source"]) if rows else pd.DataFrame(columns=["id", "text", "label", "source"])
    ver = pd.DataFrame(verifications) if verifications else pd.DataFrame()
    return aug, ver
