from __future__ import annotations

import asyncio

import pandas as pd

from src.agents.generator import GeneratorAgent
from src.agents.verifier import VerifierAgent

CONCURRENCY = 20


async def _process_one(sample: dict, generator: GeneratorAgent, verifier: VerifierAgent) -> tuple[list[dict], list[dict]]:
    target_label = 1 - int(sample["label"])
    cands = await generator.generate(
        sample=sample,
        plan={"elements_to_preserve": ["movie", "event"]},
        target_label=target_label,
        round_idx=1,
        critique="",
    )
    rows, vers = [], []
    for cand in cands:
        result = verifier.verify(sample, target_label, {"elements_to_preserve": ["movie"]}, cand)
        vers.append(result)
        if result["status"] == "pass":
            rows.append({"id": sample["id"], "text": cand["text"], "label": target_label, "source": "single_cf"})
            break
    return rows, vers


async def build_single_cf(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator = GeneratorAgent(config)
    verifier = VerifierAgent(config)
    samples = df.to_dict(orient="records")
    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded(sample):
        async with sem:
            return await _process_one(sample, generator, verifier)

    results = await asyncio.gather(*[bounded(s) for s in samples])
    rows, verifications = [], []
    for r, v in results:
        rows.extend(r)
        verifications.extend(v)

    aug = pd.DataFrame(rows, columns=["id", "text", "label", "source"]) if rows else pd.DataFrame(columns=["id", "text", "label", "source"])
    ver = pd.DataFrame(verifications)
    return aug, ver
