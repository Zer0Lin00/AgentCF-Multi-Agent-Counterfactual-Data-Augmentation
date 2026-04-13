from __future__ import annotations

import pandas as pd

from src.agents.generator import GeneratorAgent
from src.agents.verifier import VerifierAgent

async def build_single_cf(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator = GeneratorAgent(config)
    verifier = VerifierAgent(config)
    rows: list[dict] = []
    verifications: list[dict] = []
    for sample in df.to_dict(orient="records"):
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
            verifications.append(result)
            if result["status"] == "pass":
                rows.append(
                    {
                        "id": sample["id"],
                        "text": cand["text"],
                        "label": target_label,
                        "source": "single_cf",
                    }
                )
                break
    aug = pd.DataFrame(rows, columns=["id", "text", "label", "source"])
    ver = pd.DataFrame(verifications)
    return aug, ver
