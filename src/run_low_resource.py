from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy

from src.models.train import run_experiment
from src.utils.config import load_config


async def run_low_resource(config_path: str) -> None:
    base = load_config(config_path)
    for ratio, sample_count in [("10%", 500), ("30%", 1500), ("50%", 3000), ("100%", base.get("train_samples", 0))]:
        cfg = deepcopy(base)
        cfg["train_samples"] = sample_count
        cfg["output_root"] = f"outputs/low_resource/{ratio}"
        # Save a temporary config-less run by monkey patching expected fields.
        # Reuse run_experiment by writing config to disk would be equivalent.
        await run_experiment_from_dict(cfg)


async def run_experiment_from_dict(cfg: dict) -> None:
    from pathlib import Path
    import yaml

    tmp = Path("outputs/logs/tmp_low_resource.yaml")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    await run_experiment(str(tmp))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    asyncio.run(run_low_resource(args.config))


if __name__ == "__main__":
    main()

