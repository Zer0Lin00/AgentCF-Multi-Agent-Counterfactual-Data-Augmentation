from __future__ import annotations

import argparse
import asyncio
from math import floor
from copy import deepcopy

from src.models.train import run_experiment
from src.utils.config import load_config


async def run_low_resource(config_path: str) -> None:
    base = load_config(config_path)
    for ratio, r in [("10%", 0.1), ("30%", 0.3), ("50%", 0.5), ("100%", 1.0)]:
        cfg = deepcopy(base)
        cfg["low_resource_ratio"] = r
        cfg["train_samples"] = 0
        cfg["output_root"] = f"outputs/low_resource/{ratio}"
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
