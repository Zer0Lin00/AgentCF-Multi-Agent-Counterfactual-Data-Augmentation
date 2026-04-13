from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from pathlib import Path

import yaml

from src.models.train import run_experiment
from src.run_ablation import run_ablation
from src.run_low_resource_matrix import run_matrix
from src.run_ood import run_ood
from src.utils.config import load_config

DEFAULT_SEEDS = [42, 43, 44]


async def _run_with_config(cfg: dict, tmp_name: str, runner) -> None:
    tmp = Path("outputs/logs/repeats") / tmp_name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    await runner(str(tmp))


async def _run_main(configs: list[str], seeds: list[int]) -> None:
    for seed in seeds:
        for config_path in configs:
            base = load_config(config_path)
            cfg = deepcopy(base)
            cfg["seed"] = seed
            cfg["output_root"] = f"outputs/repeats/main/seed_{seed}/{Path(config_path).stem}"
            await _run_with_config(cfg, f"main_seed_{seed}_{Path(config_path).stem}.yaml", run_experiment)


async def _run_low_resource(configs: list[str], seeds: list[int]) -> None:
    for seed in seeds:
        tmp_configs: list[str] = []
        for config_path in configs:
            base = load_config(config_path)
            cfg = deepcopy(base)
            cfg["seed"] = seed
            tmp = Path("outputs/logs/repeats") / f"low_resource_seed_{seed}_{Path(config_path).stem}.yaml"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            tmp_configs.append(str(tmp))
        await run_matrix(tmp_configs, f"outputs/repeats/low_resource/seed_{seed}")


async def _run_ablation(config_path: str, seeds: list[int]) -> None:
    for seed in seeds:
        base = load_config(config_path)
        cfg = deepcopy(base)
        cfg["seed"] = seed
        tmp = Path("outputs/logs/repeats") / f"ablation_seed_{seed}.yaml"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        await run_ablation(str(tmp), run_name=f"repeats/ablation/seed_{seed}")


async def _run_ood(configs: list[str], seeds: list[int]) -> None:
    for seed in seeds:
        for config_path in configs:
            base = load_config(config_path)
            cfg = deepcopy(base)
            cfg["seed"] = seed
            # Reuse the main-run output root so AgentCF can reuse generated checkpoints.
            cfg["output_root"] = f"outputs/repeats/main/seed_{seed}/{Path(config_path).stem}"
            await _run_with_config(cfg, f"ood_seed_{seed}_{Path(config_path).stem}.yaml", run_ood)


async def run_suite(tasks: list[str], seeds: list[int]) -> None:
    configs = ["configs/baseline.yaml", "configs/agentcf.yaml"]
    if "main" in tasks:
        await _run_main(configs, seeds)
    if "low_resource" in tasks:
        await _run_low_resource(configs, seeds)
    if "ablation" in tasks:
        await _run_ablation("configs/ablation.yaml", seeds)
    if "ood" in tasks:
        await _run_ood(configs, seeds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["main", "low_resource", "ablation", "ood"],
        choices=["main", "low_resource", "ablation", "ood"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()
    asyncio.run(run_suite(args.tasks, args.seeds))


if __name__ == "__main__":
    main()
