from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from typing import Any

LLM_METHODS = {
    "Single-LLM Counterfactual",
    "Single-LLM + Filtering",
    "AgentCF (Ours)",
}


def uses_llm_methods(cfg: dict[str, Any]) -> bool:
    methods = set(cfg.get("methods", []))
    return bool(methods & LLM_METHODS)


def _base_url(cfg: dict[str, Any]) -> str:
    llm_cfg = cfg.get("llm", {})
    return os.getenv(llm_cfg.get("base_url_env", "OPENAI_BASE_URL"), "")


def local_vllm_required(cfg: dict[str, Any]) -> bool:
    base_url = _base_url(cfg)
    return uses_llm_methods(cfg) and ("localhost" in base_url or "127.0.0.1" in base_url)


def start_local_vllm(cfg: dict[str, Any], *, screen_name: str = "vllm_auto") -> None:
    if not local_vllm_required(cfg):
        return
    launch_cmd = os.getenv("VLLM_LAUNCH_CMD", "").strip()
    if not launch_cmd:
        return
    subprocess.run(["screen", "-S", screen_name, "-X", "quit"], check=False)
    subprocess.run(["screen", "-wipe"], check=False)
    subprocess.run(["screen", "-dmS", screen_name, "bash", "-lc", launch_cmd], check=True)
    wait_for_vllm(cfg)


def wait_for_vllm(cfg: dict[str, Any], *, timeout_s: int = 180) -> None:
    if not local_vllm_required(cfg):
        return
    base_url = _base_url(cfg).rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    health_url = f"{base_url}/health"
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for local vLLM at {health_url}: {last_error}")


def stop_local_vllm(cfg: dict[str, Any], *, screen_name: str = "vllm_auto") -> None:
    if not local_vllm_required(cfg):
        return
    subprocess.run(["screen", "-S", screen_name, "-X", "quit"], check=False)
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
    subprocess.run(["screen", "-wipe"], check=False)
    time.sleep(2)
