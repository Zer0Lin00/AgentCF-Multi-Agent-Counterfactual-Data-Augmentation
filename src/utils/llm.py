from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from dotenv import dotenv_values, load_dotenv
from openai import AsyncOpenAI


class AsyncRateLimiter:
    def __init__(self, qps: float) -> None:
        self._interval = 1.0 / max(qps, 0.1)
        self._lock = asyncio.Lock()
        self._next_time = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            delay = self._next_time - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_time = max(self._next_time + self._interval, now + self._interval)


def _extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


class LLMClient:
    def __init__(self, config: dict[str, Any]) -> None:
        env_path = Path(".env")
        if env_path.exists():
            raw = dotenv_values(env_path)
            for k, v in raw.items():
                if k is None:
                    continue
                normalized_key = k.lstrip("\ufeff")
                if v is not None:
                    os.environ[normalized_key] = v
        else:
            load_dotenv(override=True)
        llm_cfg = config.get("llm", {})
        api_key = os.getenv(llm_cfg.get("api_key_env", "OPENAI_API_KEY"), "")
        base_url = os.getenv(llm_cfg.get("base_url_env", "OPENAI_BASE_URL"), "")
        self.enabled = bool(llm_cfg.get("enabled", True) and api_key)
        self._planner_model = os.getenv(llm_cfg.get("planner_model_env", "PLANNER_MODEL"), "gpt-5.4-mini")
        self._generator_model = os.getenv(llm_cfg.get("generator_model_env", "GENERATOR_MODEL"), "gpt-5.4-mini")
        self._verifier_model = os.getenv(llm_cfg.get("verifier_model_env", "VERIFIER_MODEL"), "gpt-5.4-mini")
        self._limiter = AsyncRateLimiter(config.get("runtime", {}).get("rate_limit_qps", 5))
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url) if self.enabled else None
        self.total_calls = 0

    def model_for(self, stage: str) -> str:
        return {
            "planner": self._planner_model,
            "generator": self._generator_model,
            "verifier": self._verifier_model,
        }.get(stage, self._generator_model)

    async def json_completion(self, *, stage: str, prompt: str, max_retries: int = 3) -> dict[str, Any]:
        if not self.enabled or self._client is None:
            raise RuntimeError("LLM client is disabled")
        model = self.model_for(stage)
        for retry in range(max_retries + 1):
            try:
                await self._limiter.wait()
                resp = await self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7 if stage == "generator" else 0.2,
                )
                self.total_calls += 1
                log_path = Path("outputs/logs/llm_calls.jsonl")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "stage": stage,
                                "model": model,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                content = resp.choices[0].message.content or ""
                if not content.strip():
                    content = await self._stream_text(model=model, prompt=prompt, stage=stage)
                if not content.strip():
                    raise ValueError("Empty model content from both non-stream and stream calls")
                return _extract_json_block(content)
            except Exception:
                if retry >= max_retries:
                    raise
                await asyncio.sleep(2**retry)
        raise RuntimeError("Unexpected retry loop exit")

    async def _stream_text(self, *, model: str, prompt: str, stage: str) -> str:
        if self._client is None:
            return ""
        await self._limiter.wait()
        stream = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7 if stage == "generator" else 0.2,
            stream=True,
        )
        self.total_calls += 1
        chunks: list[str] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                chunks.append(piece)
        return "".join(chunks)
