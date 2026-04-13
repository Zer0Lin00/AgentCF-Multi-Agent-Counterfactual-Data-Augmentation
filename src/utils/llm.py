from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
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


def _iter_json_objects(text: str) -> list[str]:
    candidates: list[str] = []
    start = -1
    depth = 0
    in_string = False
    escaped = False
    for idx, ch in enumerate(text):
        if start == -1:
            if ch == "{":
                start = idx
                depth = 1
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidates.append(text[start : idx + 1])
                start = -1
    return candidates


def _extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    for candidate in _iter_json_objects(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    preview = text[:400].replace("\n", "\\n")
    raise ValueError(f"No valid JSON object found in model output: {preview}")


def _strip_code_fence(text: str) -> str:
    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return fenced_match.group(1).strip() if fenced_match else text.strip()


def _fallback_generator_payload(text: str) -> dict[str, Any]:
    normalized = _strip_code_fence(text)
    rows: list[dict[str, str]] = []

    # Salvage truncated multi-line candidate objects by slicing from one candidate_id to the next.
    id_matches = list(re.finditer(r'"candidate_id"\s*:\s*"([^"]+)"', normalized))
    for idx, match in enumerate(id_matches):
        candidate_id = match.group(1).strip()
        end = id_matches[idx + 1].start() if idx + 1 < len(id_matches) else len(normalized)
        chunk = normalized[match.start() : end]
        txt_match = re.search(r'"text"\s*:\s*', chunk)
        if not txt_match:
            continue
        candidate_text = chunk[txt_match.end() :].strip()
        candidate_text = candidate_text.lstrip('"').rstrip(",")
        candidate_text = re.sub(r'"\s*}\s*,?\s*{\s*$', "", candidate_text)
        candidate_text = re.sub(r'}\s*,?\s*{\s*$', "", candidate_text)
        candidate_text = re.sub(r',\s*{\s*$', "", candidate_text)
        candidate_text = re.sub(r'"\s*,?\s*[}\]]*\s*$', "", candidate_text)
        candidate_text = re.sub(r"[}\]]+\s*$", "", candidate_text)
        candidate_text = candidate_text.replace('\\"', '"')
        candidate_text = re.sub(r"\s+", " ", candidate_text).strip()
        if candidate_text:
            rows.append({"candidate_id": candidate_id, "text": candidate_text})

    if not rows:
        for raw_line in normalized.splitlines():
            line = raw_line.strip()
            if '"candidate_id"' not in line or '"text"' not in line:
                continue
            cid_match = re.search(r'"candidate_id"\s*:\s*"([^"]+)"', line)
            txt_match = re.search(r'"text"\s*:\s*(.+)', line)
            if not cid_match or not txt_match:
                continue
            candidate_id = cid_match.group(1).strip()
            candidate_text = txt_match.group(1).strip().rstrip(",")
            while candidate_text.endswith("}") or candidate_text.endswith("]"):
                candidate_text = candidate_text[:-1].rstrip()
            if candidate_text.startswith('"'):
                candidate_text = candidate_text[1:]
            if candidate_text.endswith('"'):
                candidate_text = candidate_text[:-1]
            candidate_text = candidate_text.replace('\\"', '"')
            if candidate_text.count('"') >= 2:
                candidate_text = candidate_text.replace('"', "")
            candidate_text = re.sub(r"\s+", " ", candidate_text).strip()
            if candidate_text:
                rows.append({"candidate_id": candidate_id, "text": candidate_text})

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["candidate_id"], row["text"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    if not deduped:
        preview = text[:400].replace("\n", "\\n")
        raise ValueError(f"No valid fallback generator candidates found: {preview}")
    return {"candidates": deduped}


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

    @staticmethod
    def _temperature_for(stage: str) -> float:
        return 0.2 if stage == "generator" else 0.0

    @staticmethod
    def _schema_hint(stage: str) -> str:
        if stage == "planner":
            return (
                '{"id":"...","target_label":0,'
                '"causal_features":["..."],"spurious_features":["..."],'
                '"elements_to_change":["..."],"elements_to_preserve":["..."],'
                '"editing_constraints":["..."]}'
            )
        if stage == "generator":
            return (
                '{"candidates":['
                '{"candidate_id":"c1","text":"..."},'
                '{"candidate_id":"c2","text":"..."}'
                "]}"
            )
        return (
            '{"label_score":0.0,"semantic_score":0.0,"minimality_score":0.0,'
            '"consistency_score":0.0,"final_score":0.0,"status":"pass","critique":""}'
        )

    async def _repair_json(self, *, model: str, stage: str, raw_text: str) -> str:
        if self._client is None:
            return ""
        repair_prompt = (
            "Convert the following text into one valid JSON object only. "
            "Do not add explanation or markdown. Preserve the original meaning as much as possible. "
            f"Expected schema example: {self._schema_hint(stage)}\n\n"
            f"Raw text:\n{raw_text}"
        )
        await self._limiter.wait()
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": repair_prompt}],
            temperature=0.0,
        )
        self.total_calls += 1
        return resp.choices[0].message.content or ""

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
                    temperature=self._temperature_for(stage),
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
                try:
                    return _extract_json_block(content)
                except ValueError:
                    repaired = await self._repair_json(model=model, stage=stage, raw_text=content)
                    if repaired.strip():
                        try:
                            return _extract_json_block(repaired)
                        except ValueError:
                            if stage == "generator":
                                return _fallback_generator_payload(repaired)
                    if stage == "generator":
                        return _fallback_generator_payload(content)
                    raise
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
            temperature=self._temperature_for(stage),
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
