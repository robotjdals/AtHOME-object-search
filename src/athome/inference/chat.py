"""JSON-schema constrained chat completion for OpenAI-compatible servers
(OpenAI API, vLLM)."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Optional

from athome.inference.http import HttpError, get_json, post_json


def served_models(base_url: str, api_key: Optional[str], timeout: float) -> List[str]:
    """Model names of an OpenAI-compatible server (GET /v1/models, no tokens)."""
    reply = get_json(base_url.rstrip("/") + "/v1/models", timeout,
                     {"Authorization": f"Bearer {api_key}"} if api_key else None)
    return [m["id"] for m in reply["data"]]


class QuotaExceeded(RuntimeError):
    """Out of API credit / quota: retrying cannot help, stop the run."""


def _retryable(error: HttpError) -> bool:
    # Rate limits, server errors and lost connections are transient; a 429
    # for exhausted quota or credit is not.
    if error.status == 429 and any(k in error.detail for k in ("insufficient_quota", "credits")):
        return False
    return error.status is None or error.status == 429 or error.status >= 500


class ChatJSON:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: Optional[str] = None,
        timeout: float = 30.0,
        schema_name: str = "response",
        retries: int = 0,
        backoff_s: float = 1.0,
        # Server-specific request fields added as-is, e.g. vLLM's
        # {"chat_template_kwargs": {"enable_thinking": False}} for Qwen3.
        # None for the OpenAI API, which rejects unknown fields.
        extra_body: Optional[Dict] = None,
        # Output token limit (None: server default). Bounds runaway outputs.
        max_tokens: Optional[int] = None,
    ):
        headers = {}
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise RuntimeError(f"{api_key_env} 환경 변수 필요")
            headers["Authorization"] = f"Bearer {key}"
        self._headers = headers
        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self._timeout = timeout
        self._schema_name = schema_name
        self._extra_body = dict(extra_body or {})
        self._max_tokens = max_tokens
        # Exponential backoff with jitter for transient errors (OpenAI's
        # recommended handling of rate limits); 0 keeps a single attempt.
        self._retries, self._backoff_s = retries, backoff_s
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "requests": 0}
        self.last_usage: Optional[Dict] = None

    def __call__(self, messages: List[dict], schema: dict) -> Dict:
        return self.sample(messages, schema, n=1, temperature=0.0)[0]

    def sample(self, messages: List[dict], schema: dict, n: int,
               temperature: float) -> List[Dict]:
        """``n`` completions in one request (input billed once)."""
        body = {
            "model": self.model,
            "temperature": temperature,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": self._schema_name, "strict": True, "schema": schema},
            },
        }
        if n != 1:
            body["n"] = n
        if self._max_tokens is not None:
            body["max_tokens"] = self._max_tokens
        body.update(self._extra_body)
        for attempt in range(self._retries + 1):
            try:
                response = post_json(self._url, body, self._timeout, self._headers)
                break
            except HttpError as e:
                if e.status == 429 and not _retryable(e):
                    raise QuotaExceeded(str(e)) from e
                if attempt == self._retries or not _retryable(e):
                    raise
                time.sleep(self._backoff_s * 2 ** attempt * (1 + random.random()))
        usage = response.get("usage") or {}
        self.last_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "cached_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)),
        }
        for k, v in self.last_usage.items():
            self.usage[k] += v
        self.usage["requests"] += 1
        choices = response["choices"]
        if len(choices) != n:
            raise RuntimeError(f"응답 {len(choices)}개, 요청 {n}개")
        return [json.loads(c["message"]["content"]) for c in choices]
