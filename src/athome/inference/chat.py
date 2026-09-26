"""JSON-schema constrained chat completion for OpenAI-compatible servers
(OpenAI API, vLLM)."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from athome.inference.http import post_json


class ChatJSON:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: Optional[str] = None,
        timeout: float = 30.0,
        schema_name: str = "response",
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

    def __call__(self, messages: List[dict], schema: dict) -> Dict:
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": self._schema_name, "strict": True, "schema": schema},
            },
        }
        response = post_json(self._url, body, self._timeout, self._headers)
        return json.loads(response["choices"][0]["message"]["content"])
