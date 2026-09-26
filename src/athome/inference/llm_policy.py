"""Small Planner LLM policy over an OpenAI-compatible server (e.g. vLLM).

Each stage uses its own LoRA adapter, served as a separate model name:
    vllm serve Qwen/Qwen3-4B --enable-lora \
        --lora-modules room=... workspace=... search_location=...

Adapters per stage (proposal 6-3): room -> Room SFT, workspace -> Workspace
GRPO, standalone -> Search Location SFT.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional, Sequence

from athome.inference.http import HttpError, post_json
from athome.inference.prompts import build_messages
from athome.search.policy import Candidate, PlanningContext, PolicyError, Stage

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMPolicy:
    def __init__(
        self,
        base_url: str,
        models: Dict[Stage, str],
        timeout: float = 5.0,
        retries: int = 1,
        api_key: Optional[str] = None,
        constrained: bool = True,
    ):
        missing = {Stage.ROOM, Stage.WORKSPACE, Stage.STANDALONE} - set(models)
        if missing:
            raise ValueError(f"모델 미지정 단계: {sorted(s.value for s in missing)}")
        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self._models = models
        self._timeout = timeout
        self._retries = retries
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._constrained = constrained
        self.last_raw: Optional[str] = None

    def select(
        self,
        stage: Stage,
        target: str,
        candidates: Sequence[Candidate],
        context: PlanningContext,
    ) -> str:
        if len(candidates) == 1:
            return candidates[0].candidate_id
        messages, aliases = build_messages(stage, target, candidates, context)
        body = {
            "model": self._models[stage],
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 32,
            # Qwen3: answer directly without the reasoning block.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self._constrained:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "selection",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "selected_id": {"type": "string", "enum": list(aliases)},
                        },
                        "required": ["selected_id"],
                    },
                },
            }

        error = "요청 안 함"
        for _ in range(self._retries + 1):
            try:
                response = post_json(self._url, body, self._timeout, self._headers)
                return aliases[self._parse(response, aliases)]
            except (HttpError, ValueError) as e:
                error = str(e)
        raise PolicyError(f"LLM 선택 실패 ({stage.value}): {error}")

    def _parse(self, response: dict, aliases) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"응답 형식 오류: {e}") from e
        self.last_raw = content
        text = _THINK.sub("", content or "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise ValueError(f"JSON 없음: {text[:80]!r}")
        try:
            selected = json.loads(match.group(0)).get("selected_id")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 파싱 실패: {text[:80]!r}") from e
        if selected not in aliases:
            raise ValueError(f"후보 밖 출력: {selected!r}")
        return selected
