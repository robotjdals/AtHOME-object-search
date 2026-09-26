"""CLIP text embeddings from the GPU server (scripts/serve_clip_text.py).

The model must be the one perception uses for image features, otherwise
the cosine similarity is meaningless.
"""

from __future__ import annotations

from typing import List, Sequence

from athome.inference.http import post_json


class HttpTextEncoder:
    def __init__(self, base_url: str, model: str, timeout: float = 3.0):
        self._url = base_url.rstrip("/") + "/embed_text"
        self.model = model
        self._timeout = timeout

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        response = post_json(
            self._url, {"model": self.model, "texts": list(texts)}, self._timeout)
        if response.get("model") != self.model:
            raise ValueError(
                f"CLIP 모델 불일치: 요청 {self.model}, 응답 {response.get('model')}")
        embeddings = response["embeddings"]
        if len(embeddings) != len(texts):
            raise ValueError("임베딩 개수 불일치")
        return embeddings
