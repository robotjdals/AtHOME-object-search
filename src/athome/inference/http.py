"""Minimal JSON-over-HTTP client (stdlib only, runs on the robot PC)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class HttpError(RuntimeError):
    pass


def post_json(url: str, body: dict, timeout: float, headers: dict = None) -> dict:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        raise HttpError(f"HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise HttpError(f"연결 실패: {e}") from e
    except json.JSONDecodeError as e:
        raise HttpError(f"JSON 아님: {e}") from e
