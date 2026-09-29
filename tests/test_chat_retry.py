import json

import pytest

import athome.inference.chat as chat
from athome.inference.http import HttpError


def reply():
    return {"choices": [{"message": {"content": json.dumps({"ok": True})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3,
                      "prompt_tokens_details": {"cached_tokens": 4}}}


def client(monkeypatch, outcomes, retries=3):
    calls = []

    def post_json(url, body, timeout, headers):
        calls.append(1)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(chat, "post_json", post_json)
    monkeypatch.setattr(chat.time, "sleep", lambda s: None)
    return chat.ChatJSON("http://x", "m", retries=retries), calls


def test_transient_errors_are_retried_and_usage_is_counted(monkeypatch):
    c, calls = client(monkeypatch, [HttpError("rate", 429, "rate_limit_exceeded"),
                                     HttpError("down", 503), HttpError("conn"), reply()])
    assert c([], {}) == {"ok": True} and len(calls) == 4
    assert c.usage == {"prompt_tokens": 10, "completion_tokens": 3, "cached_tokens": 4, "requests": 1}


def test_quota_stops_without_retry(monkeypatch):
    c, calls = client(monkeypatch, [HttpError("q", 429, '{"error": {"code": "insufficient_quota"}}')])
    with pytest.raises(chat.QuotaExceeded):
        c([], {})
    assert len(calls) == 1


def test_bad_request_is_not_retried(monkeypatch):
    c, calls = client(monkeypatch, [HttpError("bad", 400, "invalid schema")])
    with pytest.raises(HttpError):
        c([], {})
    assert len(calls) == 1
