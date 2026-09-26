import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from athome.inference.http import HttpError
from athome.inference.llm_policy import LLMPolicy
from athome.inference.prompts import build_messages
from athome.inference.text_encoder import HttpTextEncoder
from athome.schemas import ObservedObject
from athome.search.matching import ClipMatcher
from athome.search.policy import Candidate, PlanningContext, PolicyError, Stage

ROOMS = [
    Candidate("R_A", 1.0, {"room_label": "kitchen",
                           "workspaces": [{"category": "table", "function_label": "dining"}],
                           "standalone_categories": ["fridge"]}),
    Candidate("R_B", 5.0, {"room_label": "living room", "workspaces": [],
                           "standalone_categories": []}),
]
MODELS = {Stage.ROOM: "room", Stage.WORKSPACE: "workspace", Stage.STANDALONE: "location"}


class FakeServer:
    """OpenAI-compatible stand-in. ``reply(body) -> (status, payload, delay)``."""

    def __init__(self, reply):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                status, payload, delay = reply(body)
                time.sleep(delay)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self):
        self._server.shutdown()


def chat(content):
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def server():
    servers = []

    def make(reply):
        s = FakeServer(reply)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def test_prompt_uses_aliases_and_semantic_info():
    messages, aliases = build_messages(Stage.ROOM, "cup", ROOMS, PlanningContext())
    assert aliases == {"C1": "R_A", "C2": "R_B"}
    text = messages[1]["content"]
    assert "Target: cup" in text and "kitchen" in text and "table (dining)" in text
    assert "R_A" not in text     # graph IDs are hidden behind aliases


def test_llm_policy_maps_alias_and_uses_stage_model(server):
    s = server(lambda body: (200, chat('{"selected_id": "C2"}'), 0))
    policy = LLMPolicy(s.url, MODELS)
    assert policy.select(Stage.ROOM, "cup", ROOMS, PlanningContext()) == "R_B"
    path, body = s.requests[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "room"
    assert body["response_format"]["json_schema"]["schema"]["properties"][
        "selected_id"]["enum"] == ["C1", "C2"]


def test_llm_policy_strips_thinking_block(server):
    s = server(lambda body: (200, chat('<think>hmm</think>\n{"selected_id": "C1"}'), 0))
    assert LLMPolicy(s.url, MODELS).select(Stage.ROOM, "cup", ROOMS, PlanningContext()) == "R_A"


def test_llm_policy_rejects_out_of_candidate_output(server):
    s = server(lambda body: (200, chat('{"selected_id": "C9"}'), 0))
    with pytest.raises(PolicyError, match="후보 밖"):
        LLMPolicy(s.url, MODELS, retries=1).select(Stage.ROOM, "cup", ROOMS, PlanningContext())
    assert len(s.requests) == 2       # retried once


def test_llm_policy_timeout_raises_policy_error(server):
    s = server(lambda body: (200, chat('{"selected_id": "C1"}'), 1.0))
    policy = LLMPolicy(s.url, MODELS, timeout=0.2, retries=0)
    with pytest.raises(PolicyError):
        policy.select(Stage.ROOM, "cup", ROOMS, PlanningContext())


def test_llm_policy_server_down_raises_policy_error():
    policy = LLMPolicy("http://127.0.0.1:9", MODELS, timeout=0.5, retries=0)
    with pytest.raises(PolicyError):
        policy.select(Stage.ROOM, "cup", ROOMS, PlanningContext())


def test_single_candidate_skips_the_call():
    policy = LLMPolicy("http://127.0.0.1:9", MODELS, timeout=0.1)
    assert policy.select(Stage.ROOM, "cup", ROOMS[:1], PlanningContext()) == "R_A"


def unit(*v):
    v = np.asarray(v, float)
    return tuple(v / np.linalg.norm(v))


def obj(label, feature=None):
    return ObservedObject(0, label, 0.9, (0, 0, 0), clip_feature=feature)


def test_clip_matcher_threshold_and_fallback():
    texts = {"a photo of a cup": [1.0, 0.0, 0.0]}
    matcher = ClipMatcher(lambda ts: [texts[t] for t in ts], threshold=0.8)
    assert matcher.matches("cup", obj("mug", unit(0.9, 0.1, 0.0)))       # similar feature
    assert not matcher.matches("cup", obj("cup", unit(0.0, 1.0, 0.0)))   # feature disagrees
    assert matcher.matches("cup", obj("cup"))                            # no feature -> label


def test_clip_matcher_encoder_failure_falls_back_to_label():
    def broken(texts):
        raise HttpError("down")

    matcher = ClipMatcher(broken, threshold=0.8)
    assert matcher.matches("cup", obj("cup", unit(1, 0, 0)))
    assert not matcher.matches("cup", obj("mug", unit(1, 0, 0)))
    assert "cup" in matcher.errors


def test_http_text_encoder_checks_model(server):
    s = server(lambda body: (200, {"model": body["model"], "embeddings": [[0.1, 0.2]]}, 0))
    assert HttpTextEncoder(s.url, "ViT-B-32")(["cup"]) == [[0.1, 0.2]]
    s2 = server(lambda body: (200, {"model": "other", "embeddings": [[0.1]]}, 0))
    with pytest.raises(ValueError, match="불일치"):
        HttpTextEncoder(s2.url, "ViT-B-32")(["cup"])


def test_command_parser_normalizes_and_dedupes():
    from athome.inference.command_parser import parse_command

    out = parse_command("컵과 물병, 그리고 컵을 찾아줘",
                        lambda m, s: {"targets": ["Cup", "water  bottle", "cup"]})
    assert out == ["cup", "water bottle"]


def test_command_parser_errors():
    from athome.inference.command_parser import CommandParseError, parse_command

    with pytest.raises(CommandParseError):
        parse_command("  ", lambda m, s: {"targets": ["cup"]})
    with pytest.raises(CommandParseError, match="찾을 물체"):
        parse_command("안녕", lambda m, s: {"targets": []})
    with pytest.raises(CommandParseError, match="해석 실패"):
        parse_command("컵", lambda m, s: {"wrong": 1})


def test_command_parser_over_http(server):
    from athome.inference.chat import ChatJSON
    from athome.inference.command_parser import parse_command

    s = server(lambda body: (200, chat('{"targets": ["cup", "remote control"]}'), 0))
    assert parse_command("컵이랑 리모컨", ChatJSON(s.url, "gpt-4.1")) == ["cup", "remote control"]
    _, body = s.requests[0]
    assert body["response_format"]["json_schema"]["strict"] is True
