import pytest

from athome.scene_graph.label_voting import aggregate
from athome.scene_graph.semantic_labeling import PROTOCOLS, label_rooms


def lab(label, *sources):
    return {"room_id": "_1", "room_label": label,
            "workspace_sources": [{"source_object_id": o, "function_label": f} for o, f in sources]}


def test_majority_sources_labels_and_votes():
    samples = ([lab("storage", ("shelf_1", "general_storage"), ("desk_2", "desk_work"))] * 5
               + [lab("utility_room", ("desk_2", "desk_work"))] * 3
               + [lab("storage", ("desk_2", "office"))])
    out = aggregate(samples, 5, ["storage", "utility_room"])
    assert out["room_label"] == "storage"
    assert out["workspace_sources"] == [
        {"source_object_id": "desk_2", "function_label": "desk_work"},
        {"source_object_id": "shelf_1", "function_label": "general_storage"}]
    assert out["votes"]["sources"] == {"desk_2": 9, "shelf_1": 5}
    assert not out["votes"]["room_label_tie"]


def test_minority_source_dropped_and_ties_flagged():
    samples = [lab("storage", ("box_3", "a"))] * 4 + [lab("garage")] * 4 + [lab("other")]
    out = aggregate(samples, 5, ["storage", "garage", "other"])
    assert out["workspace_sources"] == []
    assert out["room_label"] == "storage" and out["votes"]["room_label_tie"]  # vocabulary order


def test_invalid_inputs():
    with pytest.raises(ValueError):
        aggregate([], 1)
    with pytest.raises(ValueError):
        aggregate([lab("storage"), {**lab("storage"), "room_id": "_2"}], 1)
    with pytest.raises(ValueError):
        aggregate([lab("storage")], 2)


def test_label_rooms_uses_protocol_sample_count():
    room = {"room_id": "_1", "objects": [
        {"object_id": "shelf_1", "semantic_tag": "shelf", "bbox": {"size": [1, 1, 1]},
         "child_candidate_ids": []}]}
    calls = []

    def sample(messages, schema, n, temperature):
        calls.append((n, temperature))
        return [lab("storage", ("shelf_1", "general_storage"))] * n

    out = label_rooms({"rooms": [room]}, sample)
    assert calls == [(PROTOCOLS["v2"]["n"], PROTOCOLS["v2"]["temperature"])]
    assert out["rooms"][0]["votes"]["sources"] == {"shelf_1": 9}


def test_chat_sample_request_body_and_parsing(monkeypatch):
    import json
    from athome.inference import chat
    from athome.scene_graph.semantic_labeling import MODEL, OpenAIChat

    sent = {}
    answers = [lab("storage", ("shelf_1", "general_storage"))] * 7 + [lab("garage")] * 2

    def fake_post(url, body, timeout, headers):
        sent.update(body)
        return {"choices": [{"message": {"content": json.dumps(a)}} for a in answers]}

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(chat, "post_json", fake_post)
    client = OpenAIChat()
    room = {"room_id": "_1", "objects": [
        {"object_id": "shelf_1", "semantic_tag": "shelf", "bbox": {"size": [1, 1, 1]},
         "child_candidate_ids": []}]}
    out = label_rooms({"rooms": [room]}, client.sample)
    assert sent["model"] == MODEL == "gpt-4.1-2025-04-14"
    assert sent["n"] == 9 and sent["temperature"] == 1.0
    assert out["rooms"][0]["room_label"] == "storage"
    assert out["rooms"][0]["workspace_sources"][0]["source_object_id"] == "shelf_1"
    assert out["rooms"][0]["votes"]["sources"] == {"shelf_1": 7}
