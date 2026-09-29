"""Collect and analyze the label-stability batch (read-only for graphs).

Per input tag and room: room-label distribution and per-object selection
frequency over the n sampled completions. Compares with the single
temperature-0 labels, and between tags for objects whose own compact input
(size, child-candidate counts) is unchanged, which separates sampling
ambiguity from sensitivity to the rest of the room.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from openai import OpenAI

AMBIGUOUS = (0.2, 0.8)


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def user_payload(body):
    return json.loads(next(m["content"] for m in body["messages"] if m["role"] == "user"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--reference", action="append", required=True,
                        help="TAG=temperature 0 라벨(*.semantic_labels.review.json)")
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    raw = Path(state["input_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != state["input_sha256"]:
        raise SystemExit("제출 이후 입력 파일이 변경됐습니다.")
    requests = {r["custom_id"]: r for r in (json.loads(l) for l in raw.splitlines() if l.strip())}
    base = args.state.parent
    prefix = args.state.name[:-len(".state.json")]
    output = base / f"{prefix}.output.jsonl"
    if not output.exists():
        client = OpenAI()
        batch = client.batches.retrieve(state["batch_id"])
        if batch.status != "completed":
            raise SystemExit(f"아직 완료되지 않았습니다: {batch.status}")
        output.write_text(client.files.content(batch.output_file_id).text, encoding="utf-8")

    samples = {}   # (tag, room) -> list of labels
    usage = Counter()
    for row in load_jsonl(output):
        cid = row["custom_id"]
        if cid not in requests or row.get("error") or row["response"]["status_code"] != 200:
            raise SystemExit(f"{cid}: 응답 오류")
        body = row["response"]["body"]
        schema = requests[cid]["body"]["response_format"]["json_schema"]["schema"]
        labels = []
        for choice in body["choices"]:
            if choice["finish_reason"] != "stop" or choice["message"].get("refusal"):
                raise SystemExit(f"{cid}: 비정상 종료")
            label = json.loads(choice["message"]["content"])
            Draft202012Validator(schema).validate(label)
            labels.append(label)
        _, tag, _, room = cid.split(":")
        samples[(tag, room)] = labels
        for k in ("prompt_tokens", "completion_tokens"):
            usage[k] += body["usage"][k]
    if set(samples) != {tuple(c.split(":")[i] for i in (1, 3)) for c in requests}:
        raise SystemExit("누락된 응답")

    reference = {}
    for item in args.reference:
        tag, path = item.split("=", 1)
        for room in json.loads(Path(path).read_text(encoding="utf-8"))["rooms"]:
            reference[(tag, room["room_id"])] = room
    inputs = {}
    for cid, req in requests.items():
        _, tag, _, room = cid.split(":")
        inputs[(tag, room)] = {o["id"]: o for o in user_payload(req["body"])["objects"]}

    rows, rooms = [], []
    for (tag, room), labels in sorted(samples.items()):
        n = len(labels)
        freq = Counter(s["source_object_id"] for l in labels for s in l["workspace_sources"])
        t0 = {s["source_object_id"] for s in reference[(tag, room)]["workspace_sources"]}
        room_labels = Counter(l["room_label"] for l in labels)
        rooms.append({"tag": tag, "room_id": room, "n": n,
                      "room_label_distribution": dict(room_labels),
                      "room_label_t0": reference[(tag, room)]["room_label"]})
        for oid in sorted(set(freq) | t0):
            obj = inputs[(tag, room)][oid]
            rows.append({"tag": tag, "room_id": room, "object_id": oid,
                         "category": obj["category"], "frequency": freq[oid] / n,
                         "selected_t0": oid in t0})

    by_key = {(r["tag"], r["room_id"], r["object_id"]): r for r in rows}
    tags = sorted({r["tag"] for r in rows})
    cross = []
    if len(tags) == 2:
        a, b = tags
        for (tag, room, oid), r in by_key.items():
            if tag != a:
                continue
            other = by_key.get((b, room, oid), {"frequency": 0.0, "selected_t0": False})
            same_input = inputs[(a, room)].get(oid) == inputs[(b, room)].get(oid)
            cross.append({"room_id": room, "object_id": oid, "own_input_identical": same_input,
                          f"freq_{a}": r["frequency"], f"freq_{b}": other["frequency"],
                          f"t0_{a}": r["selected_t0"], f"t0_{b}": other["selected_t0"]})
        for (tag, room, oid), r in by_key.items():
            if tag == b and (a, room, oid) not in by_key:
                same_input = inputs[(a, room)].get(oid) == inputs[(b, room)].get(oid)
                cross.append({"room_id": room, "object_id": oid, "own_input_identical": same_input,
                              f"freq_{a}": 0.0, f"freq_{b}": r["frequency"],
                              f"t0_{a}": False, f"t0_{b}": r["selected_t0"]})

    def summary(tag):
        tr = [r for r in rows if r["tag"] == tag]
        amb = [r for r in tr if AMBIGUOUS[0] < r["frequency"] < AMBIGUOUS[1]]
        majority_disagree = [r for r in tr if (r["frequency"] > 0.5) != r["selected_t0"]]
        rl = [x for x in rooms if x["tag"] == tag]
        return {"candidate_objects": len(tr), "ambiguous_objects": len(amb),
                "t0_vs_majority_disagreements": len(majority_disagree),
                "rooms_with_label_variation": sum(len(x["room_label_distribution"]) > 1 for x in rl)}

    report = {
        "status": "measurement_only", "batch_id": state["batch_id"],
        "ambiguous_range": AMBIGUOUS, "usage": dict(usage),
        "summary": {t: summary(t) for t in tags},
        "rooms": rooms, "objects": rows, "cross_tag": cross,
    }
    path = base / f"{prefix}.report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=1))
    print("usage", dict(usage))
    print("\n애매한 객체 (0.2 < 선택률 < 0.8):")
    for r in rows:
        if AMBIGUOUS[0] < r["frequency"] < AMBIGUOUS[1]:
            print(f"  {r['tag']:5} {r['room_id']:4} {r['object_id']:24} {r['frequency']:.1f} t0={r['selected_t0']}")
    if cross:
        print("\n자기 입력이 같은데 선택률이 0.5 이상 차이:")
        for c in cross:
            f = [v for k, v in c.items() if k.startswith("freq_")]
            if c["own_input_identical"] and abs(f[0] - f[1]) >= 0.5:
                print("  ", c)
    print("저장:", path)


if __name__ == "__main__":
    main()
