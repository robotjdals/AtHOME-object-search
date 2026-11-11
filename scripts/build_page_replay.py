"""Search replay data for the project page (docs/assets/search_replay.json): the
scene graph of the test home and, for every scenario of the final experiment,
both planners' searches as timed steps (decisions with their candidates,
visits, the find), read from the run records.

    python3 scripts/build_page_replay.py
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOM_NAMES = {"living_room": "Living room", "office": "Office", "kitchen": "Kitchen", "bedroom": "Bedroom"}
SHORT = {"vacuum cleaner": "vacuum", "clothes hanger": "hanger", "refrigerator": "fridge"}


def load_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def graph_view(graph):
    objects = {o["object_id"]: o for o in graph["objects"]}
    rooms = []
    for r in graph["rooms"]:
        locs = []
        for wid in r["workspace_ids"]:
            w = next(x for x in graph["workspaces"] if x["workspace_id"] == wid)
            src = objects[w["source_object_id"]]["semantic_tag"]
            locs.append({"id": wid, "kind": "workspace",
                         "label": f"{src} {w['source_object_id'].rsplit('_', 1)[1]}",
                         "objects": [objects[c]["semantic_tag"] for c in w["child_object_ids"]]})
        for oid in r["standalone_object_ids"]:
            tag = objects[oid]["semantic_tag"]
            locs.append({"id": f"standalone:{oid}", "kind": "standalone", "label": SHORT.get(tag, tag), "objects": []})
        rooms.append({"id": r["room_id"], "label": ROOM_NAMES.get(r["room_label"], r["room_label"]), "locations": locs})
    return rooms


def timeline(run_dir, record):
    events = load_lines(run_dir / record["events_file"])
    decisions = load_lines(run_dir / record["decisions_file"])
    t0 = next(e["stamp"] for e in events if e["event"] == "command_start")
    steps = []
    for d in decisions:
        steps.append({"t": round(d["stamp"] - t0, 1), "type": "decide", "stage": d["stage"],
                      "candidates": [c["id"] for c in d["candidates"]], "chosen": d["applied"],
                      "llm": d["raw"] is not None, "latency_s": d["latency_s"]})
    for e in events:
        t = round(e["stamp"] - t0, 1)
        if e["event"] == "visit_start":
            steps.append({"t": t, "type": "visit", "location": e["location_id"], "recheck": e["stage"] == "recheck"})
        elif e["event"] == "visit_end":
            steps.append({"t": t, "type": "visited", "location": e["location_id"], "found": bool(e.get("found"))})
        elif e["event"] == "command_end":
            target = (e.get("targets") or [{}])[0]
            steps.append({"t": t, "type": "end", "status": target.get("status"),
                          "location": target.get("found_location"), "label": target.get("found_label")})
    steps.sort(key=lambda s: (s["t"], s["type"] != "decide"))
    return steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", type=Path, default=ROOT / "outputs/room_baram/graph.json")
    ap.add_argument("--runs", type=Path, default=ROOT / "outputs/room_baram/experiments/room_baram_final")
    ap.add_argument("--plan", type=Path, default=ROOT / "experiments/room_baram_final/scenarios.csv")
    ap.add_argument("--output", type=Path, default=ROOT / "docs/assets/search_replay.json")
    args = ap.parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    plan = {int(r["scenario"]): r for r in csv.DictReader(open(args.plan, encoding="utf-8"))}
    runs = [r for r in load_lines(args.runs / "runs.jsonl") if r.get("valid")]
    scenarios = []
    for s in sorted(plan):
        row = plan[s]
        entry = {"scenario": s, "object": row["object"], "placed": row["location_id"],
                 "room": ROOM_NAMES.get(row["room"], row["room"]), "typical": row["typical"]}
        for r in runs:
            if r["scenario"] == s:
                entry[r["variant"]] = {"success": r["target_status"] == "found" and r["duration_s"] <= 600,
                                       "visits": r["steps_completed"], "time_s": r["duration_s"],
                                       "steps": timeline(args.runs, r)}
        scenarios.append(entry)
    out = {"rooms": graph_view(graph), "scenarios": scenarios}
    args.output.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{args.output}: {args.output.stat().st_size // 1024} KB, {len(scenarios)} scenarios")


if __name__ == "__main__":
    main()
