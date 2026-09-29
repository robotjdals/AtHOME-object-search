"""Convert a Teacher run to grouped standalone candidates (prompt 0.6), no API.

Prompt 0.6 shows the standalone candidates of a room grouped by category
(athome.search.session.standalone_groups): the planner chooses a kind of
object and the robot goes to its nearest open instance. Recorded standalone
queries listed every instance; they are converted from the record alone:
- groups: the recorded instance candidates by category, nearest first (the
  same order and cost as the session);
- likelihood of a group: the largest Teacher likelihood of its instances (the
  Teacher judged every instance; same-name instances got the same value in
  almost all queries);
- label: the Teacher rule itself (athome.training.teacher.search_index_choice
  on the groups, for every recorded step cost);
- verification: athome.training.verify.GroundTruth on the nearest instance of
  the chosen group, from the recorded pose and Visited set (a group is valid
  if its nearest instance observes the target).
Room and workspace records are unchanged; every record's Student prompt is
rebuilt with the current prompts and must equal the stored one (room and
workspace text is the same in 0.5 and 0.6). The trajectory itself stays the
recorded one (state-action pairs, as in the export).
Writes a new run folder next to the input (<run>.grouped). Layout:
ATHOME_HM3D_LAYOUT.
"""
import argparse
import json
from pathlib import Path

from athome.inference.prompts import (
    PROMPT_VERSION, TEACHER_PROMPT_VERSION, build_messages, build_teacher_messages)
from athome.navigation import LocationCost, NavigationPlanner
from athome.schemas import Pose2D
from athome.search.policy import Stage
from athome.search.session import group_candidate
from athome.training.teacher import _cost_label, search_index_choice
from athome.training.verify import GroundTruth

from export_sft_dataset import planner_input
from scene_episodes import SceneProblems


def grouped(record):
    """Group candidates (nearest instance first) from the recorded instances."""
    room = record["context"]["room_id"]
    members = {}
    for c in record["candidates"]:
        members.setdefault(f"standalone_group:{room}:{c['info']['category']}", []).append(
            LocationCost(c["candidate_id"], c["cost"], ()))
    members = {gid: sorted(ms, key=lambda m: (m.cost, m.location_id)) for gid, ms in sorted(members.items())}
    return members, [group_candidate(gid, ms) for gid, ms in members.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher-run", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.teacher_run.with_name(args.teacher_run.name + ".grouped")
    if out_dir.exists():
        raise SystemExit(f"이미 존재하는 출력: {out_dir}")
    data = json.loads((args.teacher_run / "episodes.json").read_text(encoding="utf-8"))
    summary = data["summary"]
    lines = [json.loads(l) for l in (args.teacher_run / "queries.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    problems = SceneProblems().by_key()
    truths = {}
    stats = {"standalone_converted": 0, "rebuild_mismatch": 0, "became_single_group": 0,
             "label_changed_category": 0}
    out = []
    for record in lines:
        stage, candidates, context = planner_input(record)
        rebuilt, aliases = build_messages(stage, record["target"], candidates, context)
        if stage != Stage.STANDALONE:
            if rebuilt != record["student_messages"]:
                stats["rebuild_mismatch"] += 1
            record["prompt_version"], record["teacher_prompt_version"] = PROMPT_VERSION, TEACHER_PROMPT_VERSION
            out.append(record)
            continue
        members, groups = grouped(record)
        messages, aliases = build_messages(stage, record["target"], groups, context)
        teacher_messages, _ = build_teacher_messages(stage, record["target"], groups, context)
        new = {**record, "candidates": [{"candidate_id": g.candidate_id, "cost": g.cost, "info": g.info}
                                        for g in groups],
               "student_messages": messages, "teacher_messages": teacher_messages, "aliases": aliases,
               "single_candidate": len(groups) == 1, "prompt_version": PROMPT_VERSION,
               "teacher_prompt_version": TEACHER_PROMPT_VERSION,
               "ungrouped": {"candidates": len(record["candidates"]), "selected_id": record["selected_id"],
                             "verification": record["meta"].get("verification")}}
        meta = dict(record["meta"])
        if new["single_candidate"]:
            new["selected_id"] = groups[0].candidate_id
            new["selected_alias"] = next(iter(aliases))
            meta.pop("verification", None)
            meta.pop("passed_by_step_cost", None)
            stats["became_single_group"] += 1
        elif record["likelihoods"] and not record["error"]:
            likelihood = {gid: max(record["likelihoods"][m.location_id] for m in ms)
                          for gid, ms in members.items()}
            new["likelihoods"] = likelihood
            new["posterior"] = dict(likelihood)          # rooms only are discounted by search
            new["selection_by_step_cost"] = {k: search_index_choice(groups, likelihood, float(k))
                                             for k in record["selection_by_step_cost"]}
            key = _cost_label(summary["step_cost_m"])
            new["selected_id"] = new["selection_by_step_cost"][key]
            new["selected_alias"] = next(a for a, cid in aliases.items() if cid == new["selected_id"])
            old_cat = next(c["info"]["category"] for c in record["candidates"]
                           if c["candidate_id"] == record["selected_id"])
            stats["label_changed_category"] += old_cat != new["selected_id"].rsplit(":", 1)[1]
            # Verification of the nearest instance of each group.
            key_problem = (meta["component"], record["target"])
            if key_problem not in truths:
                p = problems[key_problem]
                navigation = NavigationPlanner(p.grid, p.nav_config)
                for lid, loc in p.graph.locations.items():
                    navigation.add_location(lid, loc.bbox_min, loc.bbox_max)
                truths[key_problem] = GroundTruth(p.graph, navigation, p.observer, p.target_ids,
                                                  p.gt_workspaces, gt_rooms=p.gt_rooms)
            truth = truths[key_problem]
            nearest = {gid: ms[0].location_id for gid, ms in members.items()}
            check = truth.verify("standalone", nearest[new["selected_id"]], list(nearest.values()),
                                 Pose2D(*meta["pose"]), set(meta["visited"]))
            valid = sorted(g for g, lid in nearest.items() if lid in check["valid_candidates"])
            meta["verification"] = {"passed": new["selected_id"] in valid, "valid_candidates": valid,
                                    "grouped_from_instances": True}
            meta["passed_by_step_cost"] = {k: v in valid for k, v in new["selection_by_step_cost"].items()}
        new["meta"] = meta
        stats["standalone_converted"] += 1
        out.append(new)
    out_dir.mkdir()
    with (out_dir / "queries.jsonl").open("x", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {**summary, "prompt_version": PROMPT_VERSION, "teacher_prompt_version": TEACHER_PROMPT_VERSION,
               "regrouped_from": str(args.teacher_run), "regroup_stats": stats}
    with (out_dir / "episodes.json").open("x", encoding="utf-8") as f:
        json.dump({"summary": summary, "episodes": data["episodes"]}, f, ensure_ascii=False, indent=2)
    print(args.teacher_run.name, stats)


if __name__ == "__main__":
    main()
