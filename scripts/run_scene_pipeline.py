"""Run the HM3DSem data pipeline of one scene, stage by stage.

A stage runs only when some of its outputs are missing, so the driver can be
re-run after every gate; finished stages are never overwritten. The two LLM
labelling batches are gates: the driver prints the submission command and
stops (external transfer and cost need approval, docs/hm3d_pipeline_v3_ko.md);
with ``--submit`` (approved runs) it submits, waits for the batch and goes on.
Once a batch was submitted (a state file with a batch ID exists) the next run
collects it and continues.

    python scripts/run_scene_pipeline.py --scene-dir data/scene_datasets/hm3d/minival/00800-TEEsavR23oF --version v4

Outputs: outputs/hm3d_<version>/<scene>.* and <scene>/ (graph, components,
grids, symbolic review), outputs/teacher_<version>/<scene>.* (batches, labels),
and the layout outputs/hm3d_<version>/<scene>.layout.json for the component
stage (ATHOME_HM3D_LAYOUT).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from export_sft_dataset import scene_split

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


class Gate(Exception):
    """Stop before a step that needs the user's approval."""


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def run(args, env=None):
    printable = " ".join(rel(a) if isinstance(a, Path) else str(a) for a in args)
    print(f"\n$ {printable}", flush=True)
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True,
                   env={**os.environ, **(env or {})})


def submitted(state):
    return state.exists() and json.loads(state.read_text(encoding="utf-8")).get("batch_id")


def wait_for_batch(state, poll_s=60, timeout_s=24 * 3600):
    """Poll a submitted batch until it ends (Batch API window is 24 h)."""
    from openai import OpenAI
    client, start = OpenAI(), time.time()
    batch_id = json.loads(state.read_text(encoding="utf-8"))["batch_id"]
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed":
            return
        if batch.status in ("failed", "expired", "cancelled"):
            raise SystemExit(f"Batch {batch_id}: {batch.status}")
        if time.time() - start > timeout_s:
            raise SystemExit(f"Batch {batch_id}: 대기 시간 초과 ({batch.status})")
        print(f"Batch {batch_id}: {batch.status} {batch.request_counts}", flush=True)
        time.sleep(poll_s)


def robot_traversable(robot_config):
    """The robot config states step/slope limits: its NavMesh leaves out
    stairs itself, so components are the NavMesh's connected pieces."""
    import yaml
    from athome.config import traversability
    return traversability(yaml.safe_load(Path(robot_config).read_text(encoding="utf-8"))["robot"]) is not None


def teacher_stage(a, output, env):
    if a.teacher == "openai" and not a.submit:
        raise Gate("Teacher 라벨링(API 비용)은 --submit과 함께 실행해야 합니다.")
    run([PY, "scripts/generate_teacher_episodes.py", "--teacher", a.teacher,
         "--starts", str(a.teacher_starts), "--max-steps", str(a.teacher_max_steps),
         "--output", output], env)


def grid_dir(scene_dir):
    report = scene_dir / "component_room_candidates.review.json"
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    return scene_dir / "component_grids" / f"{digest[:12]}_5cm"


def stages(a):
    S, V, T = a.scene_id, a.v, a.t
    D = V / S
    E = ROOT / f"outputs/eval_{a.version}" / S
    stair = D / "stair_triangle_selection.auto.json"
    robot_nav = D / "robot.navmesh"
    layout = V / f"{S}.layout.json"
    env = {"ATHOME_HM3D_LAYOUT": str(layout)}
    # Open-vocabulary targets: categories sampled per scene from the list
    # (scripts/build_target_categories.py); otherwise the fixed targets file.
    targets = V / f"{S}.targets.json" if a.target_categories else a.targets
    vote_state = T / f"{S}.semantic.vote.batch.state.json"
    masked_state = T / f"{S}.masked.semantic.batch.state.json"

    def gate(batch, state):
        def check():
            if not batch.read_text(encoding="utf-8").strip():
                print("새 라벨이 필요한 요청이 없어 제출하지 않습니다.")
                return
            if not submitted(state) and a.submit:
                run([PY, "scripts/submit_batch.py", "--input", batch])
            if submitted(state) and a.submit:
                wait_for_batch(state)
            if not submitted(state):
                raise Gate(f"LLM 라벨링 제출 승인 필요:\n  {PY} scripts/submit_batch.py --input {rel(batch)}"
                           "\n제출 후 이 명령을 다시 실행하면 결과를 수집하고 이어서 진행합니다.")
        return check

    def write_layout():
        layout.write_text(json.dumps({
            "scene_id": S, "root": rel(V), "graph": rel(V / f"{S}.workspace_graph.json"),
            "catalog": rel(V / f"{S}.target_catalog.json"), "annotations": rel(V / f"{S}.annotations.json"),
            "masked_graphs": rel(D / "masked_graphs"), "stair_selection": rel(stair),
            "targets": rel(targets),
            "robot": rel(a.robot),
            "search_locations": rel(ROOT / "configs/data/search_locations.json"),
        }, indent=2) + "\n", encoding="utf-8")

    return [
        ("annotations_export", [V / f"{S}.habitat_annotations.json"], lambda: run(
            [PY, "scripts/export_hm3d_annotations.py", "--scene", a.glb, "--dataset-config", a.dataset_config,
             "--output", V / f"{S}.habitat_annotations.json"])),
        ("mesh_bbox", [V / f"{S}.annotations.json"], lambda: run(
            [PY, "scripts/build_mesh_bbox_annotations.py", "--annotations", V / f"{S}.habitat_annotations.json",
             "--output", V / f"{S}.annotations.json", "--audit", V / f"{S}.mesh_bbox.audit.json"])),
        ("room_object_map", [V / f"{S}.room_object_map.json"], lambda: run(
            [PY, "-m", "athome.data.hm3d.converter", "--input", V / f"{S}.annotations.json",
             "--output", V / f"{S}.room_object_map.json"])),
        ("z_up", [V / f"{S}.room_object_map.z_up.json"], lambda: run(
            [PY, "-m", "athome.data.hm3d.coordinates", "--input", V / f"{S}.room_object_map.json",
             "--output", V / f"{S}.room_object_map.z_up.json"])),
        ("scope", [V / f"{S}.room_object_map.z_up.scoped.json"], lambda: run(
            [PY, "scripts/apply_scene_scope.py", "--input", V / f"{S}.room_object_map.z_up.json",
             "--config", ROOT / "configs/data/scene_scope.json",
             "--output", V / f"{S}.room_object_map.z_up.scoped.json"])),
        ("relations", [V / f"{S}.geometric_relations.json"], lambda: run(
            [PY, "-m", "athome.scene_graph.geometric_relations", "--input", V / f"{S}.room_object_map.z_up.scoped.json",
             "--output", V / f"{S}.geometric_relations.json", "--distance", "0.5", "--height", "0.1",
             "--overlap", "0.5"])),
        ("semantic_inputs", [V / f"{S}.semantic_inputs.json"], lambda: run(
            [PY, "-m", "athome.scene_graph.semantic_inputs", "--scene", V / f"{S}.room_object_map.z_up.scoped.json",
             "--geometry", V / f"{S}.geometric_relations.json", "--output", V / f"{S}.semantic_inputs.json"])),
        ("semantic_batch", [T / f"{S}.semantic.batch.jsonl"], lambda: (
            run([PY, "scripts/make_semantic_batch.py", "--input", V / f"{S}.semantic_inputs.json",
                 "--scene-id", S, "--output", T / f"{S}.semantic.batch.jsonl"]),
            run([PY, "scripts/validate_semantic_batch.py", "--input", V / f"{S}.semantic_inputs.json",
                 "--batch", T / f"{S}.semantic.batch.jsonl", "--scene-id", S]))),
        ("vote_batch", [T / f"{S}.semantic.vote.batch.jsonl"], lambda: run(
            [PY, "scripts/compact_semantic_batch.py", "--input", T / f"{S}.semantic.batch.jsonl",
             "--output", T / f"{S}.semantic.vote.batch.jsonl", "--protocol", a.protocol])),
        ("GATE vote_batch submission", [], gate(T / f"{S}.semantic.vote.batch.jsonl", vote_state)),
        ("collect_labels", [T / f"{S}.semantic_labels.vote.review.json"], lambda: run(
            [PY, "scripts/collect_semantic_batch.py", "--state", vote_state,
             "--labels-output", T / f"{S}.semantic_labels.vote.review.json"])),
        ("workspace_graph", [V / f"{S}.workspace_graph.json"], lambda: run(
            [PY, "scripts/build_hm3d_workspace_graph.py", "--semantic-inputs", V / f"{S}.semantic_inputs.json",
             "--labels", T / f"{S}.semantic_labels.vote.review.json",
             "--reviewed-output", T / f"{S}.semantic_labels.reviewed.json",
             "--graph-output", V / f"{S}.workspace_graph.json"])),
        *([("scene_targets", [targets], lambda: run(
            [PY, "scripts/select_scene_targets.py", "--scene-map", V / f"{S}.room_object_map.z_up.scoped.json",
             "--categories", a.target_categories, "--split", a.split,
             "--per-scene", str(a.targets_per_scene), "--output", targets]))]
          if a.target_categories else []),
        ("target_catalog", [V / f"{S}.target_catalog.json"], lambda: run(
            [PY, "scripts/build_target_catalog.py", "--graph", V / f"{S}.workspace_graph.json",
             "--config", targets, "--scene-id", S, "--output", V / f"{S}.target_catalog.json"])),
        ("masked_inputs", [D / "masked_inputs", D / "masking_audit"], lambda: run(
            [PY, "scripts/build_masked_inputs.py", "--semantic-inputs", V / f"{S}.semantic_inputs.json",
             "--config", targets, "--input-dir", D / "masked_inputs", "--audit-dir", D / "masking_audit"])),
        ("masked_batch", [T / f"{S}.masked.semantic.batch.jsonl", T / f"{S}.masked.semantic.manifest.json"],
         lambda: run(
            [PY, "scripts/make_masked_semantic_batch.py", "--scene-id", S, "--masked-dir", D / "masked_inputs",
             "--labels", T / f"{S}.semantic_labels.reviewed.json",
             "--baseline-batch", T / f"{S}.semantic.vote.batch.jsonl",
             "--batch-output", T / f"{S}.masked.semantic.batch.jsonl",
             "--manifest-output", T / f"{S}.masked.semantic.manifest.json", "--config", targets])),
        ("GATE masked_batch submission", [], gate(T / f"{S}.masked.semantic.batch.jsonl", masked_state)),
        ("collect_masked_labels", [T / f"{S}.masked_labels"], lambda: run(
            [PY, "scripts/collect_masked_semantic_batch.py", "--state", masked_state,
             "--masked-dir", D / "masked_inputs", "--labels-dir", T / f"{S}.masked_labels",
             "--config", targets])),
        ("masked_graphs", [D / "masked_graphs"], lambda: run(
            [PY, "scripts/build_masked_graphs.py", "--input-dir", D / "masked_inputs",
             "--label-dir", T / f"{S}.masked_labels", "--output-dir", D / "masked_graphs",
             "--catalog", V / f"{S}.target_catalog.json", "--config", targets])),
        ("layout", [layout], write_layout),
        ("floors", [D / "floor_environments.json"], lambda: run(
            [PY, "scripts/prepare_floor_environments.py", "--graph", V / f"{S}.workspace_graph.json",
             "--building-id", S, "--output", D / "floor_environments.json"])),
        ("robot_navmesh", [robot_nav], lambda: run(
            [PY, "scripts/build_robot_navmesh.py", "--scene", a.glb, "--dataset-config", a.dataset_config,
             "--navmesh", a.navmesh, "--robot-config", a.robot, "--output", robot_nav])),
        ("navmesh_levels", [stair], lambda: run(
            [PY, "scripts/segment_navmesh_levels.py", "--navmesh", rel(robot_nav), "--scene-id", S,
             *(["--mode", "islands"] if robot_traversable(a.robot) else []),
             "--output", stair])),
        ("component_rooms", [D / "component_room_candidates.review.json"], lambda: run(
            [PY, "scripts/inspect_component_rooms.py"], env)),
        ("membership", [D / "component_membership.review.json"], lambda: run(
            [PY, "scripts/prepare_component_membership.py"], env)),
        ("membership_decisions", [D / "component_membership.decisions.review.json"], lambda: run(
            [PY, "scripts/decide_component_membership.py"], env)),
        ("component_graphs", [D / "component_graphs.review"], lambda: run(
            [PY, "scripts/build_component_graphs_review.py"], env)),
        ("grids", lambda: [grid_dir(D) / "A.metadata.json"], lambda: run(
            [PY, "scripts/build_component_grids.py"], env)),
        ("goal_candidates", lambda: [grid_dir(D) / "workspace_goal_candidates.clearance_0p1.review.json"],
         lambda: run([PY, "scripts/generate_workspace_goal_candidates.py"], env)),
        ("workspace_paths", lambda: [grid_dir(D) / "path_review/workspace_paths.review.json"], lambda: run(
            [PY, "scripts/check_workspace_paths.py"], env)),
        ("masked_workspace_paths", lambda: [grid_dir(D) / "masked_workspace_checks/summary.review.json"],
         lambda: run([PY, "scripts/check_masked_workspace_paths.py"], env)),
        ("component_masked_graphs", [D / "component_masked_graphs.review"], lambda: run(
            [PY, "scripts/build_component_masked_graphs.py"], env)),
        ("component_navigation", [D / "component_navigation.review.json"], lambda: run(
            [PY, "scripts/check_component_navigation.py"], env)),
        ("symbolic_review", [D / "symbolic_wall_los_2d.include_absent.review.json"], lambda: run(
            [PY, "scripts/run_symbolic_review.py", "--max-steps", "30", "--include-absent",
             "--output", D / "symbolic_wall_los_2d.include_absent.review.json"], env)),
        # Fixed findable start states: Val/Test evaluation and GRPO rollouts.
        ("start_states", [E / "start_states.jsonl"], lambda: run(
            [PY, "scripts/export_start_states.py", "--starts", str(a.eval_starts),
             "--output", E / "start_states.jsonl"], env)),
        # Teacher episodes for SFT (API cost: only with --teacher openai and --submit).
        ("teacher", [T / f"episodes_{S}" / "episodes.json"] if a.teacher != "none" else [E / "start_states.jsonl"],
         lambda: teacher_stage(a, T / f"episodes_{S}", env)),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-dir", type=Path, required=True,
                        help="<scene>.basis.glb/.navmesh/.semantic.glb/.semantic.txt 폴더")
    parser.add_argument("--dataset-config", type=Path,
                        help="기본값: 장면 폴더 두 단계 위의 hm3d_annotated_basis.scene_dataset_config.json")
    parser.add_argument("--version", required=True, help="출력 폴더 outputs/hm3d_<version>, outputs/teacher_<version>")
    parser.add_argument("--protocol", default="v2", help="라벨링 프로토콜 (semantic_labeling.PROTOCOLS)")
    parser.add_argument("--robot", type=Path, default=ROOT / "configs/robot/demo.yaml",
                        help="실제 로봇 설정(반경·goal offset; A*·Nav2·MPPI와 공유)")
    parser.add_argument("--target-categories", type=Path,
                        help="개방 어휘 목표 범주 목록(build_target_categories.py). 주면 장면마다 목표를 뽑음")
    parser.add_argument("--targets-per-scene", type=int, default=12)
    parser.add_argument("--targets", type=Path, default=ROOT / "configs/data/pilot_targets.v0.2.json",
                        help="목표 범주·별칭 설정")
    parser.add_argument("--submit", action="store_true",
                        help="LLM batch를 직접 제출하고 완료까지 기다림(외부 전송·비용; 승인된 경우에만)")
    parser.add_argument("--teacher", choices=["none", "mincost", "openai"], default="none",
                        help="Teacher 에피소드 생성 (openai는 --submit 필요, API 비용)")
    parser.add_argument("--teacher-starts", type=int, default=2, help="목표 조합당 Teacher 시작 위치 수")
    parser.add_argument("--teacher-max-steps", type=int, default=60, help="Teacher 에피소드 방문 상한(비용 관리)")
    parser.add_argument("--eval-starts", type=int, default=5, help="목표 조합당 평가·GRPO 시작 상태 수")
    parser.add_argument("--dry-run", action="store_true", help="단계별 상태만 출력")
    a = parser.parse_args()
    scene_dir = (ROOT / a.scene_dir).resolve()
    a.scene_id = scene_dir.name.split("-", 1)[-1]
    a.glb = scene_dir / f"{a.scene_id}.basis.glb"
    a.navmesh = (ROOT / a.scene_dir / f"{a.scene_id}.basis.navmesh")
    a.dataset_config = (a.dataset_config or scene_dir.parents[1]
                        / "hm3d_annotated_basis.scene_dataset_config.json").resolve()
    for path in (a.glb, a.navmesh, a.dataset_config, scene_dir / f"{a.scene_id}.semantic.txt"):
        if not path.exists():
            raise SystemExit(f"입력 없음(주석이 없는 장면일 수 있음): {path}")
    a.targets = (ROOT / a.targets).resolve()
    if a.target_categories:
        a.target_categories = (ROOT / a.target_categories).resolve()
        split_config = json.loads((ROOT / "configs/data/scene_splits.json").read_text(encoding="utf-8"))
        a.split = scene_split(a.glb, split_config)   # building-level Train/Val/Test
    a.robot = (ROOT / a.robot).resolve()
    a.v, a.t = ROOT / f"outputs/hm3d_{a.version}", ROOT / f"outputs/teacher_{a.version}"
    for d in (a.v / a.scene_id, a.t):
        d.mkdir(parents=True, exist_ok=True)

    for name, outputs, action in stages(a):
        try:
            paths = outputs() if callable(outputs) else outputs
        except FileNotFoundError:
            paths = None  # depends on an earlier output that does not exist yet
        done = bool(paths) and all(p.exists() for p in paths)
        if a.dry_run:
            print(f"{'완료' if done else '대기'}  {name}")
            continue
        if done:
            print(f"[완료] {name}")
            continue
        print(f"\n=== {name} ===", flush=True)
        try:
            action()
        except Gate as gate:
            print(f"\n[중단] {gate}")
            return
        except subprocess.CalledProcessError as error:
            raise SystemExit(f"[실패] {name}: 종료 코드 {error.returncode}")
    if not a.dry_run:
        print(f"\n{a.scene_id}: 모든 단계 완료. layout: {rel(a.v / f'{a.scene_id}.layout.json')}")


if __name__ == "__main__":
    main()
