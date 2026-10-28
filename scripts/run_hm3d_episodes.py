"""Run exported HM3D episodes (scripts/export_hm3d_robot_episodes.py) on the
ROS search stack and compare each with the symbolic run of the same setup.

    source /opt/ros/humble/setup.bash && source install/setup.bash
    ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 python3 scripts/run_hm3d_episodes.py \
        outputs/hm3d_robot/GPyDUnjwZQy --setup robot --states GPyDUnjwZQy:A:pillow:1

Per episode: search_server with the setup's robot config, fake_motion_server
(starts at the episode's start pose, ends every motion exactly on the goal)
and fake_perception (targets visible from each goal cell under the training
observation model, within the camera view). The visits of the command
(/athome/search/events) are compared with the symbolic run: same locations
in the same order, same goal poses and planned costs, same result.
Results: <problem>/ros_<setup>.jsonl.

--setup robot_llm (the LLM planner through the robot config's server; the
SSH tunnel must be up and ATHOME_PLANNER_API_KEY in .env) has no symbolic
reference: the LLM is not deterministic across servers, so its runs are
compared by distribution with the training evaluation's Student
(--training-eval, evaluate_student.py --log-trajectory output).
"""
import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from athome_interfaces.action import SearchObjects

CAMERA_HFOV_DEG = 70.6      # configs/robot/demo.yaml sensors.camera_hfov_deg


def start(cmd, log, env=None):
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)


def stop(procs):
    for p in procs:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGINT)
    deadline = time.monotonic() + 15
    for p in procs:
        try:
            p.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()


def run_command(node, target, max_steps, timeout):
    """Send one search command; (events of the command, action status).

    The command's end is read from /athome/search/events (command_end): just
    after a server starts, a goal response can be lost although the goal was
    accepted (the client's response reader not yet matched), so an accepted
    command is recognized by its command_start event and never sent twice."""
    events = []
    sub = node.create_subscription(String, "/athome/search/events",
                                   lambda m: events.append(json.loads(m.data)), 100)
    client = ActionClient(node, SearchObjects, "/athome/search_objects")

    def seen(kind):
        return any(e["event"] == kind for e in events)

    def spin_until(done, seconds):
        end = time.monotonic() + seconds
        while not done() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        return done()

    try:
        if not client.wait_for_server(timeout_sec=30):
            raise RuntimeError("search_server 없음")
        goal = SearchObjects.Goal(targets=[target], max_steps=max_steps)
        deadline, handle = time.monotonic() + 30, None
        while not seen("command_start"):    # rejected until TF and the motion server are up
            future = client.send_goal_async(goal)
            spin_until(lambda: future.done() or seen("command_start"), 10)
            if future.done() and future.result() is not None and future.result().accepted:
                handle = future.result()
                break
            if time.monotonic() > deadline:
                raise RuntimeError("명령 거절됨")
            spin_until(lambda: seen("command_start"), 1.0)
        result = handle.get_result_async() if handle is not None else None
        if not spin_until(lambda: seen("command_end"), timeout):
            raise RuntimeError(f"{timeout}s 안에 끝나지 않음")
        spin_until(lambda: result is None or result.done(), 2)
        status = result.result().status if result is not None and result.done() else None
        return events, status
    finally:
        node.destroy_subscription(sub)
        client.destroy()


def metrics(events, step_cost=3.0):
    """found, visits, planned distance (sum of decision costs) and D + step_cost N."""
    visits = [e for e in events if e["event"] == "visit_start"]
    end = next((e for e in events if e["event"] == "command_end"), None)
    distance = sum(v["cost_m"] for v in visits)
    return {"found": end is not None and end["targets"][0]["status"] == "found",
            "visits": len(visits), "distance_m": distance, "cost": distance + step_cost * len(visits)}


def distribution(records, training):
    """Mean found / visits / distance / cost of the ROS runs and of the
    training evaluation's Student and MinCost on the same start states,
    with the paired cost difference."""
    import statistics

    rows = [(r, training[r["state_id"]]) for r in records if r["state_id"] in training and r["events"]]
    if not rows:
        return
    ros = [metrics(r["events"]) for r, _ in rows]
    print(f"분포 비교 (같은 시작 상태 {len(rows)}개):")
    for name, ms in (("ROS LLM", ros), ("학습 평가 Student", [t["student"] for _, t in rows]),
                     ("학습 평가 MinCost", [t["mincost"] for _, t in rows])):
        print(f"  {name:16} 성공 {statistics.mean(m['found'] for m in ms):.3f}  "
              f"방문 {statistics.mean(m['visits'] for m in ms):5.2f}  "
              f"거리 {statistics.mean(m['distance_m'] for m in ms):6.2f}  비용 {statistics.mean(m['cost'] for m in ms):6.2f}")
    diffs = [m["cost"] - t["student"]["cost"] for m, (_, t) in zip(ros, rows)]
    same = sum(abs(d) < 1e-3 for d in diffs)
    print(f"  비용 차이 (ROS - Student): 평균 {statistics.mean(diffs):+.2f}, 같은 에피소드 {same}/{len(diffs)}")


def compare(events, reference):
    """Differences between the ROS command and the symbolic run (empty: same)."""
    visits = [e for e in events if e["event"] == "visit_start"]
    end = next((e for e in events if e["event"] == "command_end"), None)
    diffs = []
    ours = [v["location_id"] for v in visits]
    theirs = [v["location_id"] for v in reference["trajectory"]]
    if ours != theirs:
        diffs.append(f"방문 순서: ROS {ours} / 기호 {theirs}")
    for v, r in zip(visits, reference["trajectory"]):
        if v["location_id"] != r["location_id"]:
            break
        if abs(v["cost_m"] - r["cost_m"]) > 1e-3:
            diffs.append(f"{v['location_id']} 비용 {v['cost_m']} / {r['cost_m']:.3f}")
        if math.hypot(v["goals"][0][0] - r["goal"][0], v["goals"][0][1] - r["goal"][1]) > 1e-3:
            diffs.append(f"{v['location_id']} 목표 {v['goals'][0][:2]} / {r['goal'][:2]}")
    found = end is not None and end["targets"][0]["status"] == "found"
    if found != reference["found"]:
        diffs.append(f"찾음: ROS {found} / 기호 {reference['found']}")
    return diffs, found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--setup", default="robot", help="robot (데이터셋 규칙) | robot_r037")
    parser.add_argument("--states", nargs="*", default=[])
    parser.add_argument("--max-steps", type=int, default=60, help="export와 같은 값")
    parser.add_argument("--speed", type=float, default=2.0, help="가짜 주행 속도 [m/s] (결과에 영향 없음)")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--training-eval", type=Path,
                        help="evaluate_student.py --log-trajectory 결과 (LLM 분포 비교)")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parents[1] / ".env",
                        help="search_server에 넘길 API 키 파일 (값은 출력하지 않음)")
    args = parser.parse_args()
    env = dict(os.environ)
    if args.env_file.is_file():
        from dotenv import dotenv_values

        env.update({k: v for k, v in dotenv_values(args.env_file).items() if v is not None})

    rclpy.init()
    node = Node("athome_hm3d_episodes")
    summary = []
    try:
        for problem in sorted(p for p in args.scene_dir.iterdir() if (p / "episodes.jsonl").is_file()):
            config = problem / f"{args.setup}.yaml"
            rows = [json.loads(l) for l in (problem / "episodes.jsonl").read_text().splitlines() if l.strip()]
            rows = [r for r in rows if not args.states or r["state_id"] in args.states]
            if not rows:
                continue
            with (problem / f"ros_{args.setup}.jsonl").open("a", encoding="utf-8") as out:
                for row in rows:
                    reference = row[args.setup]
                    if reference.get("start_not_free"):
                        print(f"{row['state_id']}: 출발 위치가 주행 불가 (건너뜀)")
                        continue
                    x, y, yaw = reference["start"]
                    log = (problem / f"ros_{args.setup}.{row['state_id'].split(':')[-1]}.log").open("w")
                    procs = [
                        start(["ros2", "run", "athome_ros", "search_server", "--ros-args",
                               "-p", f"robot_config:={config.resolve()}"], log, env),
                        start(["ros2", "run", "athome_ros", "fake_motion_server", "--ros-args",
                               "-p", f"robot_config:={config.resolve()}",
                               "-p", f"initial_pose:=[{x}, {y}, {yaw}]", "-p", "exact_arrival:=true",
                               "-p", f"linear_speed:={args.speed}", "-p", "angular_speed:=3.0"], log),
                        start(["ros2", "run", "athome_ros", "fake_perception", "--ros-args",
                               "-p", f"visibility:={(problem / 'visibility.json').resolve()}",
                               "-p", f"fov_deg:={CAMERA_HFOV_DEG}", "-p", "rate_hz:=20.0"], log),
                    ]
                    t0 = time.monotonic()
                    try:
                        events, status = run_command(node, row["target"], args.max_steps, args.timeout)
                        if "trajectory" in reference:
                            diffs, found = compare(events, reference)
                        else:
                            diffs, found = [], metrics(events)["found"]
                    except RuntimeError as e:
                        events, status, diffs, found = [], None, [f"실행 실패: {e}"], None
                    finally:
                        stop(procs)
                        log.close()
                    record = {"state_id": row["state_id"], "setup": args.setup, "action_status": status,
                              "found": found, "same": (not diffs) if "trajectory" in reference else None,
                              "differences": diffs,
                              "wall_s": round(time.monotonic() - t0, 1), "events": events}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    summary.append(record)
                    visits = sum(e["event"] == "visit_start" for e in events)
                    verdict = "기록" if record["same"] is None else "일치" if record["same"] else "불일치"
                    print(f"{row['state_id']} [{args.setup}]: {verdict}, "
                          f"방문 {visits}, 찾음 {found}, {record['wall_s']}s"
                          + "".join(f"\n    {d}" for d in diffs), flush=True)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    checked = [r for r in summary if r["same"] is not None]
    if checked:
        print(f"결과: {sum(r['same'] for r in checked)}/{len(checked)} 일치")
    if args.training_eval:
        rows = json.loads(args.training_eval.read_text(encoding="utf-8"))["rows"]
        distribution(summary, {r["state_id"]: r for r in rows})


if __name__ == "__main__":
    main()
