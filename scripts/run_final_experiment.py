"""Planner comparison from a terminal: the same runner as the operator console's
Experiment tab (athome_ros.experiment), for when no browser is at hand.

    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 scripts/run_final_experiment.py --robot-config configs/robot/room_baram.yaml \\
        --plan experiments/room_baram_final/scenarios.csv --out outputs/room_baram/experiments/room_baram_final

The console tab writes the same records when launched with
experiment_plan:=$PWD/experiments/room_baram_final/scenarios.csv (default out:
<graph dir>/experiments/<plan folder>, the path above). Use one of the two at a
time. Commands: save <start> (the robot standing there), check, start; during a
session the prompt asks for what the phase needs. Ctrl-C aborts the session
(the running move or command is canceled first).
"""
import argparse
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from athome.config import load_robot_config
from athome_ros.experiment import ExperimentRunner


def show(view):
    p = view["progress"]
    print(f"\n[{view['phase_text']}] 유효 {p['valid']}/{p['total']} 무효 {p['invalid']} 중단 {p['aborted']}"
          + (f" · planner {view['variant']}" if view["variant"] else ""))
    sc = view["scenario"]
    if sc:
        print(f"  시나리오 {sc['scenario']}: {sc['object']} → {sc['room']} {sc['desk']} · 시작 {sc['start']}")
    if view["text"]:
        print("  " + view["text"])
    r = view["record"]
    if view["phase"] == "verdict" and r:
        print(f"  결과: {r.get('target_status')} ({r.get('status')}) · 방문 {r.get('steps_completed')} · "
              f"{r.get('duration_s')}초 · 발견 {r.get('found_label')} @ {r.get('found_location')} · "
              f"사진 {r.get('found_photo')} · planner 대체 {r.get('fallbacks')}/{r.get('decisions')}")


def wait_change(exp, phase, text):
    while True:
        v = exp.view()
        if v["phase"] != phase or v["text"] != text:
            return v
        time.sleep(0.3)


def session(exp):
    v = exp.view()
    show(v)
    while exp.busy():
        phase, text = v["phase"], v["text"]
        if phase == "place":
            a = input("  Enter=놓았음, s=건너뛰기: ").strip()
            print(exp.act("skip" if a == "s" else "placed").get("message", ""))
        elif phase == "manual":
            input("  조치했으면 Enter: ")
            exp.act("continue")
        elif phase == "verdict":
            a = input("  Enter=유효, x 이유=무효: ").strip()
            if a.startswith("x"):
                exp.act("verdict", valid=False, note=a[1:].strip() or "사유 없음")
            else:
                exp.act("verdict", valid=True)
        v = wait_change(exp, phase, text)
        show(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = load_robot_config(args.robot_config)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("athome_experiment_cli")
    exp = ExperimentRunner(node, config, args.plan, args.out)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    print(f"시작 위치: {', '.join(exp.view()['start_names'])}")
    print("명령: save <시작 위치 이름> | check | start | q")
    try:
        while True:
            line = input("> ").strip()
            if line in ("q", "quit"):
                break
            if line.startswith("save "):
                print(exp.save_start(line[5:].strip())["message"])
            elif line == "check":
                r = exp.check()
                for row in r["rows"]:
                    print(f"  {row['scenario']:>2} {row['object']:<8} {row['start']:<12} "
                          f"{row['l_star_m'] if row['l_star_m'] is not None else '-':>6} {row['problem']}")
                print(r["message"])
            elif line == "start":
                r = exp.start()
                if not r["ok"]:
                    print(r["message"])
                    continue
                time.sleep(0.5)
                session(exp)
            elif line:
                print("명령: save <시작 위치 이름> | check | start | q")
    except (KeyboardInterrupt, EOFError):
        if exp.busy():
            print("\n중단: 진행 중인 이동이나 명령을 취소하는 중...")
            exp.abort()
            while exp.busy():
                time.sleep(0.2)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
