"""Run the scene pipeline over many HM3DSem scenes (e.g. the train split).

Every scene folder with a semantic annotation (<id>.semantic.txt) is run
through scripts/run_scene_pipeline.py in its own process. A failing scene
does not stop the others; its stage and error tail are recorded. Finished
stages are skipped by the scene driver, so the command can be re-run after
an interruption. The status file is rewritten after every scene. Scenes are
independent, so ``--jobs N`` runs N scene processes at once (each mostly
waits for its OpenAI batch).

    python scripts/run_dataset.py --split-dir data/scene_datasets/hm3d/train --version v5 \
        --submit --teacher openai --teacher-starts 2 [--limit 10]

API cost comes from the scene-graph batches and the Teacher (--submit with
--teacher openai); check the estimate before large runs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]


def annotated_scenes(split_dir: Path):
    for scene in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        scene_id = scene.name.split("-", 1)[-1]
        if (scene / f"{scene_id}.semantic.txt").is_file():
            yield scene


def last_stage(log: str) -> str:
    stages = [line[4:-4] for line in log.splitlines() if line.startswith("=== ") and line.endswith(" ===")]
    return stages[-1] if stages else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, required=True, help="HM3D split 폴더 (예: .../hm3d/train)")
    parser.add_argument("--version", required=True)
    parser.add_argument("--limit", type=int, help="앞에서부터 이 개수만 (비용 확인용 1차 실행)")
    parser.add_argument("--status", type=Path, help="기본: outputs/hm3d_<version>/dataset_status.<split>.json")
    parser.add_argument("--jobs", type=int, default=1, help="동시에 돌릴 장면 수")
    args, passthrough = parser.parse_known_args()
    split = args.split_dir.name
    status_path = args.status or ROOT / f"outputs/hm3d_{args.version}/dataset_status.{split}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    logs = status_path.parent / "logs"
    logs.mkdir(exist_ok=True)
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}

    scenes = list(annotated_scenes(ROOT / args.split_dir))
    if args.limit:
        scenes = scenes[:args.limit]
    print(f"주석 장면 {len(scenes)}개 ({args.split_dir})", flush=True)
    lock = threading.Lock()
    done = [0]

    def run_scene(scene):
        log_path = logs / f"{scene.name}.log"
        cmd = [sys.executable, "scripts/run_scene_pipeline.py", "--scene-dir", str(scene),
               "--version", args.version, *passthrough]
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n##### {datetime.now().isoformat(timespec='seconds')} {' '.join(cmd)}\n")
            log.flush()
            code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        text = log_path.read_text(encoding="utf-8").split("\n##### ")[-1]
        stopped = "[중단]" in text
        entry = {
            "result": "failed" if code else ("gated" if stopped else "done"),
            "exit_code": code, "last_stage": "complete" if "모든 단계 완료" in text else last_stage(text),
            "tail": "\n".join(text.strip().splitlines()[-5:]) if code or stopped else "",
            "log": str(log_path), "updated": datetime.now().isoformat(timespec="seconds"),
        }
        with lock:
            status[scene.name] = entry
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            done[0] += 1
            print(f"[{done[0]}/{len(scenes)}] {scene.name}: {entry['result']} ({entry['last_stage']})", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        list(pool.map(run_scene, scenes))
    counts = {}
    for s in status.values():
        counts[s["result"]] = counts.get(s["result"], 0) + 1
    print("요약:", counts, "→", status_path)


if __name__ == "__main__":
    main()
