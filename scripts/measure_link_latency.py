"""Network latency between this machine and a remote perception/planner
server over SSH (the path the robot's requests take through the tunnel).

Starts a small echo program on the server through one SSH session (it
reads each message and answers one byte; nothing is written on the server)
and measures, one request at a time:
- small messages (64 B): round-trip time and its jitter;
- camera frames (the JPEG files of ``--frames``): send a frame, wait for the
  answer; time per frame and throughput;
- frames with an extra ``--extra-kb`` of data (e.g. a depth image).
Run on the robot's laptop on the robot's network for the numbers that
matter; the server address is given on the command line only.

    python scripts/measure_link_latency.py --ssh "-p <port> root@<server>" \\
        --frames data/perception/.../frames/images --output outputs/network/link_latency.json
"""
import argparse
import json
from pathlib import Path
import shlex
import socket
import statistics
import struct
import subprocess
import time

import numpy as np

ECHO = ("import sys,struct\n"
        "r,w=sys.stdin.buffer,sys.stdout.buffer\n"
        "while True:\n"
        " h=r.read(4)\n"
        " if len(h)<4: break\n"
        " n=struct.unpack('>I',h)[0]\n"
        " while n: n-=len(r.read(min(n,1<<20)))\n"
        " w.write(b'k'); w.flush()\n")


class Echo:
    def __init__(self, ssh_args):
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", *shlex.split(ssh_args),
                   "python3 -u -c " + shlex.quote(ECHO)]
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=0)

    def send(self, payload: bytes) -> float:
        t0 = time.perf_counter()
        self.proc.stdin.write(struct.pack(">I", len(payload)) + payload)
        self.proc.stdin.flush()
        if self.proc.stdout.read(1) != b"k":
            raise RuntimeError("서버 응답 없음 (연결 끊김)")
        return time.perf_counter() - t0

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def stats(seconds):
    ms = np.array(seconds) * 1000
    return {"n": len(ms), "median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
            "max_ms": float(ms.max()), "std_ms": float(ms.std())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ssh", required=True, help='ssh 인자 (예: "-p <port> root@<server>")')
    parser.add_argument("--frames", type=Path, required=True, help="JPEG 사진 폴더")
    parser.add_argument("--pings", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=2, help="사진 폴더를 몇 번 보낼지")
    parser.add_argument("--extra-kb", type=int, default=400, help="사진에 더할 데이터 (깊이 영상 대신)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    frames = [p.read_bytes() for p in sorted(args.frames.glob("*.jpg"))]
    if not frames:
        raise SystemExit("사진이 없습니다.")
    rng = np.random.default_rng(0)
    extra = rng.integers(0, 256, args.extra_kb * 1024, dtype=np.uint8).tobytes()   # incompressible
    echo = Echo(args.ssh)
    echo.send(b"warm-up")
    ping = []
    for _ in range(args.pings):
        ping.append(echo.send(b"x" * 64))
        time.sleep(0.1)
    frame_times = [echo.send(f) for _ in range(args.rounds) for f in frames]
    frame_extra_times = [echo.send(f + extra) for f in frames]
    echo.close()
    mean_kb = statistics.fmean(len(f) for f in frames) / 1024
    result = {
        "host": socket.gethostname(), "frames": len(frames), "mean_frame_kb": mean_kb, "extra_kb": args.extra_kb,
        "ping_64B": stats(ping),
        "frame": {**stats(frame_times), "throughput_MBps": len(frame_times) * mean_kb / 1024 / sum(frame_times)},
        "frame_plus_extra": {**stats(frame_extra_times),
                             "throughput_MBps": len(frames) * (mean_kb + args.extra_kb) / 1024 / sum(frame_extra_times)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
