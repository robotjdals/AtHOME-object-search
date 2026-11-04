"""Does the real-time detector see this object? Camera and detection server
only, no robot, SLAM or map: the page shows the camera with each detection's
name and confidence as the search would judge it (confidence, CLIP rank), and
a table per name. Boxes are the detector's 2D boxes, so there are no positions;
what the 3D lifting does is not part of this check.

    ros2 launch athome_bringup perception_check.launch.py      # http://localhost:8090
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
import urllib.request
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import athome
import rclpy
from PIL import Image, ImageDraw
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

from athome.config import load_robot_config
from athome.inference.factory import make_matcher
from athome.scene_graph.vocabulary import Vocabulary
from athome.schemas import ObservedObject
from athome.search.matching import explain
from athome_ros.debug_view import _font

REPO = Path(athome.__file__).resolve().parents[2]
FLOOR = 0.1                     # drawn from here; the search's own floor comes from its config
RECENT_FRAMES = 30
GREEN, RED, ORANGE, GREY = (0, 220, 0), (230, 40, 40), (255, 140, 0), (170, 170, 170)


def _names(path: Path):
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


class PerceptionCheck(Node):
    def __init__(self):
        super().__init__("athome_perception_check")
        p = self.declare_parameter
        config = load_robot_config(p("robot_config", str(REPO / "configs/robot/room406.yaml")).value)
        self.port = p("port", 8090).value
        self._url = p("server", "http://127.0.0.1:18200").value.rstrip("/") + "/detect"
        vocabulary = Vocabulary.load(config.target_categories)
        self._canonical = vocabulary.canonical
        self._categories = set(vocabulary.categories)
        self._matcher = make_matcher(config.matcher, vocabulary.canonical, vocabulary.target_keys,
                                     vocabulary.categories)
        self.clip_ready = hasattr(self._matcher, "prepare") and self._matcher.prepare()
        base = _names(REPO / "configs/perception/vocabulary_base.txt")
        candidates = _names(REPO / "configs/perception/test_candidates.txt")
        self._vocab = list(dict.fromkeys(base + candidates))
        # What can be a search target (the candidates may be outside the training
        # categories, as a command's target can); furniture only competes.
        self._categories |= {vocabulary.canonical(n) for n in candidates}
        self._targets = [n for n in self._vocab if vocabulary.canonical(n) in self._categories]
        # CG-D makes its own names (RAM tags + household list): it is given the candidates.
        self._query = candidates if p("detector", "yoloe").value == "cgd" else []
        self._font = _font(18)
        self._lock = threading.Condition()
        self._latest = None
        self._image, self._seq = None, 0
        self._error = None
        self.reset()
        self.create_subscription(CompressedImage, p("camera_topic", "/camera/camera/color/image_raw/compressed").value,
                                 self._on_camera, qos_profile_sensor_data)
        threading.Thread(target=self._loop, daemon=True).start()

    def reset(self):
        with self._lock:
            self._frames = 0
            self._times = deque(maxlen=30)
            self._rows = defaultdict(lambda: dict(frames=0, best=0.0, last=0.0, clip=0, ranks=[],
                                                  recent=deque(maxlen=RECENT_FRAMES)))
            self._ms = deque(maxlen=30)

    def _on_camera(self, msg):
        with self._lock:
            self._latest = bytes(msg.data)
            self._lock.notify_all()

    def _detect(self, jpeg):
        body = json.dumps({"image": base64.b64encode(jpeg).decode("ascii"), "vocab": self._vocab,
                           "targets": self._targets, "query": self._query, "min_confidence": FLOOR}).encode()
        req = urllib.request.Request(self._url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=40.0) as r:
            return json.loads(r.read())["detections"]

    def _judge(self, d):
        """Reason under its own name, as the search's found check would say."""
        if self._canonical(d["label"]) not in self._categories:
            return "other"
        o = ObservedObject(object_id=-1, label=d["label"], confidence=d["confidence"],
                           centroid=(0.0, 0.0, 0.0), clip_feature=d.get("clip_feature") or [])
        return explain(self._matcher, d["label"], o)

    def _loop(self):
        while rclpy.ok():
            with self._lock:
                self._lock.wait_for(lambda: self._latest is not None, 1.0)
                jpeg, self._latest = self._latest, None
            if jpeg is None:
                continue
            t = time.monotonic()
            try:
                dets = self._detect(jpeg)
                self._error = None
            except Exception as e:  # noqa: BLE001 - shown on the page, tried again
                self._error = f"검출 서버: {e}"
                self._publish_view(jpeg, [])
                time.sleep(1.0)
                continue
            judged = [(d, self._judge(d)) for d in dets]
            self._count(judged, time.monotonic() - t)
            self._publish_view(jpeg, judged)

    def _count(self, judged, seconds):
        with self._lock:
            self._frames += 1
            self._ms.append(seconds * 1e3)
            self._times.append(time.monotonic())
            seen = defaultdict(lambda: None)
            for d, reason in judged:
                if reason == "other":
                    continue
                name = self._canonical(d["label"])
                row = self._rows[name]
                row["best"] = max(row["best"], d["confidence"])
                if reason == "ok":
                    seen[name] = max(seen[name] or 0.0, d["confidence"])
                elif reason.startswith("clip:"):
                    row["clip"] += 1
                    row["ranks"].append(int(reason[5:]))
            for name, row in self._rows.items():
                passed = seen.get(name) is not None
                row["recent"].append(passed)
                row["frames"] += passed
                if passed:
                    row["last"] = seen[name]

    def _publish_view(self, jpeg, judged):
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        d = ImageDraw.Draw(img)
        style = lambda r: (GREEN, 4) if r == "ok" else (RED, 3) if r.startswith("clip") else (
            ORANGE, 2) if r in ("confidence", "outside") else (GREY, 1)
        for det, reason in sorted(judged, key=lambda x: x[1] == "ok"):
            colour, width = style(reason)
            box = det["box"]
            d.rectangle(box, outline=colour, width=width)
            text = f"{det['label']} {det['confidence']:.2f}" + (f" CLIP {reason[5:]}위" if reason.startswith("clip") else "")
            y = max(box[1] - 22, 0)
            d.rectangle((box[0], y, box[0] + d.textlength(text, font=self._font) + 6, y + 22), fill=colour)
            d.text((box[0] + 3, y + 1), text, fill=(0, 0, 0), font=self._font)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=80)
        with self._lock:
            self._image, self._seq = out.getvalue(), self._seq + 1
            self._lock.notify_all()

    def next_image(self, seq, timeout=2.0):
        with self._lock:
            self._lock.wait_for(lambda: self._seq != seq, timeout)
            return (self._image, self._seq) if self._seq != seq else (None, seq)

    def state(self):
        with self._lock:
            rows = []
            for name, r in self._rows.items():
                recent = list(r["recent"])
                rows.append(dict(name=name, frames=r["frames"], recent=round(sum(recent) / len(recent), 2) if recent else 0,
                                 best=round(r["best"], 2), last=round(r["last"], 2), clip=r["clip"],
                                 rank=min(r["ranks"]) if r["ranks"] else None))
            rows.sort(key=lambda x: (-x["recent"], -x["frames"], -x["best"]))
            ms = sorted(self._ms)
            span = self._times[-1] - self._times[0] if len(self._times) > 1 else 0.0
            return dict(rows=rows, frames=self._frames, fps=round((len(self._times) - 1) / span, 1) if span else 0.0,
                        ms=round(ms[len(ms) // 2]) if ms else None, error=self._error, seq=self._seq,
                        clip=self.clip_ready, names=len(self._vocab))


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>인식 확인</title>
<style>:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#14171a;color:#e8ebed;font:15px/1.45 system-ui,"Noto Sans CJK KR","NanumGothic",sans-serif}
main{display:grid;grid-template-columns:minmax(0,1fr) 400px;gap:12px;padding:12px;height:100vh}@media (max-width:1000px){main{grid-template-columns:1fr;height:auto}}
.view{background:#000;border:1px solid #333b42;border-radius:8px;display:flex;align-items:center;justify-content:center;overflow:hidden;min-height:300px}.view img{width:100%;height:100%;object-fit:contain}
aside{background:#1e2327;border:1px solid #333b42;border-radius:8px;padding:12px;overflow:auto}h2{margin:0 0 6px;font-size:16px}#meta,.note{color:#9aa5ad;font-size:13px}#err{color:#ff9a9d;font-size:13px;overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;font-size:14px;margin-top:10px}th{color:#9aa5ad;font-weight:600;text-align:left;font-size:12px}th+th,td+td{text-align:right}td,th{padding:4px;border-top:1px solid #333b42}
.bar{height:6px;background:#2fa84f;border-radius:3px;margin-top:2px}button{font:inherit;font-weight:600;border:0;border-radius:6px;padding:8px 12px;background:#39424a;color:#fff;cursor:pointer;margin-top:8px;width:100%}
.k{display:inline-block;width:10px;height:10px;margin:0 3px 0 8px;border-radius:2px}</style></head><body><main>
<div class="view"><img id="cam" src="/stream.mjpg" alt="카메라와 검출 결과"></div>
<aside><h2>인식 확인</h2><div id="meta">연결 중</div><div id="err"></div>
<div class="note" style="margin-top:6px"><span class="k" style="background:#00dc00;margin-left:0"></span>통과<span class="k" style="background:#e62828"></span>CLIP 탈락<span class="k" style="background:#ff8c00"></span>신뢰도 미달<span class="k" style="background:#aaa"></span>가구 등</div>
<button id="reset">표 처음부터</button>
<table><thead><tr><th>물체</th><th>최근 30프레임</th><th>최고</th><th>CLIP 탈락</th></tr></thead><tbody id="rows"></tbody></table>
<div class="note" style="margin-top:8px">물체를 카메라 앞에 두고 거리와 각도를 바꿔 보세요. 최근 30프레임은 그 이름이 통과한 비율입니다.</div></aside></main>
<script>const $=i=>document.getElementById(i);const esc=s=>{const d=document.createElement("div");d.textContent=s;return d.innerHTML};
$("reset").onclick=()=>fetch("/reset",{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});
$("cam").onerror=()=>setTimeout(()=>{$("cam").src="/stream.mjpg?"+Date.now()},1000);
async function poll(){try{const s=await(await fetch("/state")).json();
$("meta").textContent=`${s.frames}프레임, ${s.fps} Hz, 장당 ${s.ms??"-"} ms, 묻는 이름 ${s.names}개`+(s.clip?"":", CLIP 확인 꺼짐");
$("err").textContent=s.error||(s.seq?"":"카메라 영상 대기 중");
$("rows").innerHTML=s.rows.map(r=>`<tr><td>${esc(r.name)}<div class="bar" style="width:${Math.round(100*r.recent)}%"></div></td><td>${Math.round(100*r.recent)}%</td><td>${r.best}</td><td>${r.clip?r.clip+" (최고 "+r.rank+"위)":"-"}</td></tr>`).join("")||'<tr><td colspan="4">아직 없음</td></tr>';
}catch(_){$("err").textContent="페이지 서버 연결 끊김"}}poll();setInterval(poll,500);</script></body></html>"""


def make_handler(node):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, body, kind, code=200):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/state":
                self._send(json.dumps(node.state(), ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            elif path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                seq = -1
                try:
                    while rclpy.ok():
                        image, seq = node.next_image(seq)
                        if image is not None:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(image))
                            self.wfile.write(image + b"\r\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            size = int(self.headers.get("Content-Length", 0))
            self.rfile.read(size)
            if self.path != "/reset" or self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                self._send(b"not found", "text/plain", 404)
                return
            node.reset()
            self._send(b"{}", "application/json")

    return Handler


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionCheck()
    server = ThreadingHTTPServer(("127.0.0.1", node.port), make_handler(node))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    line = "=" * 64
    node.get_logger().info(f"\n{line}\n인식 확인 화면: http://localhost:{node.port}\n{line}")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
