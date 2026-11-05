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
from athome.perception.name_list import NameList
from athome.scene_graph.vocabulary import Vocabulary
from athome.schemas import ObservedObject
from athome.search.matching import explain
from athome_ros.debug_view import _font

REPO = Path(athome.__file__).resolve().parents[2]
CGD_CONFIDENCE, YOLOE_CONFIDENCE = 0.35, 0.15    # each detector's own floor for target labels
CLASSES = ("list", "target", "ram")
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
        vocabulary = Vocabulary.load(config.target_categories, label_aliases=config.label_aliases)
        self._canonical = vocabulary.canonical
        self._is_target = vocabulary.is_target
        self._categories = set(vocabulary.categories)
        self._cgd = p("detector", "yoloe").value == "cgd"
        # CG-D: the search's confidence of target labels (configs/robot/room406.yaml), asked one target at a time.
        matcher = dict(config.matcher, min_confidence=CGD_CONFIDENCE if self._cgd else YOLOE_CONFIDENCE)
        self._matcher = make_matcher(matcher, vocabulary.canonical, vocabulary.target_keys, vocabulary.categories)
        self.clip_ready = hasattr(self._matcher, "prepare") and self._matcher.prepare()
        base = _names(REPO / "configs/perception/vocabulary_base.txt")
        candidates = _names(REPO / "configs/perception/test_candidates.txt")
        self._candidates = candidates
        # The fixed list, changed from the page (not the search's list).
        default = _names(REPO / "configs/perception/vocabulary_search.txt") if self._cgd else base
        self._list = NameList(REPO / f"outputs/perception_check/list_{'cgd' if self._cgd else 'yoloe'}.json", default)
        # What can be a search target (the candidates may be outside the training
        # categories, as a command's target can); furniture only competes.
        self._categories |= {vocabulary.canonical(n) for n in candidates}
        self._furniture_free = set(self._categories)
        self._update_targets()
        self._query = []                # CG-D: names given as a command's targets
        self._classes = "list"
        self._font = _font(18)
        self._lock = threading.Condition()
        self._latest = None
        self._image, self._seq = None, 0
        self._raw, self._raw_seq = None, 0       # the camera as it comes, nothing drawn or waited for
        self._latest_stamp = 0.0
        self._found, self._crops = [], []        # passing detections of the last result, with their crops
        self._latency = None                     # capture to result of the last result [s]
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
            self._latest = self._raw = bytes(msg.data)
            self._latest_stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
            self._raw_seq += 1
            self._lock.notify_all()

    def _detect(self, jpeg):
        # YOLOE: asked for a name, the vocabulary of a command for it (base + name).
        with self._lock:
            names = list(self._list.names)
        vocab = names if self._cgd else list(dict.fromkeys(names + (self._query or self._candidates)))
        targets = list(dict.fromkeys(self._targets + self._query))
        body = json.dumps({"image": base64.b64encode(jpeg).decode("ascii"), "vocab": vocab,
                           "targets": targets, "query": self._query, "classes": self._classes, "masks": False,
                           "min_confidence": FLOOR}).encode()
        req = urllib.request.Request(self._url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=40.0) as r:
            return json.loads(r.read())["detections"]

    def _judge(self, d):
        """Reason as the search's found check would say."""
        if d["label"] == "other item" or (not self._cgd and self._canonical(d["label"]) not in self._categories):
            return "other"
        o = ObservedObject(object_id=-1, label=d["label"], confidence=d["confidence"],
                           centroid=(0.0, 0.0, 0.0), clip_feature=d.get("clip_feature") or [])
        # A name asked for judges the detections that count as it (trash can for can), as the search does.
        target = next((t for t in self._query if self._is_target(t, d["label"])), d["label"])
        return explain(self._matcher, target, o)

    def _update_targets(self):
        """CLIP features for what can be a search target and for added names."""
        added = {self._canonical(n) for n in self._list.names if n not in self._list.default}
        self._categories = self._furniture_free | added
        names = list(dict.fromkeys(self._list.names + self._candidates))
        self._targets = [n for n in names if self._canonical(n) in self._categories]

    def edit_list(self, action, name=""):
        """Add or remove a name of the fixed list, or return to the default
        (table from the start). The error for the page, '' if none."""
        with self._lock:
            error = ""
            if action == "add":
                error = self._list.add(name)
            elif action == "remove":
                self._list.remove(name)
            elif action == "reset":
                self._list.reset()
            else:
                return "모르는 동작"
            self._update_targets()
        self.reset()
        return error

    def set_query(self, names, classes):
        """What CG-D is asked, as a command would (table from the start)."""
        with self._lock:
            self._query = [n.strip().lower() for n in names.split(",") if n.strip()]
            self._classes = classes if classes in CLASSES else "list"
        self.reset()

    def _loop(self):
        while rclpy.ok():
            with self._lock:
                self._lock.wait_for(lambda: self._latest is not None, 1.0)
                jpeg, self._latest = self._latest, None
                stamp = self._latest_stamp
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
            with self._lock:
                self._latency = time.time() - stamp if stamp else None

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
        found, crops = [], []
        for det, reason in sorted(judged, key=lambda x: -x[0]["confidence"]):
            if reason == "ok":
                x0, y0, x1, y1 = det["box"]
                pad = 0.1 * max(x1 - x0, y1 - y0)
                crop = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad)))
                crop.thumbnail((240, 240))
                out = io.BytesIO()
                crop.save(out, format="JPEG", quality=80)
                crops.append(out.getvalue())
                found.append(dict(name=det["label"], confidence=round(det["confidence"], 2)))
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
            self._found, self._crops = found, crops
            self._lock.notify_all()

    def next_image(self, seq, timeout=2.0):
        with self._lock:
            self._lock.wait_for(lambda: self._seq != seq, timeout)
            return (self._image, self._seq) if self._seq != seq else (None, seq)

    def next_raw(self, seq, timeout=2.0):
        with self._lock:
            self._lock.wait_for(lambda: self._raw_seq != seq, timeout)
            return (self._raw, self._raw_seq) if self._raw_seq != seq else (None, seq)

    def crop(self, index):
        with self._lock:
            return self._crops[index] if 0 <= index < len(self._crops) else None

    def state(self):
        with self._lock:
            rows = []
            for name, r in self._rows.items():
                recent = list(r["recent"])
                rows.append(dict(name=name, frames=r["frames"], recent=round(sum(recent) / len(recent), 2) if recent else 0,
                                 best=round(r["best"], 2), last=round(r["last"], 2), clip=r["clip"],
                                 rank=min(r["ranks"]) if r["ranks"] else None))
            rows.sort(key=lambda x: (x["name"] not in self._query, -x["recent"], -x["frames"], -x["best"]))
            ms = sorted(self._ms)
            span = self._times[-1] - self._times[0] if len(self._times) > 1 else 0.0
            return dict(rows=rows, frames=self._frames, fps=round((len(self._times) - 1) / span, 1) if span else 0.0,
                        ms=round(ms[len(ms) // 2]) if ms else None, error=self._error, seq=self._seq,
                        clip=self.clip_ready, cgd=self._cgd, query=self._query,
                        classes=self._classes, found=self._found, list=list(self._list.names), list_default=self._list.is_default,
                        latency=round(self._latency, 1) if self._latency is not None else None)


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>인식 확인</title>
<style>
:root{color-scheme:dark;--bg:#111417;--panel:#1a1f24;--line:#2c343b;--text:#e8ebed;--muted:#94a0a9;--accent:#3f8cff;--ok:#2fb457;--clip:#e62828;--low:#ff8c00;--other:#8a939a}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,"Noto Sans CJK KR","NanumGothic",sans-serif;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:20px;padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap}
h1{margin:0;font-size:16px;font-weight:700;white-space:nowrap}.badge{display:inline-block;margin-left:6px;padding:1px 8px;border-radius:10px;background:var(--accent);color:#fff;font-size:12px;font-weight:600;vertical-align:1px}
form{margin:0}#ask{display:flex;align-items:center;gap:8px;flex:1;min-width:320px;max-width:620px}#ask label{color:var(--muted);white-space:nowrap}
input,select,button{font:inherit;color:var(--text);border-radius:6px;border:1px solid var(--line);background:var(--bg);padding:6px 10px}input{flex:1;min-width:0}input:focus,select:focus,button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
button{background:#2a323a;font-weight:600;cursor:pointer;white-space:nowrap}button:hover{background:#343e47}button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button.link{background:none;border:0;color:var(--muted);font-weight:400;padding:2px 4px}button.link:hover{color:var(--text)}
.stats{display:flex;gap:18px;margin-left:auto}.stat{text-align:right;white-space:nowrap}.stat b{display:block;font-size:15px;font-variant-numeric:tabular-nums}.stat span{color:var(--muted);font-size:12px}
#err{display:none;padding:7px 16px;background:#4a1d1f;color:#ffc4c4;overflow-wrap:anywhere}
main{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:12px;padding:12px}
.left{min-height:0;overflow:auto;display:flex;flex-direction:column;gap:12px}.views{flex:none;display:grid;grid-template-columns:1fr 1fr;gap:12px}
.view{position:relative;aspect-ratio:16/9;background:#000;border:1px solid var(--line);border-radius:8px;overflow:hidden}.view img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
.cap{position:absolute;z-index:1;top:8px;left:8px;background:#000b;padding:2px 8px;border-radius:4px;font-size:12px}
.legend{color:var(--muted);font-size:12px;display:flex;gap:14px;flex-wrap:wrap}.k{display:inline-block;width:10px;height:10px;margin-right:4px;border-radius:2px;vertical-align:-1px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}h2{margin:0;font-size:14px;font-weight:700}.head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}.note{color:var(--muted);font-size:12px}
#found{display:flex;gap:10px;overflow-x:auto;min-height:144px;align-items:stretch}.card{flex:none;width:130px;background:var(--bg);border:1px solid var(--line);border-top:3px solid var(--ok);border-radius:6px;padding:6px;text-align:center}.card img{width:100%;height:84px;object-fit:contain;display:block;margin-bottom:4px}.card b{display:block;overflow-wrap:anywhere;line-height:1.25}.card span{color:var(--muted);font-size:12px}
.empty{color:var(--muted);align-self:center;width:100%;text-align:center}
aside{min-height:0;display:flex;flex-direction:column;padding:0;overflow:hidden}.tabs{display:flex;border-bottom:1px solid var(--line)}.tabs button{flex:1;background:none;border:0;border-bottom:2px solid transparent;border-radius:0;padding:10px;color:var(--muted)}.tabs button[aria-selected=true]{color:var(--text);border-bottom-color:var(--accent)}
.tab{flex:1;min-height:0;overflow:auto;padding:12px}
table{width:100%;border-collapse:collapse;table-layout:fixed}th{color:var(--muted);font-weight:500;font-size:12px;text-align:right;padding:4px}th:first-child,td:first-child{text-align:left;width:46%}th:nth-child(2),td:nth-child(2){width:16%}th:nth-child(3),td:nth-child(3){width:15%}td{padding:6px 4px;border-top:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}.nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}tr.target td:first-child{font-weight:700;color:#9cc4ff}.bar{height:4px;margin-top:3px;background:var(--line);border-radius:2px}.bar i{display:block;height:100%;background:var(--ok);border-radius:2px}
details{margin-top:12px}summary{cursor:pointer;color:var(--muted);font-size:13px}
#add{display:flex;gap:6px;margin-bottom:6px}#listerr{color:#ff9a9d;font-size:12px;min-height:16px}
#names{display:flex;flex-wrap:wrap;gap:4px;margin:6px 0 10px}#names button{font-weight:400;font-size:13px;padding:2px 8px;border-radius:12px}#names button::after{content:" ×";color:var(--muted)}
@media (max-width:1100px){html,body{height:auto}main{grid-template-columns:1fr}.views{grid-template-columns:1fr;flex:none}.stats{margin-left:0}.tab{max-height:480px}}
</style></head><body>
<header><h1>인식 확인<span class="badge" id="name">-</span></h1>
<form id="ask"><label for="q">찾을 이름</label><input id="q" placeholder="예: mouse" autocomplete="off"><select id="c" aria-label="DINO 이름 목록 방식"><option value="list">고정 목록 + 목표</option><option value="target">목표만</option><option value="ram">RAM 포함</option></select><button class="primary">적용</button></form>
<div class="stats"><div class="stat"><b id="now">없음</b><span>찾는 중</span></div><div class="stat"><b id="floor">-</b><span>신뢰도 기준</span></div><div class="stat"><b id="ms">-</b><span>사진 한 장</span></div><div class="stat"><b id="lat">-</b><span>촬영→결과</span></div></div></header>
<div id="err" role="alert"></div>
<main><div class="left"><div class="views">
<div class="view"><span class="cap">실시간 카메라</span><img id="raw" src="/camera.mjpg" alt="카메라 영상"></div>
<div class="view"><span class="cap">검출 결과</span><img id="cam" src="/stream.mjpg" alt="검출 결과가 그려진 카메라 영상"></div></div>
<div class="legend"><span><i class="k" style="background:var(--ok)"></i>통과</span><span><i class="k" style="background:var(--clip)"></i>CLIP 탈락</span><span><i class="k" style="background:var(--low)"></i>신뢰도 미달</span><span><i class="k" style="background:var(--other)"></i>판정 대상 아님</span></div>
<section class="panel"><div class="head"><h2>인식된 물체</h2><span class="note" id="foundn"></span></div><div id="found"></div></section></div>
<aside class="panel"><div class="tabs" role="tablist"><button role="tab" id="t1" aria-selected="true">결과 표</button><button role="tab" id="t2" aria-selected="false">이름 목록 <span id="count"></span></button></div>
<div class="tab" id="p1"><div class="head"><span class="note">최근 30프레임 통과 비율 · 총 <span id="frames">0</span>프레임</span><button class="link" id="reset">처음부터</button></div>
<table><thead><tr><th>이름</th><th>통과</th><th>최고</th><th title="CLIP 순위에서 탈락한 횟수와 가장 좋았던 순위">CLIP</th></tr></thead><tbody id="rows"></tbody></table>
<details id="more"><summary>나머지 <span id="othern">0</span>개</summary><table><tbody id="others"></tbody></table></details></div>
<div class="tab" id="p2" hidden><form id="add"><input id="new" placeholder="추가할 이름 (영어)" aria-label="추가할 이름" autocomplete="off"><button>추가</button></form><div id="listerr"></div><div id="names"></div>
<div class="head"><span class="note" id="listnote"></span><button class="link" id="defaults">기본값으로</button></div><p class="note" style="margin:6px 0 0">시험용 목록입니다. 다시 켜도 유지되고, 로봇 탐색에는 적용되지 않습니다.</p></div></aside></main>
<script>const $=i=>document.getElementById(i);const esc=s=>{const d=document.createElement("div");d.textContent=s;return d.innerHTML};const J={"Content-Type":"application/json"};
const tab=n=>{$("t1").setAttribute("aria-selected",n==1);$("t2").setAttribute("aria-selected",n==2);$("p1").hidden=n!=1;$("p2").hidden=n!=2};$("t1").onclick=()=>tab(1);$("t2").onclick=()=>tab(2);
$("reset").onclick=()=>fetch("/reset",{method:"POST",headers:J,body:"{}"});
$("ask").onsubmit=e=>{e.preventDefault();fetch("/query",{method:"POST",headers:J,body:JSON.stringify({names:$("q").value,classes:$("c").value})})};
const edit=async b=>{const r=await(await fetch("/list",{method:"POST",headers:J,body:JSON.stringify(b)})).json();$("listerr").textContent=r.error||""};
$("add").onsubmit=e=>{e.preventDefault();edit({action:"add",name:$("new").value});$("new").value=""};
$("defaults").onclick=()=>{if(confirm("이름 목록을 기본값으로 되돌릴까요?"))edit({action:"reset"})};
$("names").onclick=e=>{const n=e.target.closest("button");if(n)edit({action:"remove",name:n.dataset.name})};let listed="",shown=-1;
$("raw").onerror=()=>setTimeout(()=>{$("raw").src="/camera.mjpg?"+Date.now()},1000);$("cam").onerror=()=>setTimeout(()=>{$("cam").src="/stream.mjpg?"+Date.now()},1000);
const row=(r,target)=>`<tr class="${target?"target":""}"><td><div class="nm" title="${esc(r.name).replace(/"/g,"&quot;")}">${esc(r.name)}</div><div class="bar"><i style="width:${Math.round(100*r.recent)}%"></i></div></td><td>${Math.round(100*r.recent)}%</td><td>${r.best.toFixed(2)}</td><td>${r.clip?`<span title="${r.clip}회 탈락, 최고 ${r.rank}위">${r.clip}회·${r.rank}위</span>`:"-"}</td></tr>`;
async function poll(){try{const s=await(await fetch("/state")).json();
$("c").hidden=!s.cgd;const det=s.cgd?"CG-D":"YOLOE";document.title=det+" 인식 확인";$("name").textContent=det;
$("now").textContent=s.query.join(", ")||"없음";$("floor").textContent=s.cgd?"0.35":"0.15";$("ms").textContent=s.ms!=null?(s.ms/1000).toFixed(1)+"초":"-";$("lat").textContent=s.latency!=null?s.latency+"초":"-";$("frames").textContent=s.frames;
const msg=s.error||(s.seq?"":"카메라 영상 대기 중")||(s.clip?"":"CLIP 확인이 꺼져 있습니다");$("err").textContent=msg;$("err").style.display=msg?"block":"none";
const key=JSON.stringify(s.list);if(key!==listed){listed=key;$("count").textContent=s.list.length;$("listnote").textContent=s.list_default?"기본값":"수정됨";
$("names").innerHTML=s.list.map(n=>`<button type="button" title="${esc(n).replace(/"/g,"&quot;")} 삭제" data-name="${esc(n).replace(/"/g,"&quot;")}">${esc(n)}</button>`).join("")}
if(s.seq!==shown){shown=s.seq;$("foundn").textContent=s.found.length?s.found.length+"개":"";$("found").innerHTML=s.found.length?s.found.map((f,i)=>`<div class="card"><img src="/crop/${i}.jpg?${s.seq}" alt="${esc(f.name)}"><b>${esc(f.name)}</b><span>${f.confidence.toFixed(2)}</span></div>`).join(""):'<div class="empty">통과한 물체 없음</div>'}
const tgt=r=>s.query.includes(r.name),main=s.rows.filter(r=>tgt(r)||r.recent>0),rest=s.rows.filter(r=>!(tgt(r)||r.recent>0));
$("rows").innerHTML=main.map(r=>row(r,tgt(r))).join("")||'<tr><td colspan="4" class="note">아직 통과한 이름이 없습니다</td></tr>';$("others").innerHTML=rest.map(r=>row(r,false)).join("");$("othern").textContent=rest.length;$("more").style.display=rest.length?"":"none";
}catch(_){$("err").textContent="페이지 서버 연결이 끊겼습니다";$("err").style.display="block"}}poll();setInterval(poll,500);</script></body></html>"""


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
            elif path.startswith("/crop/"):
                image = node.crop(int(path[6:].split(".")[0])) if path[6:].split(".")[0].isdigit() else None
                if image is None:
                    self._send(b"not found", "text/plain", 404)
                else:
                    self._send(image, "image/jpeg")
            elif path in ("/stream.mjpg", "/camera.mjpg"):
                source = node.next_image if path == "/stream.mjpg" else node.next_raw
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                seq = -1
                try:
                    while rclpy.ok():
                        image, seq = source(seq)
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
            body = self.rfile.read(size)
            if (self.path not in ("/reset", "/query", "/list")
                    or self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json"):
                self._send(b"not found", "text/plain", 404)
                return
            if self.path == "/list":
                request = json.loads(body or b"{}")
                error = node.edit_list(str(request.get("action", "")), str(request.get("name", "")))
                self._send(json.dumps({"error": error}, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
                return
            if self.path == "/query":
                request = json.loads(body or b"{}")
                node.set_query(str(request.get("names", "")), str(request.get("classes", "list")))
            else:
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
