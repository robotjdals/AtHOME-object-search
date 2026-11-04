"""The operator console's page (athome_ros.operator_console): one file, no
external resources, so it works without internet on the robot's network."""

PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AtHOME 조작 콘솔</title>
<style>
:root{color-scheme:dark;--bg:#14171a;--panel:#1e2327;--line:#333b42;--text:#e8ebed;--dim:#9aa5ad;
--accent:#3d8bfd;--ok:#2fa84f;--warn:#e0a100;--bad:#e5484d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,"Noto Sans CJK KR","NanumGothic",sans-serif}
main{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:12px;padding:12px;height:100vh}
@media (max-width:1000px){main{grid-template-columns:1fr;height:auto}}
.view{background:#000;border:1px solid var(--line);border-radius:8px;overflow:hidden;display:flex;align-items:center;justify-content:center;min-height:300px}
.view img{width:100%;height:100%;object-fit:contain}
.view .none{color:var(--dim);padding:24px;text-align:center}
aside{display:flex;flex-direction:column;gap:10px;min-height:0;overflow:auto}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
h2{margin:0 0 8px;font-size:13px;font-weight:600;color:var(--dim)}
#state{font-size:19px;font-weight:600;line-height:1.35;overflow-wrap:anywhere}
#stale{color:var(--bad);font-size:13px;margin-top:4px}
.meta{color:var(--dim);font-size:13px;margin-top:6px;overflow-wrap:anywhere}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.chip{border:1px solid var(--line);border-radius:999px;padding:2px 10px;font-size:13px}
.chip.found{border-color:var(--ok);color:#7ee09a}.chip.failed{border-color:var(--bad);color:#ff9a9d}
input,select,textarea{width:100%;background:#12151a;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:8px;font:inherit}
button{font:inherit;font-weight:600;border:0;border-radius:6px;padding:9px 12px;background:var(--accent);color:#fff;cursor:pointer}
button:disabled{opacity:.45;cursor:default}
button.alt{background:#39424a}
.row{display:flex;gap:8px;margin-top:8px}.row>*{flex:1}
#stop{width:100%;padding:16px;font-size:20px;background:var(--bad)}
.note{color:var(--dim);font-size:12px;margin-top:6px}
#msg{min-height:1.4em;font-size:13px;margin-top:6px;color:var(--warn);overflow-wrap:anywhere}
.good{background:var(--ok)}.warn{background:#a97900}.bad{background:var(--bad)}
ul{list-style:none;margin:0;padding:0;font-size:13px}
li{padding:5px 0;border-top:1px solid var(--line);overflow-wrap:anywhere}li:first-child{border-top:0}
li span{color:var(--dim)}
.now{font-size:17px;font-weight:600;margin-top:10px;padding:8px;border-radius:6px;background:#2a3036;overflow-wrap:anywhere}
.now.ok{background:#1d4d2c}.now.weak{background:#5c4700}.now.none{background:#2a3036}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
th{font-weight:600;color:var(--dim);text-align:left}th+th,td+td{text-align:right}
td,th{padding:2px 4px;border-top:1px solid var(--line)}
#camsrc{position:absolute;top:8px;left:8px;background:#000c;color:var(--warn);font-size:12px;padding:2px 8px;border-radius:4px}
.view{position:relative}
</style></head><body><main>
<div class="view"><div id="camsrc" hidden>원본 카메라 (판정 영상 없음: 인지·탐색 서버 확인)</div><img id="cam" alt="판정 카메라 영상" hidden><div id="nocam" class="none">카메라 영상 대기 중<br>(탐색 서버와 인지 노드 실행 확인)</div></div>
<aside>
<section aria-live="polite"><h2>현재 상태</h2><div id="state">연결 중</div><div id="stale" hidden></div>
<div id="targets" class="chips"></div><div id="meta" class="meta"></div></section>
<section><button id="stop">정지</button>
<div class="note">진행 중인 명령을 취소하고 로봇 정지를 확인합니다. 비상 정지가 아니니 조이스틱도 준비해 두세요.</div>
<div class="row"><button id="resume" class="alt" hidden>일시정지된 명령 재개</button></div></section>
<section><h2>명령</h2>
<form id="cmdform"><input id="cmd" placeholder="예: 컵 찾아줘" autocomplete="off" aria-label="자연어 명령">
<div class="row"><button id="parse" type="submit">해석</button></div></form>
<div id="confirm" hidden><h2 style="margin-top:12px">찾을 물체 (쉼표로 구분, 고칠 수 있음)</h2>
<input id="targetsEdit" aria-label="찾을 물체"><div id="vocab" class="note"></div>
<div class="row"><button id="go">이대로 찾기</button><button id="drop" class="alt">취소</button></div></div>
<div id="msg" role="status"></div></section>
<section><h2>인식 시험 (명령 없이 계속 판정)</h2>
<div class="row" style="margin-top:0"><button type="button" id="watchAll">전체 물체 시작 / 처음부터</button><button type="button" id="watchStop" class="alt">끝내기</button></div>
<div id="scene" hidden><div id="sceneMeta" class="meta"></div><ul id="sceneRows" style="margin-top:6px"></ul></div>
<form id="watchform" style="margin-top:8px"><input id="watchTarget" placeholder="물체 하나만 자세히 (영어, 예: cup)" autocomplete="off" aria-label="시험할 물체"></form>
<div id="watch" hidden>
<div id="watchNow" class="now"></div><div id="watchMeta" class="meta"></div>
<table><thead><tr><th>거리</th><th>본 프레임</th><th>통과</th></tr></thead><tbody id="watchDist"></tbody></table>
<table><thead><tr><th>각도</th><th>본 프레임</th><th>통과</th></tr></thead><tbody id="watchAngle"></tbody></table>
</div>
<div class="note">물체들을 놓고 조이스틱으로 거리와 각도를 바꿔 보세요. 잡힌 물체마다 줄이 생기고, 잡힌 자리를 기준으로 시야 안인데 놓친 프레임까지 셉니다. 명령을 넣으면 그동안 쉽니다.</div></section>
<section><h2>수동 체크</h2>
<input id="checkTarget" placeholder="물체 이름" aria-label="체크할 물체">
<input id="checkNote" placeholder="메모 (선택: 예: 주방 책상 왼쪽에 있었음)" aria-label="메모" style="margin-top:8px">
<div class="row"><button class="good" data-v="correct">맞음</button><button class="warn" data-v="missed">미검출</button><button class="bad" data-v="false_positive">오검출</button></div>
<div class="note">누른 순간의 상태와 카메라 화면이 함께 저장됩니다.</div></section>
<section><h2>최근 체크</h2><ul id="checks"></ul></section>
<section id="sharebox" hidden><h2>다른 컴퓨터에서 열기</h2><div id="share" class="meta" style="margin-top:0;user-select:all"></div></section>
</aside></main>
<script>
const $=id=>document.getElementById(id);
async function post(path,body={}){
  const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  return r.json();
}
function say(t){$("msg").textContent=t||"";}
function esc(s){const d=document.createElement("div");d.textContent=s==null?"":String(s);return d.innerHTML;}
$("cmdform").onsubmit=async e=>{
  e.preventDefault();$("parse").disabled=true;say("해석 중…");
  try{
    const r=await post("/parse",{instruction:$("cmd").value});
    if(!r.ok){say(r.message);$("confirm").hidden=true;return;}
    say("");$("targetsEdit").value=r.targets.map(t=>t.name).join(", ");
    const out=r.targets.filter(t=>!t.known).map(t=>t.name);
    $("vocab").textContent=out.length?`학습 어휘 밖: ${out.join(", ")} (잘 못 찾을 수 있음)`:"";
    $("confirm").hidden=false;$("go").focus();
  }catch(_){say("콘솔 서버 연결 실패");}finally{$("parse").disabled=false;}
};
$("go").onclick=async()=>{
  const targets=$("targetsEdit").value.split(",").map(s=>s.trim()).filter(Boolean);
  $("go").disabled=true;
  try{
    const r=await post("/start",{targets});
    say(r.ok?"":r.message);
    if(r.ok){$("confirm").hidden=true;$("cmd").value="";if(targets.length===1)$("checkTarget").value=targets[0];}
  }catch(_){say("콘솔 서버 연결 실패");}finally{$("go").disabled=false;}
};
$("drop").onclick=()=>{$("confirm").hidden=true;say("");};
$("stop").onclick=async()=>{
  say("취소 요청 중…");
  try{say((await post("/stop")).message);}catch(_){say("콘솔 서버 연결 실패: 조이스틱으로 정지하세요");}
};
$("resume").onclick=async()=>{const r=await post("/resume");say(r.ok?"":r.message);};
document.querySelectorAll("[data-v]").forEach(b=>b.onclick=async()=>{
  const target=$("checkTarget").value.trim();
  if(!target){say("체크할 물체 이름을 적으세요");$("checkTarget").focus();return;}
  const r=await post("/check",{target,verdict:b.dataset.v,note:$("checkNote").value});
  say(r.ok?`체크 저장: ${target} ${b.textContent}`:r.message);
  if(r.ok)$("checkNote").value="";
});
$("watchform").onsubmit=async e=>{
  e.preventDefault();const target=$("watchTarget").value.trim();
  if(!target){say("시험할 물체 이름을 적으세요");return;}
  await post("/watch",{target});$("checkTarget").value=target;
};
$("watchStop").onclick=()=>post("/watch",{target:""});
$("watchAll").onclick=()=>post("/watch",{all:true});
const WHY={ok:"통과",confidence:"이름은 맞지만 신뢰도 미달",outside:"탐색 구역 밖",none:"이 이름의 검출 없음"};
function rows(list){return list.filter(r=>r.frames).map(r=>`<tr><td>${esc(r.bin)}</td><td>${r.frames}</td><td>${r.passed} (${Math.round(100*r.passed/r.frames)}%)</td></tr>`).join("")||'<tr><td colspan="3">아직 없음</td></tr>';}
function showScene(w){
  const all=!!(w&&w.mode==="all");$("scene").hidden=!all;if(!all)return;
  $("sceneMeta").textContent=(w.paused?"명령 실행 중: 시험 쉼  |  ":"")+`${w.frames}프레임, 묻는 이름 ${w.names}개, 물체 ${w.objects.length}개`+(w.flickers?`, 한 번만 잡힌 것 ${w.flickers}개`:"");
  $("sceneRows").innerHTML=w.objects.map(o=>{
    const pct=o.rate==null?"-":Math.round(100*o.rate)+"%";
    const bins=o.by_distance.map(r=>`${esc(r.bin)} ${r.passed}/${r.frames}`).join(", ");
    const others=o.other_labels.length?` · 같은 자리 다른 이름: ${o.other_labels.map(l=>esc(l[0])+" "+l[1]).join(", ")}`:"";
    return `<li><b>${esc(o.name)}</b> ${pct} (${o.passed}프레임, 최고 ${o.best_confidence}) <span>(${o.place[0]}, ${o.place[1]})<br>${bins}${others}</span></li>`;
  }).join("")||"<li><span>아직 잡힌 물체 없음</span></li>";
}
function showWatch(w){
  showScene(w);
  if(w&&w.mode==="all")w=null;
  $("watch").hidden=!w;if(!w)return;
  const c=w.current||{reason:"none"};
  const why=c.reason.startsWith("clip:")?`CLIP 탈락 (${c.reason.slice(5)}위)`:(WHY[c.reason]||c.reason);
  const where=c.distance_m!=null?`, ${c.distance_m} m, ${c.angle_deg}°`:"";
  const conf=c.label?`${c.label} ${c.confidence}`:"";
  $("watchNow").className="now "+(c.reason==="ok"?"ok":c.reason==="none"?"none":"weak");
  $("watchNow").textContent=w.paused?"명령 실행 중: 시험 쉼":`${w.target}: ${why} ${conf}${where}`;
  const m=[`${w.frames}프레임 중 통과 ${w.passed}`,`최근 ${w.recent_rate==null?"-":Math.round(100*w.recent_rate)+"%"}`,`최고 신뢰도 ${w.best_confidence}`,w.confirmed?"발견 판정 가능":"발견 판정 아직"];
  if(w.elsewhere)m.push(`다른 자리 통과 ${w.elsewhere}`);
  if(w.labels_at_object&&w.labels_at_object.length)m.push("그 자리 이름: "+w.labels_at_object.map(l=>`${l[0]} ${l[1]}`).join(", "));
  $("watchMeta").textContent=m.join("  |  ");
  $("watchDist").innerHTML=rows(w.by_distance);$("watchAngle").innerHTML=rows(w.by_angle);
}
let streaming=false;
function startStream(){if(streaming)return;streaming=true;$("cam").src="/stream.mjpg";$("cam").hidden=false;$("nocam").hidden=true;}
$("cam").onerror=()=>{streaming=false;$("cam").hidden=true;$("nocam").hidden=false;};
async function poll(){
  try{
    const s=await (await fetch("/state")).json();
    if(s.image_seq>0)startStream();
    $("camsrc").hidden=!streaming||s.judged;
    showWatch(s.status&&s.status.watch);
    $("sharebox").hidden=!s.share_url;
    if(s.share_url)$("share").textContent=s.share_url;
    const st=s.status;
    $("state").textContent=st?st.text:"탐색 서버 상태 없음";
    const stale=s.status_age_s!=null&&s.status_age_s>3;
    $("stale").hidden=!stale;
    if(stale)$("stale").textContent=`탐색 서버 상태가 ${Math.round(s.status_age_s)}초째 안 옴: 탐색 서버 확인`;
    $("targets").innerHTML=(st&&st.targets||[]).map(t=>`<span class="chip ${esc(t.status)}">${esc(t.name)} · ${esc(t.text)}</span>`).join("");
    const m=[];
    if(st&&st.last)m.push("직전: "+st.last);
    if(s.result){
      m.push(`결과: ${s.result.text}${s.result.reason?" ("+s.result.reason+")":""}, ${s.result.steps}회 방문`);
    }
    $("meta").textContent=m.join("  |  ");
    $("resume").hidden=!(s.result&&s.result.status===4&&!s.running);
    $("checks").innerHTML=s.checks.map(c=>`<li>${esc(c.clock)} <b>${esc(c.target)}</b> ${esc(c.verdict_text)}${c.note?` <span>${esc(c.note)}</span>`:""}</li>`).join("")||"<li><span>아직 없음</span></li>";
  }catch(_){
    $("state").textContent="콘솔 서버 연결 끊김";
  }
}
poll();setInterval(poll,500);
</script></body></html>
"""
