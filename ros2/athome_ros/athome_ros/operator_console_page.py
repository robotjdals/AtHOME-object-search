"""The operator console's page (athome_ros.operator_console): one file, no
external resources, so it works without internet on the robot's network."""

PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AtHOME 탐색 콘솔</title>
<style>
:root{color-scheme:dark;--bg:#0e1215;--panel:#171c20;--raised:#20272c;--line:#2d363c;--text:#eef2f4;--dim:#9aa6ae;
--accent:#4a9bff;--accent-bg:#14304f;--ok:#37c871;--ok-bg:#123a23;--warn:#f2b632;--warn-bg:#42330d;--bad:#f25f63;--bad-bg:#4a1c1e}
*{box-sizing:border-box}
[hidden]{display:none!important}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,"Noto Sans CJK KR","NanumGothic",sans-serif;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:14px;padding:10px 14px;background:var(--panel);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
.brand{font-size:17px;font-weight:700;white-space:nowrap}.brand span{color:var(--dim);font-weight:500;margin-left:6px;font-size:14px}
#health{display:flex;gap:6px;flex-wrap:wrap;flex:1;min-width:0}
.hc{display:inline-flex;align-items:center;gap:6px;font-size:13px;padding:3px 10px;border-radius:999px;background:var(--raised);color:var(--dim);white-space:nowrap}
.hc i{width:8px;height:8px;border-radius:50%;background:#66727a}
.hc.ok{color:var(--text)}.hc.ok i{background:var(--ok)}
.hc.warn{color:var(--warn);background:var(--warn-bg)}.hc.warn i{background:var(--warn)}
.hc.bad{color:#ffb4b6;background:var(--bad-bg)}.hc.bad i{background:var(--bad)}
#elapsed{font:600 20px/1 ui-monospace,"DejaVu Sans Mono",monospace;font-variant-numeric:tabular-nums;white-space:nowrap}
button{font:inherit;font-weight:600;border:0;border-radius:8px;padding:9px 14px;background:var(--accent);color:#04121f;cursor:pointer}
button:hover{filter:brightness(1.1)}button:disabled{opacity:.45;cursor:default;filter:none}
button:focus-visible,input:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.alt{background:var(--raised);color:var(--text)}
button.small{padding:4px 10px;font-size:13px;border-radius:6px}
#resume{background:var(--ok);white-space:nowrap}
#stop{background:var(--bad);color:#fff;font-size:18px;font-weight:700;padding:10px 34px;white-space:nowrap}
main{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 410px;gap:12px;padding:12px}
.left{display:flex;flex-direction:column;gap:12px;min-height:0;min-width:0}
.hero{background:var(--panel);border:1px solid var(--line);border-left:5px solid #66727a;border-radius:10px;padding:12px 16px}
.hero[data-tone=work]{border-left-color:var(--accent)}.hero[data-tone=wait]{border-left-color:var(--warn)}
.hero[data-tone=good]{border-left-color:var(--ok);background:var(--ok-bg)}.hero[data-tone=mixed]{border-left-color:var(--warn);background:var(--warn-bg)}
.hero[data-tone=bad]{border-left-color:var(--bad);background:var(--bad-bg)}
.herohead{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
#badge{font-size:13px;font-weight:700;padding:2px 10px;border-radius:999px;background:var(--raised)}
.hero[data-tone=work] #badge{background:var(--accent-bg);color:#a9d1ff}.hero[data-tone=wait] #badge{background:var(--warn-bg);color:var(--warn)}
.hero[data-tone=good] #badge{background:var(--ok);color:#04210f}.hero[data-tone=mixed] #badge{background:var(--warn);color:#2a1e00}.hero[data-tone=bad] #badge{background:var(--bad);color:#fff}
#visitNo{color:var(--dim);font-size:13px}
#state{font-size:23px;font-weight:650;line-height:1.3;margin-top:6px;overflow-wrap:anywhere}
#advice{margin-top:4px;font-size:14px;min-height:20px;overflow-wrap:anywhere}
#stale{color:#ffb4b6;font-size:14px;font-weight:600;margin-top:4px}
#flow{display:flex;gap:4px;list-style:none;margin:10px 0 0;padding:0}
#flow li{flex:1;text-align:center;font-size:13px;padding:4px 2px;border-radius:5px;background:var(--raised);color:var(--dim);white-space:nowrap}
#flow li.past{color:var(--text)}#flow li.now{background:var(--accent);color:#04121f;font-weight:700}
#flow li.end{background:var(--ok);color:#04210f;font-weight:700}
#heads{display:flex;align-items:center;gap:4px;margin-top:8px;height:20px;font-size:13px;color:var(--dim)}
#heads b{flex:1;height:7px;border-radius:4px;background:var(--raised)}#heads b.past{background:#58707f}#heads b.now{background:var(--accent)}
#heads span{white-space:nowrap;margin-right:6px}
.view{flex:1;min-height:240px;background:#000;border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;align-items:center;justify-content:center;position:relative}
.view img{width:100%;height:100%;object-fit:contain;display:block;position:absolute;inset:0}
.none{color:var(--dim);padding:24px;text-align:center}
#camsrc{position:absolute;top:8px;left:8px;background:#000c;color:var(--warn);font-size:13px;padding:3px 10px;border-radius:5px;z-index:1}
aside{display:flex;flex-direction:column;background:var(--panel);border:1px solid var(--line);border-radius:10px;min-height:0;overflow:hidden}
.tabs{display:flex;border-bottom:1px solid var(--line);flex:none}
.tabs button{flex:1;background:none;color:var(--dim);border-radius:0;padding:11px 4px;border-bottom:3px solid transparent}
.tabs button[aria-selected=true]{color:var(--text);border-bottom-color:var(--accent)}
.tabs button .dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--warn);margin-left:6px;vertical-align:2px}
.tab{flex:1;overflow:auto;padding:14px;min-height:0}
h2{margin:0 0 8px;font-size:13px;font-weight:600;color:var(--dim)}
.block+.block{margin-top:18px}
input{width:100%;background:#0c0f12;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:10px;font:inherit}
input:disabled{opacity:.5}
.row{display:flex;gap:8px;margin-top:8px}.row>*{flex:1}
.note{color:var(--dim);font-size:12.5px;margin-top:6px}
#confirm{background:var(--raised);border:1px solid var(--accent);border-radius:10px;padding:12px;margin-top:10px}
#confirm h2{color:var(--text)}
#vocab{color:var(--warn)}
.card{background:var(--raised);border-radius:10px;padding:10px 12px;margin-bottom:8px;border-left:4px solid #66727a}
.card.searching{border-left-color:var(--accent)}.card.found{border-left-color:var(--ok)}.card.failed{border-left-color:var(--bad)}
.cardhead{display:flex;justify-content:space-between;align-items:center;gap:8px}
.cardhead b{font-size:16px;overflow-wrap:anywhere}
.pill{font-size:12.5px;font-weight:600;padding:1px 9px;border-radius:999px;background:#2f393f;color:var(--dim);white-space:nowrap}
.card.searching .pill{background:var(--accent-bg);color:#a9d1ff}.card.found .pill{background:var(--ok);color:#04210f}.card.failed .pill{background:var(--bad);color:#fff}
.where{color:var(--dim);font-size:13.5px;margin-top:4px;overflow-wrap:anywhere}
.shot{display:block;width:100%;padding:0;margin-top:8px;background:#000;border-radius:7px;overflow:hidden}
.shot img{display:block;width:100%;max-height:190px;object-fit:contain}
.verdict{display:flex;gap:6px;align-items:center;margin-top:8px}
.verdict button.good{background:var(--ok);color:#04210f}.verdict button.bad{background:var(--bad);color:#fff}.verdict button.miss{background:var(--warn);color:#2a1e00}
.done{font-size:13px;color:var(--dim)}
ol.visits,ul.plain{list-style:none;margin:0;padding:0}
ol.visits li{display:grid;grid-template-columns:26px 1fr auto;gap:0 8px;padding:8px 0;border-top:1px solid var(--line);font-size:13.5px}
ol.visits li:first-child{border-top:0}
.no{color:var(--dim);font-variant-numeric:tabular-nums}
.vmain{overflow-wrap:anywhere;min-width:0}.vsub{color:var(--dim);font-size:12.5px;grid-column:2/4;overflow-wrap:anywhere}
.vtime{color:var(--dim);white-space:nowrap;font-variant-numeric:tabular-nums}
.vfound{color:var(--ok);font-weight:600}.vbad{color:#ffb4b6}.vwarn{color:var(--warn)}
ul.plain li{padding:7px 0;border-top:1px solid var(--line);font-size:13.5px;overflow-wrap:anywhere}ul.plain li:first-child{border-top:0}
ul.plain li span{color:var(--dim)}
.empty{color:var(--dim);font-size:13.5px}
#watchNow{font-size:16px;font-weight:650;margin-top:10px;padding:9px;border-radius:7px;background:var(--raised);overflow-wrap:anywhere}
#watchNow.ok{background:var(--ok-bg)}#watchNow.weak{background:var(--warn-bg)}
.meta{color:var(--dim);font-size:13px;margin-top:6px;overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
th{font-weight:600;color:var(--dim);text-align:left}th+th,td+td{text-align:right}
td,th{padding:3px 4px;border-top:1px solid var(--line)}
#toast{position:fixed;left:50%;bottom:20px;transform:translateX(-50%);max-width:min(620px,92vw);background:#2a3238;border:1px solid var(--line);border-radius:9px;padding:10px 16px;font-size:14px;z-index:20;box-shadow:0 6px 24px #0009}
#toast.err{background:var(--bad-bg);border-color:var(--bad)}
#tab-exp table{font-size:13px}#tab-exp td,#tab-exp th{white-space:nowrap}#tab-exp td:last-child{white-space:normal}
.exphead{font-size:18px;font-weight:700;margin:2px 0 4px}.exphead[data-phase=place],.exphead[data-phase=verdict],.exphead[data-phase=manual]{color:var(--warn)}.exphead[data-phase=error]{color:var(--bad)}.exphead[data-phase=done]{color:var(--ok)}
dialog{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:10px;max-width:94vw}
dialog::backdrop{background:#000b}
dialog img{display:block;max-width:90vw;max-height:80vh}
dialog .row{justify-content:space-between;align-items:center}dialog .row>*{flex:none}
@media (max-width:1000px){
 html,body{height:auto}main{grid-template-columns:1fr}.view{flex:none;height:56vw;min-height:220px}aside{min-height:60vh}
 header{flex-wrap:wrap}#health{order:5;flex-basis:100%}#stop{margin-left:auto;padding:10px 22px}
 #state{font-size:20px}
}
</style></head><body>
<header>
 <div class="brand">AtHOME<span>물체 탐색</span></div>
 <div id="health" aria-label="시스템 상태"></div>
 <div id="elapsed" title="명령 경과 시간"></div>
 <button id="resume" hidden>재개</button>
 <button id="stop" title="진행 중인 명령을 취소합니다. 비상 정지가 아닙니다.">정지</button>
</header>
<main>
<div class="left">
 <section class="hero" id="hero" data-tone="idle">
  <div class="herohead"><span id="badge" role="status">연결 중</span><span id="visitNo"></span></div>
  <div id="state">콘솔 서버에 연결하는 중</div><div id="advice"></div><div id="stale" hidden></div>
  <ol id="flow" aria-label="진행 단계"><li>위치 결정</li><li>이동</li><li>관측</li><li>결과</li></ol>
  <div id="heads" aria-label="관측 방향"></div>
 </section>
 <div class="view"><div id="camsrc" hidden></div><img id="cam" alt="판정 카메라 영상" hidden><div id="nocam" class="none">카메라 영상 대기 중<br>탐색 서버와 인지 노드가 켜져 있는지 확인하세요</div></div>
</div>
<aside>
 <div class="tabs" role="tablist">
  <button role="tab" data-tab="search" aria-selected="true">탐색</button>
  <button role="tab" data-tab="test" aria-selected="false">인식 시험<span id="testDot" class="dot" hidden></span></button>
  <button role="tab" data-tab="log" aria-selected="false">수동 체크</button>
  <button role="tab" data-tab="exp" aria-selected="false" id="expTab" hidden>실험<span id="expDot" class="dot" hidden></span></button>
 </div>
 <div class="tab" id="tab-search">
  <div class="block">
   <form id="cmdform"><input id="cmd" placeholder="예: 컵 찾아줘" autocomplete="off" aria-label="자연어 명령">
   <div class="row"><button id="parse" type="submit">명령 해석</button></div></form>
   <div id="busy" class="note" hidden>명령 실행 중에는 새 명령을 넣을 수 없습니다.</div>
   <div id="percwait" class="note" hidden></div>
   <form id="confirm" hidden><h2>이 물체들을 찾을까요? (쉼표로 구분, 고칠 수 있음)</h2>
    <input id="targetsEdit" aria-label="찾을 물체" autocomplete="off"><div id="vocab" class="note"></div>
    <div class="row"><button id="go" type="submit">이대로 찾기</button><button id="drop" type="button" class="alt">취소</button></div></form>
  </div>
  <div class="block"><h2>찾을 물체</h2><div id="targets"><div class="empty">명령을 넣으면 여기에 표시됩니다.</div></div></div>
  <div class="block"><h2>방문 기록</h2><ol class="visits" id="visits"></ol><div id="visitsEmpty" class="empty">아직 방문한 곳이 없습니다.</div></div>
  <div class="block note">정지는 명령을 취소하고 로봇이 멈췄는지 확인합니다. 비상 정지가 아니니 조이스틱을 준비해 두세요.</div>
 </div>
 <div class="tab" id="tab-test" hidden>
  <div class="block">
   <h2>명령 없이 계속 판정</h2>
   <div class="row" style="margin-top:0"><button type="button" id="watchAll">전체 물체 시작 / 처음부터</button><button type="button" id="watchStop" class="alt">끝내기</button></div>
   <form id="watchform" style="margin-top:8px"><input id="watchTarget" placeholder="물체 하나만 자세히 (영어, 예: cup)" autocomplete="off" aria-label="시험할 물체"></form>
   <div class="note">물체를 놓고 조이스틱으로 거리와 각도를 바꿔 보세요. 잡힌 물체마다 줄이 생기고, 시야 안인데 놓친 프레임까지 셉니다. 명령을 넣으면 그동안 쉽니다.</div>
  </div>
  <div class="block" id="scene" hidden><div id="sceneMeta" class="meta"></div><ul class="plain" id="sceneRows" style="margin-top:6px"></ul></div>
  <div class="block" id="watch" hidden>
   <div id="watchNow"></div><div id="watchMeta" class="meta"></div>
   <table><thead><tr><th>거리</th><th>본 프레임</th><th>통과</th></tr></thead><tbody id="watchDist"></tbody></table>
   <table><thead><tr><th>각도</th><th>본 프레임</th><th>통과</th></tr></thead><tbody id="watchAngle"></tbody></table>
  </div>
 </div>
 <div class="tab" id="tab-log" hidden>
  <div class="block"><h2>직접 체크 (찾을 물체 카드의 버튼과 같음)</h2>
   <input id="checkTarget" placeholder="물체 이름" aria-label="체크할 물체">
   <input id="checkNote" placeholder="메모 (선택: 예: 주방 책상 왼쪽에 있었음)" aria-label="메모" style="margin-top:8px">
   <div class="row verdict"><button class="good" data-v="correct">맞음</button><button class="miss" data-v="missed">미검출</button><button class="bad" data-v="false_positive">오검출</button></div>
   <div class="note">누른 순간의 상태와 카메라 화면이 함께 저장됩니다.</div></div>
  <div class="block"><h2>최근 체크</h2><ul class="plain" id="checks"></ul></div>
  <div class="block" id="sharebox" hidden><h2>다른 컴퓨터에서 열기</h2><div id="share" class="meta" style="margin-top:0;user-select:all"></div></div>
 </div>
 <div class="tab" id="tab-exp" hidden>
  <div class="block"><h2>실험 진행 <span id="expProgress" class="meta"></span></h2>
   <div id="expPhase" class="exphead"></div>
   <div id="expText"></div>
   <div id="expScenario" class="meta"></div>
   <input id="expNote" placeholder="무효 이유 (예: 사람이 시야에 들어옴)" aria-label="무효 이유" autocomplete="off" hidden style="margin-top:8px">
   <div class="row" id="expButtons"></div>
  </div>
  <div class="block" id="expResult" hidden></div>
  <div class="block"><h2>시작 위치</h2><ul class="plain" id="expStarts"></ul>
   <div class="row"><button type="button" class="alt small" data-exp="check">경로 검사</button></div>
   <div id="expCheckOut" class="meta"></div>
   <div class="note">로봇을 각 시작 위치에 세우고 '현재 위치로 저장'을 누르세요. 저장 후 경로 검사로 모든 시나리오가 1-30 m인지 확인합니다.</div></div>
  <div class="block"><h2>최근 실행</h2>
   <table><thead><tr><th>회</th><th>물체</th><th>플래너</th><th>결과</th><th>방문</th><th>시간</th><th>판정</th></tr></thead><tbody id="expRuns"></tbody></table></div>
  <div class="block note" id="expPaths"></div>
 </div>
</aside>
</main>
<div id="toast" role="status" hidden></div>
<dialog id="photo"><img id="photoImg" alt="발견 사진"><div class="row"><span id="photoCap" class="meta"></span><button class="alt small" id="photoClose">닫기</button></div></dialog>
<script>
const $=id=>document.getElementById(id);
const esc=s=>{const d=document.createElement("div");d.textContent=s==null?"":String(s);return d.innerHTML;};
async function post(path,body={}){
  const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  return r.json();
}
let toastTimer;
function say(text,bad=false){
  clearTimeout(toastTimer);const t=$("toast");t.hidden=!text;t.textContent=text||"";t.classList.toggle("err",bad);
  if(text&&!bad)toastTimer=setTimeout(()=>{t.hidden=true;},5000);
}
const cache={};
function put(id,html){if(cache[id]!==html){cache[id]=html;$(id).innerHTML=html;}}
function clock(s){s=Math.floor(s);const m=Math.floor(s/60);return `${String(m).padStart(2,"0")}:${String(s%60).padStart(2,"0")}`;}
function seconds(s){return s<60?`${Math.round(s)}초`:clock(s);}

document.querySelectorAll("[data-tab]").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("[data-tab]").forEach(x=>x.setAttribute("aria-selected",x===b));
  document.querySelectorAll(".tab").forEach(x=>x.hidden=x.id!=="tab-"+b.dataset.tab);
});
$("cmdform").onsubmit=async e=>{
  e.preventDefault();$("parse").disabled=true;say("명령 해석 중…");
  try{
    const r=await post("/parse",{instruction:$("cmd").value});
    if(!r.ok){say(r.message,true);$("confirm").hidden=true;return;}
    say("");$("targetsEdit").value=r.targets.map(t=>t.name).join(", ");
    const out=r.targets.filter(t=>!t.known).map(t=>t.name);
    $("vocab").textContent=out.length?`학습 어휘 밖: ${out.join(", ")} (잘 못 찾을 수 있음)`:"";
    $("confirm").hidden=false;$("go").focus();
  }catch(_){say("콘솔 서버 연결 실패",true);}finally{$("parse").disabled=false;}
};
$("confirm").onsubmit=async e=>{
  e.preventDefault();
  const targets=$("targetsEdit").value.split(",").map(s=>s.trim()).filter(Boolean);
  $("go").disabled=true;
  try{
    const r=await post("/start",{targets});
    if(r.ok){say("");$("confirm").hidden=true;$("cmd").value="";if(targets.length===1)$("checkTarget").value=targets[0];}
    else say(r.message,true);
  }catch(_){say("콘솔 서버 연결 실패",true);}finally{$("go").disabled=false;}
};
$("drop").onclick=()=>{$("confirm").hidden=true;say("");};
$("stop").onclick=async()=>{
  say("취소 요청 중…");
  try{const r=await post("/stop");say(r.message,!r.ok);}catch(_){say("콘솔 서버 연결 실패: 조이스틱으로 정지하세요",true);}
};
$("resume").onclick=async()=>{const r=await post("/resume");say(r.ok?"재개했습니다":r.message,!r.ok);};
async function sendCheck(target,verdict,note=""){
  const r=await post("/check",{target,verdict,note});
  say(r.ok?`체크 저장: ${target}`:r.message,!r.ok);
  return r.ok;
}
document.querySelectorAll("[data-v]").forEach(b=>b.onclick=async()=>{
  const target=$("checkTarget").value.trim();
  if(!target){say("체크할 물체 이름을 적으세요",true);$("checkTarget").focus();return;}
  if(await sendCheck(target,b.dataset.v,$("checkNote").value))$("checkNote").value="";
});
$("targets").onclick=e=>{
  const b=e.target.closest("button");if(!b)return;
  if(b.dataset.verdict)sendCheck(b.dataset.target,b.dataset.verdict);
  else if(b.dataset.photo){$("photoImg").src="/found/"+encodeURIComponent(b.dataset.photo);$("photoCap").textContent=b.dataset.cap;$("photo").showModal();}
};
$("photoClose").onclick=()=>$("photo").close();
$("photo").onclick=e=>{if(e.target===$("photo"))$("photo").close();};
$("watchform").onsubmit=async e=>{
  e.preventDefault();const target=$("watchTarget").value.trim();
  if(!target){say("시험할 물체 이름을 적으세요",true);return;}
  await post("/watch",{target});$("checkTarget").value=target;
};
$("watchStop").onclick=()=>post("/watch",{target:""});
$("watchAll").onclick=()=>post("/watch",{all:true});

const WHY={ok:"통과",confidence:"이름은 맞지만 신뢰도 미달",outside:"탐색 구역 밖",none:"이 이름의 검출 없음"};
function rows(list){return list.filter(r=>r.frames).map(r=>`<tr><td>${esc(r.bin)}</td><td>${r.frames}</td><td>${r.passed} (${Math.round(100*r.passed/r.frames)}%)</td></tr>`).join("")||'<tr><td colspan="3">아직 없음</td></tr>';}
function showScene(w){
  const all=!!(w&&w.mode==="all");$("scene").hidden=!all;if(!all)return;
  $("sceneMeta").textContent=(w.paused?"명령 실행 중: 시험 쉼  |  ":"")+`${w.frames}프레임, 묻는 이름 ${w.names}개, 물체 ${w.objects.length}개`+(w.flickers?`, 한 번만 잡힌 것 ${w.flickers}개`:"");
  put("sceneRows",w.objects.map(o=>{
    const pct=o.rate==null?"-":Math.round(100*o.rate)+"%";
    const bins=o.by_distance.map(r=>`${esc(r.bin)} ${r.passed}/${r.frames}`).join(", ");
    const others=o.other_labels.length?` · 같은 자리 다른 이름: ${o.other_labels.map(l=>esc(l[0])+" "+l[1]).join(", ")}`:"";
    return `<li><b>${esc(o.name)}</b> ${pct} (${o.passed}프레임, 최고 ${o.best_confidence}) <span>(${o.place[0]}, ${o.place[1]})<br>${bins}${others}</span></li>`;
  }).join("")||"<li><span>아직 잡힌 물체 없음</span></li>");
}
function showWatch(w){
  $("testDot").hidden=!w;showScene(w);
  if(w&&w.mode==="all")w=null;
  $("watch").hidden=!w;if(!w)return;
  const c=w.current||{reason:"none"};
  const why=c.reason.startsWith("clip:")?`CLIP 탈락 (${c.reason.slice(5)}위)`:(WHY[c.reason]||c.reason);
  const where=c.distance_m!=null?`, ${c.distance_m} m, ${c.angle_deg}°`:"";
  const conf=c.label?`${c.label} ${c.confidence}`:"";
  $("watchNow").className=c.reason==="ok"?"ok":c.reason==="none"?"":"weak";
  $("watchNow").textContent=w.paused?"명령 실행 중: 시험 쉼":`${w.target}: ${why} ${conf}${where}`;
  const m=[`${w.frames}프레임 중 통과 ${w.passed}`,`최근 ${w.recent_rate==null?"-":Math.round(100*w.recent_rate)+"%"}`,`최고 신뢰도 ${w.best_confidence}`,w.confirmed?"발견 판정 가능":"발견 판정 아직"];
  if(w.elsewhere)m.push(`다른 자리 통과 ${w.elsewhere}`);
  if(w.labels_at_object&&w.labels_at_object.length)m.push("그 자리 이름: "+w.labels_at_object.map(l=>`${l[0]} ${l[1]}`).join(", "));
  $("watchMeta").textContent=m.join("  |  ");
  put("watchDist",rows(w.by_distance));put("watchAngle",rows(w.by_angle));
}

const STATE={idle:["대기","idle",-1],parsing:["명령 해석","work",-1],awaiting_confirmation:["확인 대기","wait",-1],
  inferring:["위치 결정","work",0],planning:["위치 결정","work",0],navigating:["이동","work",1],retry_wait:["주행 재시도 대기","wait",1],
  visiting:["방문","work",2],rotating:["회전","work",2],observing:["관측","work",2],confirming:["후보 확인","wait",2],relooking:["다시 보기","wait",2],
  stopping:["정지 중","bad",-1]};
const RESULT={completed:"탐색 완료",max_steps:"최대 횟수 도달",time_budget:"시간 초과",canceled:"취소됨",paused:"일시정지"};
function showHero(s){
  const st=s.status,tl=s.timeline||{},hero=$("hero");
  if(!st){hero.dataset.tone="bad";$("badge").textContent="연결 안 됨";$("state").textContent="탐색 서버 상태가 들어오지 않습니다";$("advice").textContent="탐색 launch가 켜져 있는지 확인하세요.";$("visitNo").textContent="";flow(-1);put("heads","");return;}
  let [badge,tone,step]=STATE[st.state]||[st.state,"work",-1],text=st.text,advice="";
  if(st.state==="done"){
    const ts=st.targets||[],found=ts.filter(t=>t.status==="found").length;
    badge=RESULT[st.result]||st.result;step=["paused","canceled"].includes(st.result)?-1:3;
    tone=st.result==="paused"?"bad":st.result==="canceled"?"idle":found===ts.length&&ts.length?"good":"mixed";
    text=ts.length?`${ts.length}개 중 ${found}개 찾음`:badge;
    const why=tl.ended&&tl.ended.explained;
    if(st.result==="paused"){text=why?"일시정지: "+why.text:st.text;advice=why&&why.action||"";if(tl.ended&&tl.ended.detail)advice+=` (${tl.ended.detail})`;}
    else if(st.result==="canceled")text="명령을 취소했습니다";
  }else if(st.state==="idle"){text="명령을 기다리는 중";advice="오른쪽에 찾을 물체를 입력하세요.";}
  hero.dataset.tone=tone;$("badge").textContent=badge;$("state").textContent=text;$("advice").textContent=advice;
  $("visitNo").textContent=st.step!=null&&st.state!=="done"?`${st.step}번째 방문${st.target?` · 목표 ${st.target}`:""}`:"";
  flow(step,st.state==="done");
  const h=st.heading;
  put("heads",!h?"":`<span>방향 ${h[0]}/${h[1]}</span>`+Array.from({length:h[1]},(_,i)=>`<b class="${i+1<h[0]?"past":i+1===h[0]?"now":""}"></b>`).join(""));
}
function flow(step,ended){
  [...$("flow").children].forEach((li,i)=>{li.className=i<step?"past":i===step?(ended?"end":"now"):"";});
}
function chip(tone,text,title=""){return `<span class="hc ${tone}" title="${esc(title)}"><i></i>${esc(text)}</span>`;}
function showHealth(s){
  const stale=s.status_age_s==null||s.status_age_s>3,c=[];
  c.push(s.status_age_s==null?chip("bad","탐색 서버 없음"):stale?chip("bad",`탐색 서버 ${Math.round(s.status_age_s)}초 무응답`):chip("ok","탐색 서버"));
  const cold=s.image_age_s==null||s.image_age_s>3;
  c.push(s.image_age_s==null?chip("bad","카메라 없음"):cold?chip("bad",`카메라 ${Math.round(s.image_age_s)}초 끊김`):s.judged?chip("ok","인지·카메라"):chip("warn","인지 결과 없음","원본 카메라만 들어옵니다. 인지 노드를 확인하세요."));
  if(s.perception&&!s.perception.ready)c.push(chip("warn",s.perception.short,s.perception.text));
  const p=(s.timeline||{}).planner;
  c.push(!p||!p.calls?chip("","플래너 대기","이번 명령에서 아직 추론하지 않았습니다"):p.last_fallback?chip("warn",`플래너 실패: 최소 비용 대체 ${p.fallbacks}회`,p.error):chip("ok",p.fallbacks?`플래너 (대체 ${p.fallbacks}회)`:"플래너",p.error));
  put("health",c.join(""));
  const judgedNote=$("camsrc");
  judgedNote.hidden=!streaming||(s.judged&&!cold);
  judgedNote.textContent=cold?"영상이 끊겼습니다: 마지막 화면":"원본 카메라 (인지 결과 없음)";
}
function showTargets(s){
  const st=s.status,tl=s.timeline||{},found=tl.found||{},list=st&&st.targets||[];
  const done=st&&st.state==="done";
  put("targets",list.map(t=>{
    const f=found[t.name];
    // Command numbers start again with every launch of the search server.
    const checks=(s.checks||[]).filter(c=>c.command_id===st.command_id&&c.target===t.name&&tl.started_at!=null&&c.time>=tl.started_at);
    const cls=t.status==="pending"?"":esc(t.status);
    let body="";
    if(t.status==="found"){
      const label=f&&f.label&&f.label!==t.name?` · 검출 이름 ${esc(f.label)}`:"";
      body+=`<div class="where">${f&&f.place?esc(f.place):"위치 정보 없음"}${label}</div>`;
      if(f&&f.photo)body+=`<button type="button" class="shot" data-photo="${esc(f.photo)}" data-cap="${esc(t.name)}"><img src="/found/${encodeURIComponent(f.photo)}" alt="${esc(t.name)} 발견 사진"></button>`;
    }
    if(checks.length)body+=`<div class="verdict done">체크함: ${esc(checks[0].verdict_text)}</div>`;
    else if(t.status==="found")body+=`<div class="verdict"><button type="button" class="small good" data-target="${esc(t.name)}" data-verdict="correct">맞음</button><button type="button" class="small bad" data-target="${esc(t.name)}" data-verdict="false_positive">오검출</button></div>`;
    else if(done)body+=`<div class="verdict"><button type="button" class="small miss" data-target="${esc(t.name)}" data-verdict="missed">있었는데 못 찾음</button></div>`;
    return `<div class="card ${cls}"><div class="cardhead"><b>${esc(t.name)}</b><span class="pill">${esc(t.text)}</span></div>${body}</div>`;
  }).join("")||'<div class="empty">명령을 넣으면 여기에 표시됩니다.</div>');
}
function showVisits(s){
  const visits=(s.timeline||{}).visits||[];
  $("visitsEmpty").hidden=visits.length>0;
  put("visits",visits.map(v=>{
    const out=v.found.length?`<span class="vfound">발견 ${esc(v.found.join(", "))}</span>`:v.status==="completed"?"목표 없음":v.status?`<span class="vbad">${esc(v.status_text)}${v.reason?": "+esc(v.reason):""}</span>`:'<span class="vwarn">진행 중</span>';
    const sub=[esc(v.stage),`목표 ${esc(v.target)}`,v.cost_m!=null?`거리 ${v.cost_m.toFixed(1)} m`:"",v.relooks?`다시 보기 ${v.relooks}회`:"",v.fallback?'<span class="vwarn">최소 비용 대체</span>':""].filter(Boolean).join(" · ");
    return `<li><span class="no">${v.step}</span><span class="vmain"><b>${esc(v.place)}</b> ${out}</span><span class="vtime">${v.duration_s!=null?seconds(v.duration_s):""}</span><span class="vsub">${sub}</span></li>`;
  }).join(""));
}
let streaming=false;
// One picture at a time, the next when the last has loaded (see latest_image in the console).
let shownSeq=-1,loading=false;
function showFrame(seq){
  if(loading||seq===shownSeq)return;
  loading=true;shownSeq=seq;$("cam").src="/frame.jpg?seq="+seq;
}
$("cam").onload=()=>{loading=false;streaming=true;$("cam").hidden=false;$("nocam").hidden=true;};
$("cam").onerror=()=>{loading=false;shownSeq=-1;streaming=false;$("cam").hidden=true;$("nocam").hidden=false;};
const EXP_BUTTONS={
  idle:[["start","실험 시작 / 이어하기",""]],done:[["start","실험 시작 / 이어하기","alt"]],error:[["start","다시 시작","" ]],
  place:[["placed","놓았음","good"],["skip","이 시나리오 건너뛰기","alt"],["abort","실험 중단","bad"]],
  running:[["abort","실험 중단","bad"]],
  manual:[["continue","계속","good"],["abort","실험 중단","bad"]],
  verdict:[["valid","유효","good"],["invalid","무효 (다시 시행)","miss"],["abort","실험 중단","bad"]]};
const TSTATUS={found:"발견",failed:"못 찾음",pending:"미완료",searching:"탐색 중"};
function showExperiment(s){
  const e=s.experiment;$("expTab").hidden=!e;if(!e)return;
  $("expDot").hidden=!e.busy;
  const p=e.progress;$("expProgress").textContent=`유효 ${p.valid}/${p.total}${p.invalid?` · 무효 ${p.invalid}`:""}`;
  $("expPhase").textContent=e.phase_text+(e.variant?` · planner ${e.variant}`:"");
  $("expPhase").dataset.phase=e.phase;
  $("expText").textContent=e.text||"";
  const sc=e.scenario;
  $("expScenario").textContent=sc?`시나리오 ${sc.scenario}: ${sc.object} → ${sc.room||""} ${sc.desk||""} · 시작 ${sc.start} · 순서 ${sc.order.join(" → ")}`:"";
  $("expNote").hidden=e.phase!=="verdict";
  const startsReady=e.start_names.every(n=>e.starts[n]);
  put("expButtons",(EXP_BUTTONS[e.phase]||[]).map(([a,label,cls])=>`<button type="button" class="${cls}" data-exp="${a}"${a==="start"&&!startsReady?' disabled title="시작 위치를 먼저 모두 저장하세요"':""}>${label}</button>`).join(""));
  const r=e.record;$("expResult").hidden=!(e.phase==="verdict"&&r);
  if(r){
    const shot=r.found_photo?`<button type="button" class="shot" data-photo="${esc(r.found_photo)}" data-cap="${esc(r.object)}"><img src="/found/${encodeURIComponent(r.found_photo)}" alt="발견 사진"></button>`:'<div class="empty">발견 사진 없음</div>';
    const warn=r.planner_variant_used&&r.planner_variant_used!==r.variant?`<div class="vbad">탐색 서버 기록의 planner: ${esc(r.planner_variant_used)}</div>`:"";
    put("expResult",`<h2>이번 결과</h2><div><b>${esc(r.object)}</b> ${esc(TSTATUS[r.target_status]||r.target_status||"-")} · ${esc(RESULT[r.status]||r.status||"-")} · 방문 ${r.steps_completed} · ${seconds(r.duration_s||0)}${r.fallbacks?` · <span class="vwarn">planner 대체 ${r.fallbacks}/${r.decisions}</span>`:""}</div>`+
      `<div class="meta">발견 위치 ${esc(r.found_location||"-")} · 시작 이동 ${esc(r.move_to_start&&r.move_to_start.status||"-")}</div>${warn}${shot}`);
  }
  put("expStarts",e.start_names.map(n=>{const v=e.starts[n];
    return `<li><b>${esc(n)}</b> ${v?`(${v[0].toFixed(2)}, ${v[1].toFixed(2)}, ${Math.round(v[2]*180/Math.PI)}°)`:'<span class="vwarn">저장 안 됨</span>'} <button type="button" class="alt small" data-exp="save" data-name="${esc(n)}"${e.busy?" disabled":""}>현재 위치로 저장</button></li>`;}).join(""));
  const c=e.check;
  put("expCheckOut",!c?"":c.filter(x=>!x.ok).map(x=>`시나리오 ${x.scenario} (${esc(x.object)}, ${esc(x.start)}): ${esc(x.problem)}`).join("<br>")||`모든 시나리오 통과 (최단 경로 ${Math.min(...c.map(x=>x.l_star_m)).toFixed(1)}~${Math.max(...c.map(x=>x.l_star_m)).toFixed(1)} m)`);
  put("expRuns",e.runs.map(x=>`<tr><td>${x.run}</td><td>${x.scenario}. ${esc(x.object)}</td><td>${esc(x.variant)}</td><td>${esc(TSTATUS[x.target_status]||x.status||"-")}</td><td>${x.steps_completed??"-"}</td><td>${x.duration_s!=null?seconds(x.duration_s):"-"}</td><td>${x.valid?"유효":`<span class="vbad">무효</span> ${esc(x.note||"")}`}</td></tr>`).join("")||'<tr><td colspan="7">아직 없음</td></tr>');
  $("expPaths").textContent=`계획 ${e.plan} · 기록 ${e.out}`;
}
$("tab-exp").onclick=async ev=>{
  const b=ev.target.closest("button");if(!b)return;
  if(b.dataset.photo){$("photoImg").src="/found/"+encodeURIComponent(b.dataset.photo);$("photoCap").textContent=b.dataset.cap;$("photo").showModal();return;}
  const a=b.dataset.exp;if(!a)return;
  let r;b.disabled=true;
  try{
    if(a==="save")r=await post("/exp/save_start",{name:b.dataset.name});
    else if(a==="check"){say("경로 검사 중…");r=await post("/exp/check");}
    else if(a==="valid")r=await post("/exp/verdict",{valid:true});
    else if(a==="invalid"){const note=$("expNote").value.trim();if(!note){say("무효 이유를 적으세요",true);$("expNote").focus();return;}r=await post("/exp/verdict",{valid:false,note});if(r.ok)$("expNote").value="";}
    else if(a==="abort"){if(!confirm("실험을 중단할까요? 진행 중인 이동이나 명령을 취소합니다."))return;r=await post("/exp/abort");}
    else r=await post("/exp/"+a);
    say(r.message||(r.ok?"":"실패"),!r.ok);
  }catch(_){say("콘솔 서버 연결 실패",true);}finally{b.disabled=false;}
};
async function poll(){
  try{
    const s=await (await fetch("/state")).json();
    if(s.image_seq>0)showFrame(s.image_seq);
    showWatch(s.status&&s.status.watch);showHero(s);showHealth(s);showTargets(s);showVisits(s);showExperiment(s);
    const stale=s.status_age_s!=null&&s.status_age_s>3;
    $("stale").hidden=!stale;
    if(stale)$("stale").textContent=`탐색 서버 상태가 ${Math.round(s.status_age_s)}초째 안 옵니다. 아래 내용은 오래된 정보입니다.`;
    const el=(s.timeline||{}).elapsed_s;$("elapsed").textContent=el==null?"":clock(el);
    const st=s.status,state=st&&st.state;
    $("resume").hidden=!(state==="done"&&st.result==="paused"&&!s.running);
    const busy=!!st&&!["idle","done","parsing","awaiting_confirmation"].includes(state);
    const waiting=!!s.perception&&!s.perception.ready;
    $("percwait").hidden=!waiting||busy;$("percwait").textContent=waiting?s.perception.text:"";
    $("resume").disabled=waiting;
    const exp=!!(s.experiment&&s.experiment.busy);
    $("busy").hidden=!busy&&!exp;$("busy").textContent=exp?"실험 진행 중에는 명령을 넣을 수 없습니다 (실험 탭).":"명령 실행 중에는 새 명령을 넣을 수 없습니다.";
    $("cmd").disabled=busy||exp;$("parse").disabled=busy||exp;$("go").disabled=busy||waiting||exp;
    $("sharebox").hidden=!s.share_url;
    if(s.share_url)$("share").textContent=s.share_url;
    put("checks",(s.checks||[]).map(c=>`<li>${esc(c.clock)} <b>${esc(c.target)}</b> ${esc(c.verdict_text)}${c.note?` <span>${esc(c.note)}</span>`:""}</li>`).join("")||'<li><span>아직 없음</span></li>');
  }catch(_){
    $("hero").dataset.tone="bad";$("badge").textContent="연결 끊김";$("state").textContent="콘솔 서버 연결이 끊겼습니다";$("advice").textContent="정지가 필요하면 조이스틱을 쓰세요.";
    put("health",chip("bad","콘솔 서버 끊김"));
  }
}
poll();setInterval(poll,500);
</script></body></html>
"""
