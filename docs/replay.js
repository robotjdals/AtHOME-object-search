// Search replay: the scene graph of the test home, lit up step by step as each
// planner searched it in the final experiment (docs/assets/search_replay.json,
// built by scripts/build_page_replay.py from the run records).
const SVG = "http://www.w3.org/2000/svg";
// The page loads this file as replay.js?v=<commit>; the data gets the same version, so a
// browser holding the page from cache (GitHub Pages: 10 min) never mixes old and new files.
const VERSION = new URL(document.currentScript.src).searchParams.get("v") || "1";
const W = 520, H = 340, COL = W / 4, NODE_W = 112, NODE_H = 42, ROOM_Y = 78, LOC_Y0 = 132, LOC_DY = 50;
// Seconds on screen before a step that comes at the same real time as the one before it, so
// the room choice and then the furniture choice are both seen; short, so that at 40x a visit
// (25-60 s real) still absorbs them and both panels stay on the same clock.
const GAP = { decide: 0.5, visit: 0.3 }, MIN_GAP = 0.2;

const root = document.getElementById("replay");
if (root) init(root);

async function init(root) {
  let data;
  try {
    data = await (await fetch(`assets/search_replay.json?v=${VERSION}`)).json();
  } catch (e) {
    root.querySelector(".replay-status").textContent = "Replay data could not be loaded.";
    return;
  }
  const select = root.querySelector("#replay-scenario");
  for (const s of data.scenarios) {
    const o = document.createElement("option");
    o.value = s.scenario;
    o.textContent = `Scenario ${s.scenario}: ${s.object} in the ${s.room.toLowerCase()} (${s.typical})`
      + ` · trained ${s.trained.visits} visits, untrained ${s.base.visits}`;
    select.append(o);
  }
  select.value = root.dataset.default || data.scenarios[0].scenario;

  const panels = [...root.querySelectorAll(".replay-panel")].map(el => makePanel(el, data.rooms));
  const play = root.querySelector("#replay-play"), speedSel = root.querySelector("#replay-speed");
  let scenario, start = 0, elapsed = 0, running = false, frame = null;

  function load() {
    scenario = data.scenarios.find(s => String(s.scenario) === select.value);
    for (const p of panels) p.load(scenario, scenario[p.variant]);
    elapsed = 0;
    render();
  }
  function render() {
    const speed = Number(speedSel.value);
    let done = true;
    for (const p of panels) done = p.show(elapsed, speed) && done;
    if (done && running) stop();
  }
  function tick() {
    elapsed = (performance.now() - start) / 1000;
    render();
    if (running) frame = setTimeout(tick, 40);
  }
  function startPlay() {
    if (panels.every(p => p.finished(elapsed, Number(speedSel.value)))) elapsed = 0;
    running = true;
    play.textContent = "Pause";
    start = performance.now() - elapsed * 1000;
    tick();
  }
  function stop() {
    running = false;
    play.textContent = "Play";
    clearTimeout(frame);
  }
  play.addEventListener("click", () => (running ? stop() : startPlay()));
  root.querySelector("#replay-restart").addEventListener("click", () => { stop(); elapsed = 0; render(); });
  select.addEventListener("change", () => { stop(); load(); });
  speedSel.addEventListener("change", () => { const was = running; stop(); elapsed = 0; render(); if (was) startPlay(); });
  load();
  // Start when the replay scrolls into view (not before: the visitor would miss the start).
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduce && "IntersectionObserver" in window) {
    const io = new IntersectionObserver(entries => {
      if (entries.some(e => e.isIntersecting)) { io.disconnect(); startPlay(); }
    }, { threshold: 0.4 });
    io.observe(root);
  }
}

const fit = (text, max = 19) => (text.length > max ? text.slice(0, max - 1) + "…" : text);

function el(name, attrs = {}, parent) {
  const e = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
}

function makePanel(container, rooms) {
  const variant = container.dataset.variant;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", class: "replay-graph" });
  container.querySelector(".replay-figure").append(svg);
  const status = container.querySelector(".replay-status");
  const clock = container.querySelector(".replay-clock");
  const nodes = {}, roomOf = {};

  const edges = el("g", { class: "edges" }, svg);
  const home = { x: W / 2, y: 26 };
  el("rect", { x: home.x - 34, y: home.y - 14, width: 68, height: 28, rx: 14, class: "node home" }, svg);
  el("text", { x: home.x, y: home.y + 4, class: "label strong" }, svg).textContent = "Home";

  rooms.forEach((room, i) => {
    const cx = COL * i + COL / 2;
    el("line", { x1: home.x, y1: home.y + 14, x2: cx, y2: ROOM_Y - 15, class: "edge" }, edges);
    const g = el("g", { class: "node-g" }, svg);
    el("rect", { x: cx - 52, y: ROOM_Y - 15, width: 104, height: 30, rx: 15, class: "node room" }, g);
    el("text", { x: cx, y: ROOM_Y + 5, class: "label strong" }, g).textContent = room.label;
    nodes[room.id] = g;
    const spine = cx - NODE_W / 2 + 8;
    const last = LOC_Y0 + LOC_DY * (room.locations.length - 1) + NODE_H / 2;
    el("line", { x1: spine, y1: ROOM_Y + 15, x2: spine, y2: last, class: "edge" }, edges);
    room.locations.forEach((loc, j) => {
      const y = LOC_Y0 + LOC_DY * j;
      el("line", { x1: spine, y1: y + NODE_H / 2, x2: spine + 6, y2: y + NODE_H / 2, class: "edge" }, edges);
      const lg = el("g", { class: `node-g ${loc.kind}` }, svg);
      el("rect", { x: spine + 6, y, width: NODE_W - 14, height: NODE_H, rx: 8, class: "node loc" }, lg);
      el("text", { x: spine + 14, y: y + 16, class: "label start" }, lg).textContent = loc.label;
      const sub = el("text", { x: spine + 14, y: y + 32, class: "sub start" }, lg);
      sub.textContent = fit(loc.objects.length ? loc.objects.join(", ") : (loc.kind === "standalone" ? "floor object" : ""));
      lg.dataset.objects = sub.textContent;
      nodes[loc.id] = lg;
      roomOf[loc.id] = room.id;
    });
  });

  let schedule = [], meta = null, run = null;
  return {
    variant,
    load(scenario, r) {
      meta = scenario;
      run = r;
      // Screen time: real time / speed, but at least MIN_GAP after the previous step.
      schedule = r.steps;
      svg.setAttribute("aria-label", `Scene graph of the test home, ${variant} planner searching for the ${scenario.object}`);
    },
    times(speed) {
      let prev = -1;
      return schedule.map(s => (prev = Math.max(s.t / speed, prev + (GAP[s.type] || MIN_GAP))));
    },
    finished(elapsed, speed) {
      const t = this.times(speed);
      return elapsed >= t[t.length - 1];
    },
    show(elapsed, speed) {
      const times = this.times(speed);
      const st = { roomCand: new Set(), room: null, locCand: new Set(), loc: null, visiting: null,
                   searched: new Set(), found: null, text: "Ready", visits: 0, realT: 0 };
      times.forEach((t, i) => { if (t <= elapsed) apply(st, schedule[i]); });
      const last = times.findLastIndex(t => t <= elapsed);
      const next = times.findIndex(t => t > elapsed);
      // Real-time clock, interpolated between steps.
      let real = last >= 0 ? schedule[last].t : 0;
      if (last >= 0 && next > 0) {
        const frac = (elapsed - times[last]) / (times[next] - times[last]);
        real = schedule[last].t + frac * (schedule[next].t - schedule[last].t);
      }
      paint(st);
      clock.textContent = `${Math.round(real)} s · ${st.visits} visit${st.visits === 1 ? "" : "s"}`;
      status.textContent = st.text;
      return next === -1;
    },
  };

  function label(id) {
    const n = nodes[id];
    return n ? n.querySelector("text").textContent : id;
  }

  function apply(st, s) {
    if (s.type === "decide" && s.stage === "room") {
      st.roomCand = new Set(s.candidates);
      st.room = s.chosen;
      st.locCand = new Set();
      st.loc = null;
      st.text = s.llm ? `Picks a room: ${label(s.chosen)} (LLM, ${s.latency_s.toFixed(1)} s)` : `Only room left: ${label(s.chosen)}`;
    } else if (s.type === "decide") {
      st.roomCand = new Set();
      st.locCand = new Set(s.candidates);
      st.loc = s.chosen;
      st.room = roomOf[s.chosen] || st.room;
      st.text = s.llm ? `Picks where to look: ${label(s.chosen)} (LLM, ${s.latency_s.toFixed(1)} s)` : `Only place left: ${label(s.chosen)}`;
    } else if (s.type === "visit") {
      st.roomCand = new Set();
      st.locCand = new Set();
      st.visiting = s.location;
      st.room = roomOf[s.location] || st.room;
      st.text = s.recheck ? `Looks at ${label(s.location)} again, from another spot` : `Drives to ${label(s.location)} and looks around`;
    } else if (s.type === "visited") {
      st.visiting = null;
      st.visits += 1;
      if (s.found) st.text = `Sees the ${meta.object}…`;
      else { st.searched.add(s.location); st.text = `Not at ${label(s.location)}`; }
    } else if (s.type === "end") {
      st.visiting = null;
      if (s.status === "found") {
        st.found = { loc: meta.placed, label: s.label || meta.object, seenFrom: s.location };
        const from = s.location && s.location !== meta.placed ? `, seen from ${label(s.location)}` : "";
        st.text = `Found the ${meta.object} at ${label(meta.placed)}${from}`;
      } else {
        st.text = `Time is up: not found (${run.time_s.toFixed(0)} s)`;
      }
    }
  }

  function paint(st) {
    for (const [id, g] of Object.entries(nodes)) {
      const cls = g.classList;
      cls.toggle("cand", st.roomCand.has(id) || st.locCand.has(id));
      cls.toggle("chosen", id === st.room || id === st.loc);
      cls.toggle("visiting", id === st.visiting);
      cls.toggle("searched", st.searched.has(id) && !(st.found && st.found.loc === id));
      cls.toggle("found", !!st.found && (id === st.found.loc || id === roomOf[st.found.loc]));
      cls.toggle("placed", meta && id === meta.placed);
      const sub = g.querySelectorAll("text")[1];
      if (sub) sub.textContent = st.found && st.found.loc === id ? fit(`+ ${st.found.label} ✓`) : g.dataset.objects;
    }
  }
}
