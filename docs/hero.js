// First screen: the same placement as searched by the untrained and the trained planner, side by
// side. The recordings (assets/clips, 8x) start together and loop together; the counters follow
// the run record (assets/search_replay.json), so each shows what the robot has done so far.
(() => {
  const root = document.querySelector(".duel");
  if (!root) return;
  const SPEED = 8, LEAD_IN = 0.5, HOLD_MS = 3000;   // LEAD_IN as in replay.js
  const VERSION = new URL(document.currentScript.src).searchParams.get("v") || "1";
  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const sides = [...root.querySelectorAll(".duel-side")].map(el => ({
    el, video: el.querySelector("video"), count: el.querySelector(".duel-count"), variant: el.dataset.variant }));
  let visible = true, waiting = false;

  const text = (done, real, visits) => i18n.pick(
    `${done ? "Found · " : ""}${Math.round(real)} s · ${visits} visit${visits === 1 ? "" : "s"}`,
    `${done ? "찾음 · " : ""}${Math.round(real)}초 · ${visits}회 방문`);

  function paint() {
    for (const s of sides) {
      if (!s.run) continue;
      const v = s.video, done = v.ended;
      const real = done ? s.run.time_s : Math.min(s.run.time_s, v.currentTime * SPEED + (v.currentTime > 0 ? LEAD_IN : 0));
      const visits = done ? s.run.visits : s.visited.filter(t => t <= real).length;
      s.count.textContent = text(done, real, visits);
      s.el.classList.toggle("done", done);
    }
    requestAnimationFrame(paint);
  }

  const ready = v => v.readyState >= 3 ? Promise.resolve() : new Promise(r => v.addEventListener("canplay", r, { once: true }));
  const playAll = () => { for (const s of sides) if (!s.video.ended) s.video.play().catch(() => {}); };   // play() on an ended video would restart it
  async function start() {
    await Promise.race([Promise.all(sides.map(s => ready(s.video))), new Promise(r => setTimeout(r, 5000))]);
    for (const s of sides) s.video.currentTime = 0;
    if (visible) playAll();
  }
  function restart() {
    waiting = false;
    for (const s of sides) s.video.currentTime = 0;
    playAll();
  }
  const longest = () => sides.reduce((a, b) => ((b.video.duration || 0) > (a.video.duration || 0) ? b : a)).video;
  for (const s of sides) s.video.addEventListener("ended", () => {
    if (s.video !== longest() || waiting) return;
    waiting = true;
    setTimeout(() => { if (visible) restart(); else waiting = "later"; }, HOLD_MS);
  });
  new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    if (!visible) { for (const s of sides) s.video.pause(); return; }
    if (waiting === "later") restart(); else playAll();
  }, { threshold: 0.25 }).observe(root);

  fetch(`assets/search_replay.json?v=${VERSION}`).then(r => r.json()).then(d => {
    const sc = d.scenarios.find(x => String(x.scenario) === root.dataset.scenario);
    for (const s of sides) {
      s.run = sc[s.variant];
      s.visited = s.run.steps.filter(x => x.type === "visited").map(x => x.t);
    }
    requestAnimationFrame(paint);
    if (calm) for (const s of sides) s.video.controls = true;
    else start();
  }).catch(() => {});
})();
