// Videos: the overview plays muted in a loop unless the visitor asked for less motion; the
// console recordings of three placements show under the replay for the placement picked there.
(() => {
  const t = (en, ko) => i18n.pick(en, ko);
  const VERSION = new URL(document.currentScript.src).searchParams.get("v") || "1";

  const hero = document.querySelector(".hero-video video");
  if (hero && window.matchMedia("(prefers-reduced-motion: reduce)").matches) { hero.removeAttribute("autoplay"); hero.pause(); }

  const box = document.getElementById("clips");
  if (!box) return;
  const ids = box.dataset.scenarios.split(",").map(Number);
  const pair = box.querySelector(".clips-pair"), none = box.querySelector(".clips-empty"), chips = box.querySelector(".clips-chips");
  const videos = [...box.querySelectorAll("video")];

  document.addEventListener("replay-ready", ev => {
    const select = document.getElementById("replay-scenario");
    const find = id => ev.detail.data.scenarios.find(s => s.scenario === id);
    let current = Number(select.value);

    function draw() {
      chips.replaceChildren();
      for (const id of ids) {
        const s = find(id), b = document.createElement("button");
        b.type = "button";
        b.textContent = t(`${id} · ${s.object}, ${s.room.toLowerCase()}`, `${id} · ${i18n.word(s.object)}, ${i18n.word(s.room)}`);
        b.setAttribute("aria-pressed", String(id === current));
        b.addEventListener("click", () => { select.value = id; select.dispatchEvent(new Event("change")); });
        chips.append(b);
      }
      const has = ids.includes(current);
      pair.hidden = !has;
      none.hidden = has;
      if (!has) return;
      const s = find(current);
      for (const v of videos) {
        const run = s[v.dataset.v];
        if (v.dataset.id !== String(current)) {
          v.dataset.id = current;
          v.poster = `assets/clips/s${current}_${v.dataset.v}.jpg?v=${VERSION}`;
          v.src = `assets/clips/s${current}_${v.dataset.v}.mp4?v=${VERSION}`;
        }
        v.parentElement.querySelector(".clips-result").textContent = t(
          `${run.visits} visit${run.visits === 1 ? "" : "s"} · ${Math.round(run.time_s)} s`, `${run.visits}회 방문 · ${Math.round(run.time_s)}초`);
      }
    }

    box.querySelector(".clips-play").addEventListener("click", () => {
      for (const v of videos) { v.currentTime = 0; v.play().catch(() => {}); }
    });
    document.addEventListener("scenario-shown", e => { current = e.detail; draw(); });
    document.addEventListener("langchange", draw);
    draw();
  });
})();
