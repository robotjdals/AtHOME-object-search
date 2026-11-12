// Overview video: muted loop, still when the visitor asked for less motion. The chips above the
// replay jump to the placements that have a recording (replay.js plays it under each graph).
(() => {
  const t = (en, ko) => i18n.pick(en, ko);
  const hero = document.querySelector(".hero-video video");
  if (hero && window.matchMedia("(prefers-reduced-motion: reduce)").matches) { hero.removeAttribute("autoplay"); hero.pause(); }

  const box = document.getElementById("clips");
  if (!box) return;
  const ids = box.dataset.scenarios.split(",").map(Number), chips = box.querySelector(".clips-chips");

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
    }
    document.addEventListener("scenario-shown", e => { current = e.detail; draw(); });
    document.addEventListener("langchange", draw);
    draw();
  });
})();
