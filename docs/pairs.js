// Placement by placement: one row per placement of the final experiment, the places each
// planner visited as a pair of dots. Rows come from the replay data (replay.js hands it over
// with the "replay-ready" event); selecting a row replays that search.
document.addEventListener("replay-ready", ev => {
  const { data, show } = ev.detail;
  const box = document.getElementById("pairs");
  if (!box) return;
  const rows = box.querySelector(".pairs-rows"), status = box.querySelector(".pairs-status");
  const hint = "Select a row to replay that search above.";
  const max = Math.ceil(Math.max(...data.scenarios.flatMap(s => [s.trained.visits, s.base.visits])) / 2) * 2;
  const pct = v => `${(v / max) * 100}%`;
  const node = (tag, cls, html = "") => {
    const e = document.createElement(tag);
    e.className = cls;
    e.innerHTML = html;
    return e;
  };
  const dot = (who, run) => `<i class="dot ${run.success ? who : "fail"}" data-v="${who}" style="left:${pct(run.visits)}"></i>`;
  const told = (who, run) => `${who} ${run.visits} visit${run.visits === 1 ? "" : "s"}, ${run.success ? `${Math.round(run.time_s)} s` : "not found"}`;

  box.style.setProperty("--ticks", max / 2);
  const ticks = Array.from({ length: max / 2 + 1 }, (_, i) => `<span style="left:${pct(i * 2)}">${i * 2}</span>`).join("");
  rows.append(node("div", "pairs-axis", `<span>Places visited</span><span class="ticks">${ticks}</span><span></span>`));

  const saved = s => s.base.visits - s.trained.visits;
  const groups = [["Trained planner visited fewer places", s => saved(s) > 0],
                  ["Same number", s => saved(s) === 0],
                  ["Trained planner visited more", s => saved(s) < 0]];
  for (const [title, test] of groups) {
    const members = data.scenarios.filter(test).sort((a, b) => saved(b) - saved(a) || a.scenario - b.scenario);
    if (!members.length) continue;
    rows.append(node("p", "pairs-group", `${title} (${members.length})`));
    for (const s of members) {
      const lo = Math.min(s.trained.visits, s.base.visits), hi = Math.max(s.trained.visits, s.base.visits);
      const row = node("button", "pair",
        `<span class="pair-name"><b>${s.object}</b> <span>${s.room.toLowerCase()}</span></span>`
        + `<span class="pair-track"><i class="link" style="left:${pct(lo)};width:${pct(hi - lo)}"></i>${dot("base", s.base)}${dot("trained", s.trained)}</span>`
        + `<span class="pair-val"><b>${s.trained.visits}</b> / ${s.base.visits}</span>`);
      row.type = "button";
      const text = `Scenario ${s.scenario}: ${s.object} in the ${s.room.toLowerCase()} (${s.typical} place) · `
        + `${told("trained", s.trained)} · ${told("untrained", s.base)}`;
      row.setAttribute("aria-label", `${text}. Replay this search.`);
      const tell = () => { status.textContent = text; };
      row.addEventListener("pointerenter", tell);
      row.addEventListener("focus", tell);
      row.addEventListener("click", () => show(s.scenario));
      rows.append(row);
    }
  }
  rows.addEventListener("pointerleave", () => { status.textContent = hint; });
  status.textContent = hint;
});
