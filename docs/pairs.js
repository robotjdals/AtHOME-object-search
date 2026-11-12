// Placement by placement: one row per placement of the final experiment, the places each
// planner visited as a pair of dots. Rows come from the replay data (replay.js hands it over
// with the "replay-ready" event); selecting a row replays that search.
(() => {
const t = (en, ko) => i18n.pick(en, ko);

document.addEventListener("replay-ready", ev => {
  const { data, show, clips } = ev.detail;
  const box = document.getElementById("pairs");
  if (!box) return;
  const rows = box.querySelector(".pairs-rows"), status = box.querySelector(".pairs-status");
  const hint = () => t("Select a row marked ▶ to replay that search above.", "▶ 표시가 있는 행을 선택하면 위의 리플레이에서 그 탐색을 다시 봅니다.");
  const max = Math.ceil(Math.max(...data.scenarios.flatMap(s => [s.trained.visits, s.base.visits])) / 2) * 2;
  const pct = v => `${(v / max) * 100}%`;
  const node = (tag, cls, html = "") => {
    const e = document.createElement(tag);
    e.className = cls;
    e.innerHTML = html;
    return e;
  };
  const dot = (who, run) => `<i class="dot ${run.success ? who : "fail"}" data-v="${who}" style="left:${pct(run.visits)}"></i>`;
  const told = (who, run) => t(
    `${who} ${run.visits} visit${run.visits === 1 ? "" : "s"}, ${run.success ? `${Math.round(run.time_s)} s` : "not found"}`,
    `${who === "trained" ? "학습" : "미학습"} ${run.visits}회 방문, ${run.success ? `${Math.round(run.time_s)}초` : "찾지 못함"}`);
  const roomName = s => t(s.room.toLowerCase(), i18n.word(s.room));

  box.style.setProperty("--ticks", max / 2);
  const saved = s => s.base.visits - s.trained.visits;

  function draw() {
    rows.replaceChildren();
    const ticks = Array.from({ length: max / 2 + 1 }, (_, i) => `<span style="left:${pct(i * 2)}">${i * 2}</span>`).join("");
    rows.append(node("div", "pairs-axis", `<span>${t("Places visited", "방문한 곳 수")}</span><span class="ticks">${ticks}</span><span></span>`));
    const groups = [[t("Trained planner visited fewer places", "학습한 플래너가 더 적게 방문"), s => saved(s) > 0],
                    [t("Same number", "같은 횟수"), s => saved(s) === 0],
                    [t("Trained planner visited more", "학습한 플래너가 더 많이 방문"), s => saved(s) < 0]];
    for (const [title, test] of groups) {
      const members = data.scenarios.filter(test).sort((a, b) => saved(b) - saved(a) || a.scenario - b.scenario);
      if (!members.length) continue;
      rows.append(node("p", "pairs-group", `${title} (${members.length})`));
      for (const s of members) {
        const lo = Math.min(s.trained.visits, s.base.visits), hi = Math.max(s.trained.visits, s.base.visits);
        const playable = clips.includes(s.scenario);
        const row = node("button", playable ? "pair" : "pair static",
          `<span class="pair-name"><b>${i18n.word(s.object)}</b> <span>${roomName(s)}</span>${playable ? ' <i class="rec">▶</i>' : ""}</span>`
          + `<span class="pair-track"><i class="link" style="left:${pct(lo)};width:${pct(hi - lo)}"></i>${dot("base", s.base)}${dot("trained", s.trained)}</span>`
          + `<span class="pair-val"><b>${s.trained.visits}</b> / ${s.base.visits}</span>`);
        row.type = "button";
        const text = t(`Scenario ${s.scenario}: ${s.object} in the ${s.room.toLowerCase()} (${s.typical} place) · `,
                       `시나리오 ${s.scenario}: ${i18n.word(s.room)}의 ${i18n.word(s.object)} (${s.typical === "typical" ? "전형적" : "비전형적"} 위치) · `)
          + `${told("trained", s.trained)} · ${told("untrained", s.base)}`;
        row.setAttribute("aria-label", `${text}. ${playable ? t("Replay this search.", "선택하면 이 탐색을 다시 봅니다.") : t("No recording for this placement.", "이 배치는 영상이 없습니다.")}`);
        if (!playable) row.setAttribute("aria-disabled", "true");
        const tell = () => { status.textContent = text; };
        row.addEventListener("pointerenter", tell);
        row.addEventListener("focus", tell);
        if (playable) row.addEventListener("click", () => show(s.scenario));
        rows.append(row);
      }
    }
    status.textContent = hint();
  }

  rows.addEventListener("pointerleave", () => { status.textContent = hint(); });
  document.addEventListener("langchange", draw);
  draw();
});
})();
