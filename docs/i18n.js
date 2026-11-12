// English / Korean switch. Static text: an element carries its Korean in data-ko (the English
// stays as written in the HTML), images in data-ko-alt. Scripts that draw text (replay.js,
// pairs.js) ask i18n.pick(en, ko) and redraw on "langchange". English is the default;
// ?lang=ko opens the Korean page, and the last choice is kept in localStorage.
(() => {
  const KEY = "athome-lang", html = document.documentElement;
  const TITLE = { en: document.title, ko: "AtHOME: 씬 그래프와 소형 LLM을 이용한 물체 탐색" };
  const WORDS = { laptop: "노트북", book: "책", plant: "식물", Home: "집", Bedroom: "침실", Kitchen: "주방", "Living room": "거실", Office: "사무실" };
  const saved = () => { try { return localStorage.getItem(KEY); } catch { return null; } };
  const keep = lang => { try { localStorage.setItem(KEY, lang); } catch { /* private window: the choice lasts for this visit only */ } };

  window.i18n = {
    get lang() { return html.lang; },
    pick: (en, ko) => (html.lang === "ko" ? ko : en),
    word: w => (html.lang === "ko" && WORDS[w]) || w,
  };

  // Intro paragraphs: a line may only break between clauses (after , . ; : ? !) as long as a clause
  // fits on a line, so "where it would be" is not split across lines. CSS (style.css, .cl) does the rest.
  function chunk(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    let text = "";
    while (walker.nextNode()) { nodes.push({ node: walker.currentNode, from: text.length }); text += walker.currentNode.data; }
    const starts = [0], ends = [];
    for (const m of text.matchAll(/[.,;:?!][”’)」]?\s+/g)) { ends.push(m.index + m[0].trimEnd().length); starts.push(m.index + m[0].length); }
    ends.push(text.trimEnd().length);
    if (starts.length < 2) return;
    const at = (pos, isEnd) => {
      const hit = nodes.find(n => (isEnd ? pos > n.from : pos >= n.from) && pos <= n.from + n.node.data.length);
      return [hit.node, pos - hit.from];
    };
    for (let k = starts.length - 1; k >= 0; k--) {   // from the end, so earlier positions stay valid
      const range = document.createRange();
      range.setStart(...at(starts[k], false));
      range.setEnd(...at(ends[k], true));
      const span = document.createElement("span");
      span.className = "cl";
      span.append(range.extractContents());
      range.insertNode(span);
    }
  }

  function apply(lang) {
    html.lang = lang;
    document.title = TITLE[lang];
    for (const el of document.querySelectorAll("[data-ko]")) {
      el.dataset.en ??= el.innerHTML;
      el.innerHTML = lang === "ko" ? el.dataset.ko : el.dataset.en;
    }
    for (const p of document.querySelectorAll("main section > p")) chunk(p);
    for (const img of document.querySelectorAll("[data-ko-alt]")) {
      img.dataset.enAlt ??= img.alt;
      img.alt = lang === "ko" ? img.dataset.koAlt : img.dataset.enAlt;
    }
    for (const el of document.querySelectorAll("[data-ko-label]")) {
      el.dataset.enLabel ??= el.getAttribute("aria-label");
      el.setAttribute("aria-label", lang === "ko" ? el.dataset.koLabel : el.dataset.enLabel);
    }
    for (const b of document.querySelectorAll(".lang button")) b.setAttribute("aria-pressed", String(b.dataset.lang === lang));
    document.dispatchEvent(new Event("langchange"));
  }

  const asked = new URLSearchParams(location.search).get("lang");
  apply([asked, saved()].find(l => l === "ko" || l === "en") || "en");
  for (const b of document.querySelectorAll(".lang button")) b.addEventListener("click", () => { keep(b.dataset.lang); apply(b.dataset.lang); });
})();
