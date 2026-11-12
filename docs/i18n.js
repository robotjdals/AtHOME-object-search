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

  function apply(lang) {
    html.lang = lang;
    document.title = TITLE[lang];
    for (const el of document.querySelectorAll("[data-ko]")) {
      el.dataset.en ??= el.innerHTML;
      el.innerHTML = lang === "ko" ? el.dataset.ko : el.dataset.en;
    }
    for (const img of document.querySelectorAll("[data-ko-alt]")) {
      img.dataset.enAlt ??= img.alt;
      img.alt = lang === "ko" ? img.dataset.koAlt : img.dataset.enAlt;
    }
    for (const b of document.querySelectorAll(".lang button")) b.setAttribute("aria-pressed", String(b.dataset.lang === lang));
    document.dispatchEvent(new Event("langchange"));
  }

  const asked = new URLSearchParams(location.search).get("lang");
  apply([asked, saved()].find(l => l === "ko" || l === "en") || "en");
  for (const b of document.querySelectorAll(".lang button")) b.addEventListener("click", () => { keep(b.dataset.lang); apply(b.dataset.lang); });
})();
