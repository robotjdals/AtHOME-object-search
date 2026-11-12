// Marks the section on screen in both contents lists: the side rail (wide screens) and the top
// bar (below that). The rail shows once the header is behind.
(() => {
  const links = [...document.querySelectorAll(".toc a[href^='#'], .rail a[href^='#']")];
  const sections = [...new Set(links.map(a => a.hash))].map(h => document.querySelector(h)).filter(Boolean);
  const rail = document.querySelector(".rail"), hero = document.querySelector(".hero"), bar = document.querySelector(".toc ul");
  let queued = false, shown = null;

  function update() {
    queued = false;
    const line = window.innerHeight * 0.35;
    const atEnd = window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2;   // the last section may never reach the line
    const current = atEnd ? sections[sections.length - 1] : sections.filter(s => s.getBoundingClientRect().top <= line).pop();
    for (const a of links) {
      if (current && a.hash === "#" + current.id) a.setAttribute("aria-current", "location");
      else a.removeAttribute("aria-current");
    }
    rail.classList.toggle("on", window.scrollY > hero.offsetHeight * 0.8);
    const active = current && bar.querySelector(`a[href="#${current.id}"]`);
    if (active && current.id !== shown) {
      shown = current.id;
      bar.scrollLeft = Math.max(0, active.offsetLeft - 20);   // the bar scrolls sideways on phones
    }
  }
  addEventListener("scroll", () => { if (!queued) { queued = true; requestAnimationFrame(update); } }, { passive: true });
  addEventListener("resize", update);
  update();
})();
