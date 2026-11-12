// Fades each block of the page in as it scrolls into view; list items follow one after another.
(() => {
  const root = document.documentElement;
  if (!root.classList.contains("js")) return;   // set in <head> only when motion is welcome
  const skip = ".stats, .flow, .facts";
  const all = [...document.querySelectorAll("section > *, .stat, .flow > li, .facts > li")].filter(e => !e.matches(skip));
  const items = all.filter(e => !all.some(o => o !== e && o.contains(e)));   // the outer block only, not its parts
  const io = new IntersectionObserver(entries => {
    for (const en of entries) {
      if (en.isIntersecting || en.boundingClientRect.top < 0) {   // above the screen after a reload: show at once
        en.target.classList.add("in");
        io.unobserve(en.target);
      }
    }
  }, { threshold: 0.08, rootMargin: "0px 0px -4% 0px" });
  const seen = new Map();
  for (const e of items) {
    const n = seen.get(e.parentElement) || 0;
    seen.set(e.parentElement, n + 1);
    if (e.matches(".stat, .flow > li, .facts > li")) e.style.setProperty("--d", Math.min(n, 5) * 0.09 + "s");
    e.classList.add("reveal");
    io.observe(e);
  }
  addEventListener("scroll", () => {   // at the very end the last block may sit inside the margin
    if (scrollY + innerHeight >= root.scrollHeight - 2) for (const e of items) e.classList.add("in");
  }, { passive: true });
})();
