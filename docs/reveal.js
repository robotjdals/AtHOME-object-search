// Fades each block of the page in each time it scrolls into view; list items follow one after another.
(() => {
  const root = document.documentElement;
  if (!root.classList.contains("js")) return;   // set in <head> only when motion is welcome
  const skip = ".stats, .flow, .facts";
  const all = [...document.querySelectorAll("section > *, .stat, .flow > li, .facts > li")].filter(e => !e.matches(skip));
  const items = all.filter(e => !all.some(o => o !== e && o.contains(e)));   // the outer block only, not its parts
  const io = new IntersectionObserver(entries => {
    for (const en of entries) en.target.classList.toggle("in", en.isIntersecting);   // leaving the screen resets it, so it plays again
  }, { threshold: 0.08 });
  const seen = new Map();
  for (const e of items) {
    const n = seen.get(e.parentElement) || 0;
    seen.set(e.parentElement, n + 1);
    if (e.matches(".stat, .flow > li, .facts > li")) e.style.setProperty("--d", Math.min(n, 5) * 0.09 + "s");
    e.classList.add("reveal");
    io.observe(e);
  }
})();
