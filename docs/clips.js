// Overview video: muted loop, still when the visitor asked for less motion.
(() => {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) for (const v of document.querySelectorAll(".hero-video video")) { v.removeAttribute("autoplay"); v.pause(); }
})();
