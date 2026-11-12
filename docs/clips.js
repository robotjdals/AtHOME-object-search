// Overview video: muted loop, still when the visitor asked for less motion.
(() => {
  const hero = document.querySelector(".hero-video video");
  if (hero && window.matchMedia("(prefers-reduced-motion: reduce)").matches) { hero.removeAttribute("autoplay"); hero.pause(); }
})();
