// Overview video: muted loop, still when the visitor asked for less motion.
(() => {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) for (const v of document.querySelectorAll(".hero-video video")) { v.removeAttribute("autoplay"); v.pause(); }
  // The video further down starts when it scrolls into view and stops when it leaves.
  const later = document.querySelector(".flow-video video");
  if (later && "IntersectionObserver" in window && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    new IntersectionObserver(([e]) => { if (e.isIntersecting) later.play().catch(() => {}); else later.pause(); }, { threshold: 0.4 }).observe(later);
  }
})();
