/* Progressive enhancement for a public Plotly HTML export: keep the scientific
 * poster visible until the real player has mounted. No timers or fake playback. */
(() => {
  const poster = document.getElementById("tsd-live-poster");
  const status = document.getElementById("tsd-live-status");
  if (!poster || !status) return;
  let observer;
  function ready() {
    const graph = document.querySelector(".js-plotly-plot");
    if (!graph || !graph.querySelector('[aria-label="Animation controls"]')) return;
    poster.hidden = true;
    status.hidden = true;
    document.body.dataset.tsdReady = "true";
    observer?.disconnect();
  }
  function failed(event) {
    if (event.target?.tagName !== "SCRIPT") return;
    status.textContent = "Interactive view unavailable · showing the static image";
    observer?.disconnect();
  }
  observer = new MutationObserver(ready);
  observer.observe(document.body, { childList: true, subtree: true });
  window.addEventListener("error", failed, true);
  window.addEventListener("pagehide", () => {
    observer.disconnect();
    window.removeEventListener("error", failed, true);
  }, { once: true });
  ready();
})();
