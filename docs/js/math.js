// KaTeX auto-rendering, wired for Material for MkDocs instant navigation.
//
// Why not scope to `.arithmatex`?
// The build-time Systems generator emits several `$...$` / `$$...$$` snippets as
// RAW HTML — the Properties stat cards (`<div class="ts-prop-value">$D_{KY}=…$`),
// inline table cells, etc.  Those never pass through pymdownx.arithmatex, so they
// carry no `.arithmatex` wrapper.  Rendering only `.arithmatex` elements left that
// math as literal dollar-sign source on the page.
//
// Fix: render the whole content area (`.md-content`, else `<body>`).
// `renderMathInElement` walks text nodes, ignores code/pre/script/style by
// default, and skips already-rendered `.katex` output — so it safely upgrades
// BOTH the arithmatex blocks and the raw `$…$` snippets, and is idempotent across
// re-runs. Material's `document$` fires on every instant-nav page load, so the
// re-render wires correctly for `navigation.instant`.

const TSD_KATEX_DELIMITERS = [
  // Order matters: the longer `$$` / `\[` display delimiters must be tried
  // before the single-char `$` / `\(` inline ones.
  { left: "$$", right: "$$", display: true },
  { left: "\\[", right: "\\]", display: true },
  { left: "\\begin{aligned}", right: "\\end{aligned}", display: true },
  { left: "\\begin{align}", right: "\\end{align}", display: true },
  { left: "\\(", right: "\\)", display: false },
  { left: "$", right: "$", display: false },
];

function tsdRenderMath() {
  if (typeof renderMathInElement !== "function") return;
  const root = document.querySelector(".md-content") || document.body;
  if (!root) return;
  renderMathInElement(root, {
    delimiters: TSD_KATEX_DELIMITERS,
    // Never touch code — fenced blocks, inline code, or the copy widgets.
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    // Belt-and-braces: don't re-descend into already-rendered KaTeX output.
    ignoredClasses: ["katex", "katex-display"],
    throwOnError: false,
  });
}

if (typeof document$ !== "undefined" && document$.subscribe) {
  // Material instant-navigation: re-render after every page swap.
  document$.subscribe(() => {
    // Defer to the next frame so the swapped-in DOM is fully attached before we
    // walk it (KaTeX auto-render reads live text nodes).
    window.requestAnimationFrame(tsdRenderMath);
  });
} else {
  // Non-Material fallback (plain load).
  document.addEventListener("DOMContentLoaded", tsdRenderMath);
}
