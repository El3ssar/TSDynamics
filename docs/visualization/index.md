# Visualization

The backend-agnostic plotting layer. A plot is described once as a `PlotSpec`
intermediate representation and rendered by any of the pluggable backends, so
the same figure can become a static image, an interactive page, or an exported
data payload.

This section will cover:

- **PlotSpec** — the semantic, JSON-serializable description of a plot.
- **Renderers** — the matplotlib, plotly, three.js, and json backends, and how
  dispatch and fallback work.
- **Styling & themes** — the canonical style vocabulary and the built-in
  themes.
- **Animation** — the orthogonal animation modifier (comets, trails, spatial
  fields).
- **Composition** — overlaying and tiling panels into one figure.
- **Figure conventions** — the conventions the [system catalogue](../systems/)
  pages follow.
