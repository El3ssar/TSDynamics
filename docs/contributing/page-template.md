---
description: The eight-section contract every analysis/transform documentation page follows, and the executable-documentation gate that enforces its python fences.
---

# Documentation page template

Every hand-written **analysis / transform capability page** under
`docs/analysis/` and `docs/transforms/` follows one shape — the *eight-section
contract* below. It exists so the docs read as one consistent reference rather
than 25 personal styles, and so the [executable-documentation
gate](#the-executable-documentation-gate) can run every page's code.

This page is the contract for new pages and the checklist a reviewer runs
against a page change. The per-system catalogue pages are **auto-generated**
(`hooks/docs_autogen.py`) and are *not* governed by this template.

---

## The eight sections

A page is these eight blocks, in order. Sections 6 and the figure caption are
the only optional ones; everything else is required.

### 1 — Front matter

A YAML front-matter block with a one-sentence `description:` (used as the page's
meta description and search snippet). Keep it to a single declarative sentence
naming what the page covers.

```markdown
---
description: Recurrence plots and recurrence quantification — determinism, laminarity and entropy from when a trajectory revisits its past.
---
```

### 2 — Kicker + H1 title + lede

A `<span class="ts-kicker">` kicker (`Analysis · NN` with the page's two-digit
ordinal), the single `#` H1 title, then a **lede**: two or three sentences that
say what the capability *is*, cite the **original literature** (never competitor
software — see the [glossary](glossary.md) and `CLAUDE.md`), and name the
entry-point functions.

```markdown
<span class="ts-kicker">Analysis · 07</span>

# Recurrence & RQA

A recurrence plot marks every pair of times $(i, j)$ at which a trajectory
returns close to a state it visited before (Eckmann, Kamphorst & Ruelle 1987).
…
```

### 3 — Showcase figure

A `<figure markdown>` block with the page's representative figure and a
`<figcaption>` that reads the picture for the reader. The image path is
`../assets/figures/<area>/<page>.png`.

```markdown
<figure markdown>
![recurrence showcase](../assets/figures/analysis/recurrence.png){ loading=lazy }
<figcaption>At a fixed recurrence rate the logistic map's plot tells the
regimes apart at a glance: a period-4 cycle recurs on clean diagonals, …</figcaption>
</figure>
```

The figure itself is **generated, not hand-drawn**. The figure-generation seam
is the result type's **`to_plot_spec()`** method (stream DOCS-ENG-PLOTSEAM): the
golden-figure generator calls `result.to_plot_spec()` to obtain the renderer-
neutral `PlotSpec` IR, renders it, and the output is validated against the
committed [golden corpus](../../docs/_tooling/golden/README.md). When you add a
new capability whose result needs a showcase figure, give its result class a
`to_plot_spec()` and name the placeholder image `analysis/<page>.png`; do not
commit a hand-made PNG.

### 4 — Capability table

A three-column table — **Function · Returns · Gives you** — linking each public
entry point on the page to its in-page section. This is the page's map.

```markdown
| Function | Returns | Gives you |
|---|---|---|
| [`recurrence_matrix`](#the-recurrence-matrix) | `RecurrenceMatrix` | the sparse plot $R_{ij}$ |
| [`rqa`](#quantification-rqa) | `RQAResult` | DET / LAM / L_max / ENTR / TT … |
```

### 5 — One `##` section per entry point

One `##` section per public function/result, each with a short prose
explanation **and a runnable ```python``` example**. The first argument is a
`system` or `data` per the [calling-convention glossary](glossary.md). Use the
reader-facing short names (`ts`, `np`, the system classes) — the gate seeds them
(see below). Show the result's salient attributes by evaluating them with a
trailing comment of the expected value.

```python
import tsdynamics as ts

traj = ts.Logistic(params={"r": 4.0}).iterate(steps=500, ic=[0.31])
rm = ts.recurrence_matrix(traj.y[:, 0], threshold=0.05)
rm.epsilon          # the radius actually used → 0.05
```

### 6 — Known values (optional)

When the capability has analytic / literature reference values (a tent map's
expansion entropy is exactly `ln 2`, the Lorenz Kaplan–Yorke dimension is
≈ 2.06), a `## Known values` section pins them in a runnable block, citing the
source. This doubles as a sanity check a reader can re-run.

### 7 — See also

A `## See also` section of cross-links to the sibling capability pages and the
relevant theory page. Keeps the docs a graph, not a list.

### 8 — References

A `## References` section listing the **original papers** in author-year form.
Never cite competitor software; ideas may be absorbed but citations go to the
literature (`CLAUDE.md`).

---

## The executable-documentation gate

Every ```python``` fence on a curated page is **executed under CI** by the
doctest gate (stream `DOCS-DOCTEST-GATE`):

- `tests/test_doctests.py` runs the [curated module
  doctests](#) and the curated doc pages;
- `tests/_doctest_select.py` holds the curated lists, the shared namespace and
  the page-fence executor;
- everything runs under the suite-wide `filterwarnings = error` (a stray
  `RuntimeWarning` fails the page), with a per-module `RuntimeWarning` allowlist
  for documented numerics that legitimately trip a benign `log(0)`/`0/0`.

### What the gate does to your fences

- **Each ```python``` fence on a page is run top-to-bottom as a script**, and a
  page shares **one namespace**, so a later block sees names an earlier block
  bound — write the page the way a reader runs it, sequentially.
- The gate **seeds the namespace** with the reader-facing names so import-light
  examples run as written: `np` (NumPy), `ts` (the package), every built-in
  system class (`Lorenz`, `Henon`, …) and every public top-level analysis name
  (`recurrence_matrix`, `lyapunov_spectrum`, …). You may still write explicit
  `import` lines for readability; they are harmless.
- A fence **must not raise** and **must not emit a warning** (other than an
  allowlisted `RuntimeWarning`). Make examples deterministic — pass an explicit
  `ic=`/`seed=` to anything that would otherwise draw a random initial condition,
  so the page never flakes under the strict filter.

### Opting a fence out

A fence that is deliberately **illustrative** — pseudo-code, or a fragment that
references a placeholder the reader supplies — opts out with a leading
`# skip-doctest` comment as its first line:

````markdown
```python
# skip-doctest
result = my_analysis(system)   # `system` is whatever you are studying
```
````

A fence written as a `>>>` **doctest transcript** is also skipped by the page
executor — those belong in a *docstring*, where the module-doctest half of the
gate runs them.

### Joining the curated set

A new page is gated once its relative path is added to `CURATED_PAGES` in
`tests/_doctest_select.py` (and a module's doctests once it is in
`CURATED_MODULES`). Add it only after every fence passes locally under the
strict filter:

```bash
uv run pytest tests/test_doctests.py -m "not full"
```

Heavy examples (a multi-minute parameter sweep) go in the `full` tier
(`FULL_ONLY_MODULES`) so the change-scoped inner loop stays fast; the broad
sweep runs nightly under `-m full`.
