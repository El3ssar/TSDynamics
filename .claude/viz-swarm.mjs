export const meta = {
  name: 'viz-gapfill-swarm',
  description: 'M1 gap-fill + matplotlib renderer: implement + adversarially verify a file-disjoint batch',
  phases: [
    { title: 'Implement', detail: 'one pre-made worktree per ticket' },
    { title: 'Verify', detail: 'an independent skeptic refutes each diff' },
  ],
}

const PY = '/home/elessar/Projects/TSDynamics/.venv/bin/python3'
const ROOT = '/home/elessar/Projects/TSDynamics'

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    worktree: { type: 'string' },
    branch: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
    owns_respected: { type: 'boolean' },
    gates_green: { type: 'boolean', description: 'pytest + ruff + mypy all green' },
    test_tail: { type: 'string', description: 'last lines of the targeted pytest run' },
    acceptance_self_check: { type: 'string', description: 'each acceptance bullet -> met/not-met + evidence' },
    committed: { type: 'boolean' },
    notes: { type: 'string' },
  },
  required: ['id', 'worktree', 'branch', 'files_changed', 'owns_respected', 'gates_green', 'committed'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    worktree: { type: 'string' },
    branch: { type: 'string' },
    verdict: { type: 'string', enum: ['PASS', 'FAIL'] },
    owns_respected: { type: 'boolean' },
    weakened_assertions_found: { type: 'boolean' },
    gates_green: { type: 'boolean' },
    acceptance_checklist: { type: 'string', description: 'each acceptance bullet -> pass/fail with evidence from the diff/specs' },
    findings: { type: 'string' },
    pr_ready: { type: 'boolean' },
  },
  required: ['id', 'worktree', 'branch', 'verdict', 'owns_respected', 'weakened_assertions_found', 'gates_green', 'pr_ready'],
}

const COMMON = `
You are implementing ONE ticket of the TSDynamics visualization program (M1). Repo root: ${ROOT}.
You work in a PRE-CREATED git worktree (do NOT create a worktree, do NOT branch — it already
exists, on the correct branch, off the integration tip stream/viz-foundation which already
contains the foundation streams VIZ-VOCAB/VIZ-BASE/VIZ-SYSTEM-PLOT/VIZ-DISPATCH/etc.).

ENVIRONMENT (critical — follow exactly):
- Your worktree already has the compiled engine symlinked at src/tsdynamics/_rust.abi3.so.
- DO NOT run 'uv run' or 'uv sync' or 'maturin' — they rebuild Rust and resync uv.lock (slow,
  pollutes the diff). Use the venv python directly: ${PY}
- Run python/pytest/ruff/mypy with the worktree on PYTHONPATH, e.g. from inside your worktree:
    PYTHONPATH=$PWD/src ${PY} -m pytest <files> -q -p no:cacheprovider -o addopts=""
    ${PY} -m ruff check <changed files> ; ${PY} -m ruff format <changed files>
    ${PY} -m mypy --strict src/tsdynamics
  (-o addopts="" disables the repo's coverage gate so a targeted run isn't failed by coverage.)

THE FOUNDATION GIVES YOU (already on this branch — use these, do not redefine):
- New semantic PlotKinds: SPECTROGRAM, DIMENSION_SPECTRUM, COMPLEXITY_CURVE, ENSEMBLE_FAN,
  LYAPUNOV_SPECTRUM, EIGENVALUE_PLANE, FIXED_POINTS_OVERLAY, VECTOR_FIELD, PHASE_PORTRAIT_FIELD,
  CONTINUATION, CATEGORICAL_BAR, FEATURE_BARS (plus all the pre-existing ones: TIME_SERIES,
  PHASE_PORTRAIT_2D/3D, SPACETIME, BIFURCATION, ORBIT_DIAGRAM, COBWEB, RETURN_MAP,
  POINCARE_SECTION, BASINS_IMAGE, RECURRENCE_PLOT, POWER_SPECTRUM, SCALING_FIT, DIAGNOSTIC_CURVE,
  LINE_FAMILY, HISTOGRAM_NULL).
- New layer marks: BAR, AREA, ERRORBAR (plus LINE, LINE3D, SCATTER, MARKERS, IMAGE, QUIVER,
  SURFACE3D, HISTOGRAM).
- New Layer data channels (just dict keys on Layer.data; a renderer ignores ones it does not use):
  "c" (valid on a LINE/SCATTER too — colour-by-time/speed), "lo"/"hi" (band edges for AREA/
  ENSEMBLE_FAN), "err" (ERRORBAR magnitudes), "cat" (integer category index pairing with the
  categorical Axis.categories), "size" (per-point SCATTER size).
- Axis gained .categories + scale="categorical"; Colorbar gained .cmap / .norm / .discrete.
- AnalysisResult.overlay_on(base) and the static _overlay_on(spec, base) implement the overlay
  convention (host layers FIRST, then this result's layers). Use overlay_on for "draw X over a
  host figure" requirements (fixed points over a phase portrait, attractors over a basin image).
- Trajectory.to_plot_spec(kind=None) already exists and dispatches on is_discrete.
- The capability/dispatch seam: tsdynamics.viz.render (register_builtin_renderers, render_spec,
  normalize_kind, RendererCapabilities, RenderResult, Renderer, VisualizationDegraded).

GUARDRAILS (non-negotiable):
- Edit ONLY files matching your ticket's 'owns' globs. You MAY add a focused test file under
  tests/ for your area (e.g. tests/test_viz_<area>.py) — that is a sanctioned owns expansion;
  list it in your report. Touching any OTHER source file is an automatic FAIL — self-report it.
- to_plot_spec changes are ADDITIVE and must keep the existing gate green: the whole-layer
  contract test tests/test_viz_fake_renderer.py builds EVERY result synthetically and asserts
  to_plot_spec() returns a valid PlotSpec (real PlotKind, every layer a real mark, JSON
  round-trip) OR raises the documented VisualizationNotInstalled. After your change, run that
  gate; it MUST stay green. Also run tests/test_analysis_registry.py if you touch analysis.
- NEVER weaken/delete/skip/xfail an assertion or loosen a tolerance to make a test pass.
- Keep mypy --strict GREEN over all of src/tsdynamics. Common trap: do NOT add a 'base=' or any
  new parameter to a to_plot_spec override that its supertype lacks (Liskov — mypy [override]).
  The to_plot_spec(self, kind: str | None = None) signature is uniform across the hierarchy; for
  overlays use overlay_on, not a new parameter.
- NumPy-style docstrings; cite original papers, never competitor software; never reference the
  Julia dynamical-systems ecosystem; ruff line length 100; ruff format clean; D-rules on
  (imperative-mood first docstring line). Keep it import-light: building a spec must not import a
  plotting backend (matplotlib/plotly). Import numpy/scipy/symengine as the surrounding code does.
- Commit your work with a Conventional-Commit message: 'build: [<ID>] <summary>' (use 'feat:'
  only if it adds a genuinely new public capability like a new transform). DO NOT push, DO NOT
  open a PR (the orchestrator serializes git).
`

const TICKETS = [
  {
    id: 'GAPFILL-A',
    worktree: `${ROOT}/.claude/worktrees/tsd-GAPFILL-A`,
    owns: 'src/tsdynamics/data/trajectory.py ; src/tsdynamics/viz/producers.py (NEW FILE)',
    spec: `Core trajectory + system draw-views. In data/trajectory.py ENRICH Trajectory.to_plot_spec
and add the draw-view producers in a NEW module src/tsdynamics/viz/producers.py (pure spec
builders — engine-free, take arrays/Trajectory, return PlotSpec; NO rendering, NO engine import).
Requirements:
- map-orbit is SCATTER when is_discrete (NOT a connected LINE); multi-component + overlay time
  series (one LINE layer per named component, legend on); colour-by-time / colour-by-speed via the
  "c" channel on the LINE/SCATTER; an arbitrary component triple selects a PHASE_PORTRAIT_3D
  (LINE3D) — do NOT hardcode the first three components (accept component names/indices).
- DDE delay-embed view (x(t) vs x(t-tau)); VECTOR_FIELD + PHASE_PORTRAIT_FIELD (a QUIVER grid,
  optionally over a trajectory); 1-D COBWEB (the staircase x_{n+1} vs x_n with the y=x diagonal);
  a SPACETIME branch (component index vs time as an IMAGE, e.g. for a Lorenz96).
- Each produced spec emits the correct semantic kind and round-trips through to_dict/from_dict.
Add tests/test_viz_gapfill_a.py with engine-free spec-shape tests (build tiny arrays / a tiny
synthetic Trajectory; assert kinds, marks, channels, and round-trip). Keep test_viz_fake_renderer
green.`,
    adversarial: true,
  },
  {
    id: 'GAPFILL-B',
    worktree: `${ROOT}/.claude/worktrees/tsd-GAPFILL-B`,
    owns: 'src/tsdynamics/analysis/lyapunov/** ; src/tsdynamics/analysis/chaos/** ; src/tsdynamics/analysis/embedding/**',
    spec: `Lyapunov + chaos + embedding to_plot_spec. Requirements:
- LyapunovSpectrum -> LYAPUNOV_SPECTRUM as BAR layers (one bar per exponent; mark zero line).
- The mutual-information / autocorrelation first-minimum diagnostic -> DIAGNOSTIC_CURVE with the
  chosen lag annotated (a vline).
- EmbeddingDimension -> its curve (fnn fraction vs dimension, or Cao E1/E2) as DIAGNOSTIC_CURVE,
  selected embedding dim annotated. An Embedding point cloud -> phase portrait (2-D SCATTER /
  3-D LINE3D) via the existing ArrayResult behaviour or a bespoke spec.
- the 0-1 test translation-component plane (p,q) -> PHASE_PORTRAIT_2D (the bounded vs diffusive
  translation variables).
Each spec kind correct + round-trips. Do NOT change any estimator/numeric code — viz only. Add
tests/test_viz_gapfill_b.py (engine-free spec-shape). Keep test_viz_fake_renderer green.`,
  },
  {
    id: 'GAPFILL-C',
    worktree: `${ROOT}/.claude/worktrees/tsd-GAPFILL-C`,
    owns: 'src/tsdynamics/analysis/dimensions/** ; src/tsdynamics/analysis/entropy/**',
    spec: `Dimensions + entropy to_plot_spec. Requirements:
- DimensionSpectrum -> DIMENSION_SPECTRUM: D(q) vs q as a LINE plus an ERRORBAR layer using the
  "err" channel (error bars on D(q)).
- MultiscaleEntropy -> COMPLEXITY_CURVE (entropy vs scale factor).
- an entropy outcome distribution (e.g. permutation-pattern probabilities) -> CATEGORICAL_BAR
  (BAR layer + a categorical Axis.categories of the pattern labels).
- scalar-estimator results (single correlation/box-counting dimension etc.) keep a sensible
  wrapper spec (e.g. their scaling fit -> SCALING_FIT, already via ScalingResult — verify).
Each spec kind correct + round-trips. Viz only — no estimator changes. Add
tests/test_viz_gapfill_c.py. Keep test_viz_fake_renderer green.`,
  },
  {
    id: 'GAPFILL-D',
    worktree: `${ROOT}/.claude/worktrees/tsd-GAPFILL-D`,
    owns: 'src/tsdynamics/analysis/fixedpoints/** ; src/tsdynamics/analysis/orbits/**',
    spec: `Fixed points + orbits + sections to_plot_spec. Requirements:
- FixedPoint / FixedPointSet -> FIXED_POINTS_OVERLAY (SCATTER of the points, stable vs unstable
  styled differently) and an EIGENVALUE_PLANE (the eigenvalues/multipliers in the complex plane
  with the unit circle for maps / imaginary axis for flows). The overlay must carry the HOST
  layer first — use AnalysisResult.overlay_on(base) when given a host phase portrait.
- PeriodicOrbit (flow limit cycle) -> PHASE_PORTRAIT_3D (or 2-D) of the cycle + a Floquet-multiplier
  EIGENVALUE_PLANE; mark the trivial ~=1 multiplier distinctly.
- OrbitDiagram periods()/bifurcation_points() annotations on the bifurcation spec (vlines at the
  onsets). ReturnMap -> COBWEB (the staircase) in addition to its current scatter. estimate_period
  diagnostic curve (autocorrelation/FFT) -> DIAGNOSTIC_CURVE.
Each spec kind correct + round-trips. Viz only. Add tests/test_viz_gapfill_d.py. Keep
test_viz_fake_renderer green.`,
  },
  {
    id: 'GAPFILL-E',
    worktree: `${ROOT}/.claude/worktrees/tsd-GAPFILL-E`,
    owns: 'src/tsdynamics/derived/**',
    spec: `Derived-wrapper to_plot_spec. Requirements:
- StroboscopicMap: a strobe sampling is a SCATTER (discrete sampled points), NOT a connected LINE.
- EnsembleSystem: a trajectory collector + ENSEMBLE_FAN — a STATIC fan chart (median line + a
  shaded percentile band via the "lo"/"hi" channels on an AREA layer); lo <= hi must hold. NO
  animation.
- TangentSystem: a convergence-history result for the running Lyapunov estimates -> a
  DIAGNOSTIC_CURVE (each exponent's running estimate vs time, LINE_FAMILY/legend).
- PoincareMap / others: make sure each derived wrapper that can describe itself has a sensible
  to_plot_spec (delegating to the inner system's trajectory where natural).
Each spec kind correct + round-trips; the fan band renders (lo<=hi); the convergence curve renders.
Viz only — do NOT change the wrappers' dynamics. Add tests/test_viz_gapfill_e.py. Keep
test_viz_fake_renderer green.`,
  },
  {
    id: 'GAPFILL-F',
    worktree: `${ROOT}/.claude/worktrees/tsd-GAPFILL-F`,
    owns: 'src/tsdynamics/analysis/recurrence/** ; src/tsdynamics/analysis/surrogate/**',
    spec: `Recurrence + surrogate to_plot_spec (SPARSE — non-densifying). HARD REQUIREMENT:
- RecurrenceMatrix.to_plot_spec MUST emit the recurrence plot as SPARSE (i, j) coordinate arrays
  (a SCATTER of the recurrent pairs, kind RECURRENCE_PLOT) — it MUST NOT call .toarray() / densify
  the matrix (that is O(N^2) memory and OOMs at large N). Read the sparse matrix's COO (row/col)
  directly.
- WindowedRQA -> a measure-vs-window curve (DIAGNOSTIC_CURVE; pick a measure like determinism vs
  window centre) WITHOUT walking nested dense matrices.
- SurrogateEnsemble -> LINE_FAMILY (each surrogate a faint line) and/or an AREA band; the
  SurrogateTest null distribution keeps its HISTOGRAM_NULL with the data statistic + rejection
  tail shaded.
- a richer RQA readout (the scalar RQA measures as a small CATEGORICAL_BAR / labelled MARKERS).
Each spec kind correct + round-trips. Viz only. Add tests/test_viz_gapfill_f.py INCLUDING an OOM
regression guard: build a sparse RecurrenceMatrix at N=50000 (only a few thousand nonzeros) and
assert to_plot_spec()/to_dict() does NOT densify — i.e. the produced coordinate arrays have length
~= nnz (not N^2), and the call stays well under a fixed memory/byte budget. Keep
test_viz_fake_renderer green.`,
    adversarial: true,
  },
  {
    id: 'GAPFILL-G',
    worktree: `${ROOT}/.claude/worktrees/tsd-GAPFILL-G`,
    owns: 'src/tsdynamics/analysis/basins/**',
    spec: `Attractors + basins to_plot_spec. Requirements:
- AttractorSet -> a SCATTER of the attractor points (shared colour palette in meta["palette"],
  e.g. "tab20", with a fixed colour for the diverged set).
- BasinFractions -> CATEGORICAL_BAR (one bar per attractor id; categorical Axis.categories).
- ContinuationResult -> CONTINUATION (stacked basin fractions vs the swept parameter; tipping
  annotations — vlines — where a basin annihilates).
- UncertaintyExponent -> SCALING_FIT (it should reparent onto / present as the scaling curve
  f(eps) vs eps with the fitted slope; verify it already subclasses ScalingResult or give it a
  SCALING_FIT spec).
- BasinsResult stays BASINS_IMAGE (verify); support a 3-D slice image. Intra-result palette
  consistency: the same attractor id maps to the same colour across the AttractorSet scatter and
  the basin image (assert this in a test).
Each spec kind correct + round-trips. Viz only. Add tests/test_viz_gapfill_g.py. Keep
test_viz_fake_renderer green.`,
  },
  {
    id: 'GAPFILL-H',
    worktree: `${ROOT}/.claude/worktrees/tsd-GAPFILL-H`,
    owns: 'src/tsdynamics/transforms/**',
    spec: `Transforms to_plot_spec (+ a new spectrogram transform). Requirements:
- Add a new spectral.spectrogram transform (scipy STFT -> times, freqs, power) and register it
  like the other transforms; its result -> SPECTROGRAM kind (an IMAGE with a colorbar; cmap/log
  norm via Colorbar.cmap/norm).
- Spectrum (PSD) result -> POWER_SPECTRUM; FeatureSet result -> FEATURE_BARS (a BAR layer over a
  categorical Axis.categories of feature names; support a radar/parallel variant via meta).
- spectral-feature markers (e.g. dominant frequency, spectral centroid) as vline annotations on
  the POWER_SPECTRUM.
Wrap the transform return values in result types where needed so they carry to_plot_spec (mirror
the analysis result pattern; keep transforms self-registering into registry.transforms). Each spec
kind correct + round-trips. Add tests/test_viz_gapfill_h.py (and a property/round-trip test for the
new spectrogram). Keep test_viz_fake_renderer green and the transforms registry meta-QA green.`,
  },
  {
    id: 'VIZ-MPL-CORE',
    worktree: `${ROOT}/.claude/worktrees/tsd-VIZ-MPL-CORE`,
    owns: 'src/tsdynamics/viz/render/mpl/__init__.py (NEW) ; src/tsdynamics/viz/render/mpl/_core.py (NEW) ; tests/test_renderers_registry.py (owns expansion — see below)',
    spec: `The matplotlib 2-D reference renderer (the conformance oracle). Create the package
src/tsdynamics/viz/render/mpl/ with __init__.py and _core.py.
- Use matplotlib's OBJECT-ORIENTED API ONLY (matplotlib.figure.Figure + the Agg canvas); DO NOT
  import or use matplotlib.pyplot anywhere.
- __init__.py exposes register(registry): builds the renderer callable, attaches a
  RendererCapabilities (RendererCapabilities.all_kinds("matplotlib", supports_3d=True)), and
  registers it under name "matplotlib" (guarded so a second call is a no-op; only registers if
  matplotlib imports). This is the hook tsdynamics.viz.render.register_builtin_renderers calls.
- _core.py: a KIND_PRESETS table (semantic kind -> axis/aspect/colorbar setup) and a MARK_DISPATCH
  table (LINE, SCATTER, IMAGE, MARKERS, HISTOGRAM, BAR, AREA, ERRORBAR, QUIVER; is_discrete ->
  SCATTER; colour-by-"c" via a LineCollection). Apply axes (label/scale/limits/ticks/categories),
  colorbar (clim/cmap/norm/discrete), legend, and annotations (vline/hline/text/span). Honour the
  per-call PlotSpec tweaks already applied to the spec (relabel/rescale/limits/ticks/colorize).
  Normalise the spec's semantic kind through tsdynamics.viz.render.normalize_kind so alias/mark
  kinds resolve. The renderer returns the matplotlib Figure (or a RenderResult wrapping it).
  Every 2-D PlotKind must render on the Agg backend without error.
- 3-D marks (LINE3D/SURFACE3D) are the NEXT ticket (VIZ-MPL-3D) — you may leave a clean
  NotImplemented/skip path for them (capabilities still declare supports_3d=True so 3-D routes to
  matplotlib; a follow-up adds the 3-D drawing). Prefer: handle them minimally if easy, else raise
  a clear error only for 3-D marks.

CRITICAL test reconciliation (this is the FIRST real backend, so it flips viz from "deferred" to
"live" — you MUST update tests/test_renderers_registry.py, a sanctioned owns expansion, and keep
the WHOLE fast tier green):
- That file currently asserts the registry "ships empty" and that render('matplotlib') RAISES
  VisualizationNotInstalled (the deferred-era contract). Once register_builtin_renderers() picks
  up your mpl backend, render() will auto-register matplotlib and SUCCEED. Update those specific
  tests to the live contract: the registry is still empty AT IMPORT (lazy — assert matplotlib is
  not in sys.modules after import tsdynamics), but render('matplotlib') now returns a Figure;
  to test the genuine "no backend" path, force an empty registry (monkeypatch
  register_builtin_renderers to a no-op and clear the registry) and assert it still raises. Do NOT
  weaken the no-plot-import packaging guarantee.
- Verify the import guard holds: a fresh subprocess 'import tsdynamics' must NOT import matplotlib
  (registration is lazy, only on first render). Verify tests/test_viz_fake_renderer.py still passes
  (its fake renderer is registered first, so default selection still returns the fake).
- Add an Agg golden/smoke test: for each 2-D PlotKind build a minimal spec and assert
  spec.render("matplotlib") returns a Figure with axes and no exception (use Agg; close figures).
Mark matplotlib-using tests with importorskip("matplotlib").`,
    adversarial: true,
  },
]

phase('Implement')

const implStage = (t) =>
  agent(
    `${COMMON}\n\n=== YOUR TICKET: ${t.id} ===\nWorktree (already created, already on branch): ${t.worktree}\n` +
      `owns: ${t.owns}\n\nTASK:\n${t.spec}\n\n` +
      `Steps: (1) cd ${t.worktree}; (2) read the current code in your owns paths and the foundation ` +
      `(src/tsdynamics/viz/spec.py for the kinds/channels, src/tsdynamics/analysis/_result.py for ` +
      `overlay_on, src/tsdynamics/data/trajectory.py for the existing Trajectory.to_plot_spec); ` +
      `(3) implement strictly within owns (+ your one test file); (4) run the gates: targeted ` +
      `pytest (your test file + tests/test_viz_fake_renderer.py + any area gate), ruff check, ` +
      `ruff format, mypy --strict src/tsdynamics — all green; (5) git add only your files and ` +
      `commit 'build: [${t.id}] <summary>'. Report worktree abspath (git -C ${t.worktree} ` +
      `rev-parse --show-toplevel), branch, the exact files changed, whether every changed file is ` +
      `within owns (test file expansion noted), the gate results, and a per-acceptance-bullet ` +
      `self-check. DO NOT push or open a PR.`,
    { label: `impl:${t.id}`, phase: 'Implement', schema: IMPL_SCHEMA, effort: 'high' }
  )

const verifyStage = (impl, t) => {
  if (!impl || !impl.worktree) {
    return {
      id: t.id,
      verdict: 'FAIL',
      worktree: t.worktree,
      branch: '(unknown)',
      owns_respected: false,
      weakened_assertions_found: false,
      gates_green: false,
      findings: 'implementer returned nothing usable (likely a transient API/rate-limit failure)',
      pr_ready: false,
    }
  }
  const adv = t.adversarial
    ? '(6) ADVERSARIAL ticket: for GAPFILL-F PROVE no densification (the N=50000 sparse case keeps ' +
      'coordinate-array length ~= nnz, not N^2); for VIZ-MPL-CORE PROVE import tsdynamics does not ' +
      'import matplotlib and the empty-registry raise path still works; for GAPFILL-A re-check map ' +
      'orbits are SCATTER and 3-D is not first-3-hardcoded.'
    : ''
  return agent(
    `${COMMON}\n\n=== ADVERSARIALLY VERIFY TICKET ${t.id} ===\n` +
      `An implementer claims to have completed this ticket in the worktree:\n  ${t.worktree}\n` +
      `owns: ${t.owns}\nacceptance:\n${t.spec}\n\n` +
      `You did NOT write this code. Try to REFUTE it. Do: (1) git -C ${t.worktree} diff ` +
      `stream/viz-foundation -- read the FULL diff; (2) confirm EVERY changed file is within owns ` +
      `(a test file under tests/ for this area is allowed) — else FAIL; (3) grep the diff for ` +
      `removed assert / loosened atol|rtol / added xfail|skip -> FAIL if any unsanctioned; (4) ` +
      `re-run the gates in the worktree (PYTHONPATH=$PWD/src ${PY} -m pytest ` +
      `tests/test_viz_fake_renderer.py + the ticket test file + (analysis) ` +
      `tests/test_analysis_registry.py -q -p no:cacheprovider -o addopts="" ; ${PY} -m ruff check ` +
      `src/ ; ${PY} -m ruff format --check src/ ; ${PY} -m mypy --strict src/tsdynamics) — ALL must ` +
      `be green; (5) check EACH acceptance bullet against the ACTUAL diff/specs (build a spec and ` +
      `assert the kind/marks/channels), not the claim; ${adv} Return PASS only if every acceptance ` +
      `bullet holds and no guardrail is violated; else FAIL with specifics the implementer can act on.`,
    { label: `verify:${t.id}`, phase: 'Verify', schema: VERDICT_SCHEMA, effort: 'high' }
  )
}

// Throttle: process in sequential waves of 3 (impl->verify pipelined within a wave) to avoid the
// API rate-limit burst that killed the all-9-at-once launch.
const CHUNK = 3
const clean = []
for (let i = 0; i < TICKETS.length; i += CHUNK) {
  const group = TICKETS.slice(i, i + CHUNK)
  log(`wave ${i / CHUNK + 1}: ${group.map((t) => t.id).join(', ')}`)
  const groupReports = await pipeline(group, implStage, verifyStage)
  for (const r of groupReports) if (r) clean.push(r)
}

const passed = clean.filter((r) => r.verdict === 'PASS' && r.pr_ready)
log(`PASS: ${passed.map((r) => r.id).join(', ') || 'none'}`)
log(`FAIL: ${clean.filter((r) => !(r.verdict === 'PASS' && r.pr_ready)).map((r) => r.id).join(', ') || 'none'}`)
return { reports: clean }
