export const meta = {
  name: 'viz-m2-entrypoints',
  description: 'M2 entry points: plotly backend + JSON export — implement + adversarially verify',
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
    gates_green: { type: 'boolean' },
    acceptance_self_check: { type: 'string' },
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
    acceptance_checklist: { type: 'string' },
    findings: { type: 'string' },
    pr_ready: { type: 'boolean' },
  },
  required: ['id', 'worktree', 'branch', 'verdict', 'owns_respected', 'weakened_assertions_found', 'gates_green', 'pr_ready'],
}

const COMMON = `
You are implementing ONE ticket of the TSDynamics visualization program (M2). Repo root: ${ROOT}.
You work in a PRE-CREATED git worktree (already on the right branch, off the M1 integration tip
stream/viz-foundation which contains ALL of M1: the PlotSpec vocabulary, the dispatch seam
(tsdynamics.viz.render: register_builtin_renderers / render_spec / select_renderer / normalize_kind),
the capability layer (tsdynamics.viz.render.caps: RendererCapabilities / RenderResult / Renderer /
VisualizationDegraded), and the matplotlib reference renderer (tsdynamics.viz.render.mpl)).

ENVIRONMENT (follow exactly):
- The compiled engine is symlinked at src/tsdynamics/_rust.abi3.so. DO NOT run 'uv run'/'uv sync'/
  'maturin' (they rebuild Rust + resync uv.lock). Use the venv python directly: ${PY}
- From inside your worktree: PYTHONPATH=$PWD/src ${PY} -m pytest <files> -q -p no:cacheprovider -o addopts=""
  ; ${PY} -m ruff check <files> ; ${PY} -m ruff format <files> ; ${PY} -m mypy --strict src/tsdynamics

HOW A BACKEND PLUGS IN (study tsdynamics/viz/render/mpl/__init__.py — copy its shape):
- An in-tree backend is a module tsdynamics.viz.render.<name> exposing register(registry): it builds
  a renderer callable, attaches a RendererCapabilities descriptor, and registers it under its name —
  ONLY if its library imports (guarded), so a missing dep is a no-op and dispatch falls back.
- register_builtin_renderers() already iterates ('mpl','plotly','json','threejs'); a module that
  raises ImportError is skipped. So your new backend self-wires once its module exists.
- A backend that cannot draw a kind DECLINES via its RendererCapabilities (kinds=frozenset of what it
  supports) so select_renderer falls back to matplotlib (emitting VisualizationDegraded).

GUARDRAILS (non-negotiable):
- Edit ONLY your 'owns' globs (+ ONE focused test file tests/test_viz_<area>.py — a sanctioned owns
  expansion; list it). Touching any OTHER file is an auto-FAIL — self-report.
- Additive only. Do NOT break: tsdynamics.viz.spec (PlotSpec/PlotKind frozen vocab — extend never
  break), the matplotlib backend, the dispatch seam, the no-plot-import guarantee (import tsdynamics
  must NOT import matplotlib/plotly — keep all plot-lib imports lazy/in-method/in-register).
- NEVER weaken/delete/skip/xfail an assertion. mypy --strict GREEN over all of src/tsdynamics. ruff
  clean (line length 100, NumPy docstrings, imperative first line). No Julia/.jl references.
- Commit 'feat: [<ID>] <summary>' (a new renderer/exporter is a genuine new capability). DO NOT push
  or open a PR.
`

const TICKETS = [
  {
    id: 'PLOTLY-RENDER',
    worktree: `${ROOT}/.claude/worktrees/tsd-PLOTLY-RENDER`,
    owns: 'src/tsdynamics/viz/render/plotly/__init__.py (NEW) ; src/tsdynamics/viz/render/plotly/_core.py (NEW)',
    spec: `The plotly interactive 2-D backend. Create the package src/tsdynamics/viz/render/plotly/.
- __init__.py: register(registry) building a renderer that turns a PlotSpec into a
  plotly.graph_objects.Figure (interactive pan/zoom/hover). Attach RendererCapabilities named
  "plotly" declaring interactive=True + web_export=True, and the SET of 2-D kinds/marks it supports
  (LINE/SCATTER/MARKERS/IMAGE(heatmap)/HISTOGRAM/BAR/AREA/ERRORBAR + the 2-D semantic kinds) —
  DECLINE the 3-D marks (LINE3D/SURFACE3D) and anything it lacks, so dispatch falls back to
  matplotlib. Guard the whole thing behind 'import plotly' so a missing plotly is a no-op (registry
  fallback). Self-wires via register_builtin_renderers.
- _core.py: PlotSpec -> graph_objects traces for every supported 2-D kind/mark (go.Scatter for
  line/markers with mode, go.Heatmap for IMAGE, go.Histogram, go.Bar, fill='tonexty' band for AREA,
  error_y for ERRORBAR), applying axes (title/label/log scale/limits/categories), colorbar
  (cmap/clim), legend, and the vline/hline/text/span annotations (layout shapes). Normalise the
  semantic kind via tsdynamics.viz.render.normalize_kind. Use plotly's graph_objects ONLY (NO
  plotly.express, NO kaleido). Return the go.Figure.
plotly is NOT installed in this env — so your test file MUST importorskip('plotly') and exercise the
structural contract: register the backend, render representative 2-D specs to a go.Figure (assert
trace types/counts), and assert the caps DECLINE a 3-D spec (so select_renderer falls back to
matplotlib — assert that fallback selects matplotlib + warns VisualizationDegraded). Verify import
tsdynamics still pulls in NO plotly (lazy). Add tests/test_viz_plotly.py.`,
    adversarial: false,
  },
  {
    id: 'VIZ-JSON-EXPORT',
    worktree: `${ROOT}/.claude/worktrees/tsd-VIZ-JSON-EXPORT`,
    owns: 'src/tsdynamics/viz/export.py (NEW) ; src/tsdynamics/viz/render/json.py (NEW)',
    spec: `Versioned JSON export + a 'json' data-export renderer (pure-Python, NO extra dep).
- viz/export.py: to_json(spec) / from_json(text) built on the existing PlotSpec.to_dict /
  from_dict, stamping a SCHEMA_VERSION (e.g. {"schema_version": 1, "spec": spec.to_dict()}); round-
  trips losslessly (from_json(to_json(spec)) reproduces the spec; arrays survive). Document the
  stable schema in the module docstring. Tolerate a missing/older schema_version on read (back-compat).
- viz/render/json.py: register(registry) for a 'json' DATA-EXPORT backend (RendererCapabilities with
  data_export=True, kinds=None so it accepts everything — it serialises, it does not draw). Its
  renderer consumes a PlotSpec and RETURNS the JSON payload (the to_json string, or a RenderResult
  with payload+mimetype 'application/json'); if passed a 'path' kwarg, WRITE the file and return the
  path. Registers UNCONDITIONALLY (pure stdlib json — no optional dep), so register_builtin_renderers
  always wires it. spec.render('json') and result.plot(backend='json') return/write the payload.
Add tests/test_viz_json_export.py: to_json/from_json round-trip (incl. a 3-D spec + image spec +
every-kind smoke), schema_version stamp present, old-dict-without-version loads, spec.render('json')
returns the payload, and writing to a tmp_path file works. Engine-free.`,
    adversarial: true,
  },
]

phase('Implement')
const implStage = (t) =>
  agent(
    `${COMMON}\n\n=== YOUR TICKET: ${t.id} ===\nWorktree: ${t.worktree}\nowns: ${t.owns}\n\nTASK:\n${t.spec}\n\n` +
      `Steps: (1) cd ${t.worktree}; (2) read tsdynamics/viz/render/mpl/__init__.py (backend shape), ` +
      `tsdynamics/viz/render/__init__.py (dispatch), tsdynamics/viz/render/caps.py, ` +
      `tsdynamics/viz/spec.py (PlotSpec/to_dict); (3) implement within owns (+ your test file); ` +
      `(4) run gates (pytest your test file + tests/test_renderers_registry.py + ` +
      `tests/test_viz_fake_renderer.py, ruff check, ruff format, mypy --strict src/tsdynamics) — all ` +
      `green; (5) commit 'feat: [${t.id}] <summary>'. Report worktree abspath, branch, files changed, ` +
      `owns-respected, gate results, per-acceptance self-check. DO NOT push or PR.`,
    { label: `impl:${t.id}`, phase: 'Implement', schema: IMPL_SCHEMA, effort: 'high' }
  )

const verifyStage = (impl, t) => {
  if (!impl || !impl.worktree) {
    return { id: t.id, worktree: t.worktree, branch: '(unknown)', verdict: 'FAIL', owns_respected: false, weakened_assertions_found: false, gates_green: false, pr_ready: false, findings: 'implementer returned nothing usable' }
  }
  return agent(
    `${COMMON}\n\n=== ADVERSARIALLY VERIFY ${t.id} ===\nWorktree: ${t.worktree}\nowns: ${t.owns}\nacceptance:\n${t.spec}\n\n` +
      `You did NOT write this. Try to REFUTE it: (1) git -C ${t.worktree} diff stream/viz-foundation ` +
      `-- read the FULL diff; (2) confirm every changed file is within owns (a test file is allowed) ` +
      `else FAIL; (3) grep for removed assert / xfail / skip / loosened tol -> FAIL if any; (4) re-run ` +
      `the gates yourself (pytest + ruff + mypy --strict) — all green; (5) check EACH acceptance bullet ` +
      `against the actual code, esp: import tsdynamics imports NO plotly AND no matplotlib (fresh ` +
      `subprocess); the new backend self-registers via register_builtin_renderers; PLOTLY declines 3-D ` +
      `(dispatch falls back to matplotlib); JSON round-trips losslessly incl. a 3-D + image spec and ` +
      `stamps schema_version. Return PASS only if every bullet holds; else FAIL with specifics.`,
    { label: `verify:${t.id}`, phase: 'Verify', schema: VERDICT_SCHEMA, effort: 'high' }
  )
}

const clean = []
const reports = await pipeline(TICKETS, implStage, verifyStage)
for (const r of reports) if (r) clean.push(r)
log(`PASS: ${clean.filter((r) => r.verdict === 'PASS' && r.pr_ready).map((r) => r.id).join(', ') || 'none'}`)
log(`FAIL: ${clean.filter((r) => !(r.verdict === 'PASS' && r.pr_ready)).map((r) => r.id).join(', ') || 'none'}`)
return { reports: clean }
