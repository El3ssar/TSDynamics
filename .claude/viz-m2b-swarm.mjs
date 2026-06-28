export const meta = {
  name: 'viz-m2b',
  description: 'M2 tail: plotly 3D + plotly HTML + three.js export — implement + verify',
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
    id: { type: 'string' }, worktree: { type: 'string' }, branch: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
    owns_respected: { type: 'boolean' }, gates_green: { type: 'boolean' },
    acceptance_self_check: { type: 'string' }, committed: { type: 'boolean' }, notes: { type: 'string' },
  },
  required: ['id', 'worktree', 'branch', 'files_changed', 'owns_respected', 'gates_green', 'committed'],
}
const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string' }, worktree: { type: 'string' }, branch: { type: 'string' },
    verdict: { type: 'string', enum: ['PASS', 'FAIL'] }, owns_respected: { type: 'boolean' },
    weakened_assertions_found: { type: 'boolean' }, gates_green: { type: 'boolean' },
    acceptance_checklist: { type: 'string' }, findings: { type: 'string' }, pr_ready: { type: 'boolean' },
  },
  required: ['id', 'worktree', 'branch', 'verdict', 'owns_respected', 'weakened_assertions_found', 'gates_green', 'pr_ready'],
}

const COMMON = `
You are implementing ONE ticket of the TSDynamics visualization program (M2 tail). Repo root: ${ROOT}.
You work in a PRE-CREATED worktree (already on the right branch, off the integration tip
stream/viz-foundation which contains ALL of M1 + the plotly backend (tsdynamics.viz.render.plotly)
+ the json data-export backend (tsdynamics.viz.render.json) + viz/export.py (to_json/from_json,
SCHEMA_VERSION)). Study those before writing.

ENVIRONMENT: engine symlinked at src/tsdynamics/_rust.abi3.so. DO NOT 'uv run'/'uv sync'/'maturin'.
Use ${PY} directly with PYTHONPATH=$PWD/src. Tests: -p no:cacheprovider -o addopts="". Gates:
ruff check, ruff format, mypy --strict src/tsdynamics — all must be green.

KEY CONTRACTS (do not break — additive only):
- The dispatch (tsdynamics.viz.render): register_builtin_renderers / select_renderer / normalize_kind.
  Default selection skips data_export backends; a backend DECLINES kinds it lacks via its
  RendererCapabilities so dispatch falls back to matplotlib.
- import tsdynamics must import NO plot library (matplotlib/plotly) — keep all plot/express/io imports
  lazy/in-method/in-register (study viz/render/plotly/__init__.py + mpl/__init__.py).
- plotly is NOT installed in this env → plotly tests MUST importorskip('plotly'); exercise the
  structural/capability contract without it where possible.

GUARDRAILS: edit ONLY your 'owns' globs (+ ONE focused test file tests/test_viz_<area>.py — sanctioned,
list it). Never weaken/delete/xfail an assertion. ruff line-length 100, NumPy docstrings (imperative
first line), no Julia refs. Commit 'feat: [<ID>] <summary>'. DO NOT push or open a PR.
`

const TICKETS = [
  {
    id: 'PLOTLY-3D',
    worktree: `${ROOT}/.claude/worktrees/tsd-PLOTLY-3D`,
    owns: 'src/tsdynamics/viz/render/plotly/_threed.py (NEW) ; + the minimal wiring in plotly/_core.py + plotly/__init__.py caps (owns expansion, note it)',
    spec: `Plotly 3-D marks. Add src/tsdynamics/viz/render/plotly/_threed.py drawing LINE3D
(go.Scatter3d mode='lines'), SCATTER/MARKERS-in-3D (go.Scatter3d mode='markers', colour-by-c),
and SURFACE3D (go.Surface) — an orbitable 3-D figure. Read the camera from spec.meta['camera']
(eye/up) if present. Wire it: plotly/_core.py (or wherever the plotly render dispatches) routes a
3-D spec (ndim==3 / z axis / LINE3D/SURFACE3D mark) to _threed; plotly/__init__.py caps gain
supports_3d=True and the 3-D kinds so plotly no longer declines them (so a Lorenz attractor is
interactively orbitable in plotly). graph_objects ONLY (no express, no kaleido). Keep the 2-D path
unchanged. Test (tests/test_viz_plotly_3d.py, importorskip('plotly')): a LINE3D/SURFACE3D spec
renders to a go.Figure carrying a Scatter3d/Surface trace; caps now accept a 3-D spec. Without
plotly, assert caps.supports_3d is True structurally.`,
    adversarial: false,
  },
  {
    id: 'PLOTLY-HTML',
    worktree: `${ROOT}/.claude/worktrees/tsd-PLOTLY-HTML`,
    owns: 'src/tsdynamics/viz/render/plotly/_html.py (NEW) ; + a thin hook so the plotly renderer accepts an html=/path= kwarg (owns expansion in plotly/_core.py or __init__.py, note it)',
    spec: `Self-contained HTML export for docs/web. Add src/tsdynamics/viz/render/plotly/_html.py:
to_html(figure_or_spec, *, full_html=False, include_plotlyjs='cdn') -> str using plotly's
figure.to_html / plotly.io.to_html (include_plotlyjs='cdn' so the fragment is small and needs no
kernel; full_html=False for an embeddable mkdocs fragment). Expose it so result.plot(backend='plotly',
html=True) (or a path= to write the file) returns/writes the HTML — wire a thin kwarg in the plotly
renderer (owns expansion). The fragment must embed an interactive plotly div with no Python kernel.
graph_objects/plotly.io ONLY (no kaleido). Test (tests/test_viz_plotly_html.py, importorskip('plotly')):
a 2-D spec exports an HTML string containing a plotly div + the CDN script reference (not a full
<html> doc when full_html=False), and writing to tmp_path works.`,
    adversarial: false,
  },
  {
    id: 'VIZ-THREEJS-EXPORT',
    worktree: `${ROOT}/.claude/worktrees/tsd-VIZ-THREEJS-EXPORT`,
    owns: 'src/tsdynamics/viz/render/threejs/__init__.py (NEW) ; src/tsdynamics/viz/render/threejs/_lower.py (NEW)',
    spec: `A 'threejs' data-export renderer producing a BufferGeometry-ready payload (PURE-PYTHON,
no extra dep). threejs/_lower.py: lower a PlotSpec's drawable layers into a JSON-able payload:
- per layer a 'geometry' with type ('line'|'points'|'surface' from LINE3D/SCATTER/SURFACE3D, and
  'line'/'points' for 2-D LINE/SCATTER lifted to z=0), FLAT Float32-style 'positions' (x,y,z
  interleaved as a plain list of floats), optional flat 'colors' (from the 'c' channel mapped through
  a colormap to rgb, or per-vertex), and for a line a 'indices' list of segment endpoints
  (0,1,1,2,2,3,...); a surface emits a triangulated index list over its grid.
- top-level 'metadata': schema_version, units/labels (from axes), 'bounds' (min/max per axis), and a
  'camera' (position/target/up) derived from the bounds (or spec.meta['camera']).
threejs/__init__.py: register(registry) a 'threejs' backend (RendererCapabilities data_export=True,
kinds=None, web_export=True) that returns the payload (a dict, or a RenderResult with the dict
payload + mimetype 'application/json'); a path= kwarg writes JSON. Registers UNCONDITIONALLY (pure
python). spec.render('threejs') returns the payload. A Lorenz 3-D attractor (LINE3D) AND a 2-D phase
portrait both export; the payload validates against the documented schema (document it in the module
docstring). Build the payload on the existing viz/export to_json where useful, but the geometry
lowering is new. Test (tests/test_viz_threejs.py, engine-free): a LINE3D spec -> a 'line' geometry
with positions length == 3*n_vertices and indices length == 2*(n-1); a 2-D phase portrait (SCATTER)
-> a 'points' geometry at z=0; schema_version + bounds + camera present; round-trips through json
(json.dumps loads back); flat positions are plain floats (Float32-ready), never nested arrays;
spec.render('threejs') returns the payload AND default spec.render() (no backend) still returns a
matplotlib Figure (threejs, being data_export, must NOT shadow the drawing default).`,
    adversarial: true,
  },
]

phase('Implement')
const implStage = (t) =>
  agent(
    `${COMMON}\n\n=== YOUR TICKET: ${t.id} ===\nWorktree: ${t.worktree}\nowns: ${t.owns}\n\nTASK:\n${t.spec}\n\n` +
      `Steps: cd ${t.worktree}; read the relevant existing backend (plotly/* or json.py + export.py + ` +
      `caps.py + render/__init__.py); implement within owns (+ your test file); run gates (pytest your ` +
      `test + tests/test_renderers_registry.py + tests/test_viz_fake_renderer.py, ruff, mypy --strict); ` +
      `commit 'feat: [${t.id}] <summary>'. Report worktree abspath, branch, files changed, owns-respected, ` +
      `gate results, per-acceptance self-check. DO NOT push/PR.`,
    { label: `impl:${t.id}`, phase: 'Implement', schema: IMPL_SCHEMA, effort: 'high' }
  )

const verifyStage = (impl, t) => {
  if (!impl || !impl.worktree) {
    return { id: t.id, worktree: t.worktree, branch: '(unknown)', verdict: 'FAIL', owns_respected: false, weakened_assertions_found: false, gates_green: false, pr_ready: false, findings: 'implementer returned nothing usable' }
  }
  return agent(
    `${COMMON}\n\n=== ADVERSARIALLY VERIFY ${t.id} ===\nWorktree: ${t.worktree}\nowns: ${t.owns}\nacceptance:\n${t.spec}\n\n` +
      `Try to REFUTE: (1) git -C ${t.worktree} diff stream/viz-foundation -- read the FULL diff; (2) every ` +
      `changed file within owns (a test file allowed) else FAIL; (3) grep removed assert/xfail/skip/loosened ` +
      `tol -> FAIL if any; (4) re-run gates (pytest + ruff + mypy --strict) all green; (5) check EACH ` +
      `acceptance bullet against the actual code; (6) import tsdynamics imports NO plotly AND NO matplotlib ` +
      `(fresh subprocess). ${t.adversarial ? 'ADVERSARIAL (threejs): PROVE flat positions are plain floats ' +
      '(length 3*n, never nested), line indices = 2*(n-1), the payload json.dumps-round-trips, AND default ' +
      'spec.render() still returns a matplotlib Figure (threejs data_export must not shadow the drawing ' +
      'default).' : ''} Return PASS only if every bullet holds; else FAIL with specifics.`,
    { label: `verify:${t.id}`, phase: 'Verify', schema: VERDICT_SCHEMA, effort: 'high' }
  )
}

const clean = []
const reports = await pipeline(TICKETS, implStage, verifyStage)
for (const r of reports) if (r) clean.push(r)
log(`PASS: ${clean.filter((r) => r.verdict === 'PASS' && r.pr_ready).map((r) => r.id).join(', ') || 'none'}`)
log(`FAIL: ${clean.filter((r) => !(r.verdict === 'PASS' && r.pr_ready)).map((r) => r.id).join(', ') || 'none'}`)
return { reports: clean }
