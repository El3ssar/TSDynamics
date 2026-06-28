export const meta = {
  name: 'mypy-core-annotate',
  description: 'WS-MYPY: fan out file-disjoint clusters to drive mypy --strict to 0 across the core library (annotations only, behavior-preserving)',
  phases: [{ title: 'Annotate', detail: 'one agent per file-disjoint cluster' }],
}

const WT = '/home/elessar/Projects/TSDynamics/.claude/worktrees/tsd-WS-MYPY'

const RULES = `
You are fixing \`mypy --strict\` errors in the TSDynamics core library so the
package type-checks cleanly. You work ONLY on the files listed for your cluster.

## How to see your errors and verify
Run from the current working directory (the main repo root — it has the venv):
\`\`\`
uv run mypy --strict --config-file ${WT}/pyproject.toml ${WT}/src/tsdynamics --cache-dir /tmp/mypycache-LABEL 2>&1 | grep -F -e FILE1 -e FILE2 ...
\`\`\`
(substitute your LABEL and your file paths, e.g. \`-e src/tsdynamics/families/base.py\`).
Iterate: fix → re-run that command → repeat until it prints NOTHING for your files.
If mypy aborts with a *syntax/parse* error in a file you were NOT assigned, another
worker is mid-edit; wait a moment and re-run. NEVER edit a file outside your list.

## ABSOLUTE RULES
1. **Type annotations / typing-only changes. NEVER change runtime behavior.**
   Allowed: add parameter/return/variable annotations; add \`typing\`/\`numpy.typing\`
   imports (often under \`if TYPE_CHECKING:\`); add \`cast(...)\`; add a narrowing
   \`assert x is not None\` (or hoist to a local) where a value is provably non-None
   at runtime; add a precise \`# type: ignore[code]  # short reason\` ONLY as a last
   resort when a correct type is genuinely infeasible. Prefer real types over ignores.
2. Files almost all start with \`from __future__ import annotations\`, so annotations
   are strings and are NEVER evaluated at runtime — a \`TYPE_CHECKING\`-only import is
   the safe way to reference a type that would cause an import cycle at runtime.
3. Do NOT touch: pyproject.toml, any test, CLAUDE.md, CONTRIBUTING, .github/**, or
   any file outside your cluster's list. Do NOT reformat unrelated lines. Keep ruff
   happy (line length 100). Match surrounding style.
4. Behavior preservation is paramount: if a "fix" would require changing logic,
   prefer a \`cast\`/narrowing/ignore that leaves the runtime path byte-identical, and
   note it in your return. The ONE sanctioned tiny behavior touch is the
   \`discrete.py iterate\` missing-return below (a defensive raise on an unreachable
   path) — only if discrete.py is in your cluster.

## COMMON PATTERNS (apply the right type, do not just blanket-ignore)
- **no-any-return**: a function annotated \`-> T\` returns an \`Any\` expression (often a
  NumPy call). Wrap with \`cast(T, expr)\` or coerce (e.g. \`float(x)\`, \`np.asarray(x)\`),
  whichever is byte-identical. For \`-> float\` returning a numpy scalar, \`float(...)\`.
- **type-arg** (\`Missing type arguments for generic type "X"\`): parametrize it —
  \`dict\` → \`dict[str, float]\` (or the real value type), \`tuple\` → \`tuple[T, ...]\`,
  \`list\` → \`list[T]\`, \`Callable\` → \`Callable[..., T]\`, bare \`np.ndarray\` is fine but
  \`npt.NDArray[np.float64]\` is preferred where a dtype is known.
- **no-untyped-def**: annotate the params + return. For \`ic=None\` style, use
  \`Any | None\` if the real type is broad. Use \`-> None\` for procedures.
- **\`"object" has no attribute "dim"/"_step"/"params"/"_rhs_numeric"/"jacobian"/"resolve_ic"/"hi"\`**:
  a helper's \`system\` parameter is typed \`object\` (or untyped). Give it the real
  family type via a TYPE_CHECKING import from \`tsdynamics.families\`:
  map helpers (\`_step\`/\`_jacobian\`) → \`DiscreteMap\`; flow helpers
  (\`_rhs_numeric\`/\`jacobian\`) → \`ContinuousSystem\`; generic (\`dim\`/\`params\`/
  \`resolve_ic\`) → \`SystemBase\`. Read the docstring to pick. NEVER import these at
  runtime from analysis/** (cycle) — use \`if TYPE_CHECKING:\`.
- **\`"dict[Any, Any]" has no attribute "as_dict"/"as_tuple"/"param_hash"\`**:
  \`self.params\` is a \`ParamSet\` (defined in \`tsdynamics.families.base\`), not a plain
  dict. Fix the annotation of the \`params\` attribute / local so mypy sees ParamSet
  (a TYPE_CHECKING import + an annotation, or \`cast(ParamSet, ...)\`).
- **\`Item "None" of "ndarray | None" has no attribute "copy"/"T"\`** (union-attr/index):
  the value (e.g. \`self._state_now\`) is non-None after the implicit \`reinit()\`. Add
  \`assert self._state_now is not None\` right before the use (runtime-safe), or bind a
  local \`state = self._state_now; assert state is not None\`. Do not change the logic.
- **\`int(numpy_scalar)\` arg-type**: use \`int(np.asarray(x).item())\` or \`int(x)  # type: ignore[arg-type]\`
  — pick the byte-identical one (\`.item()\` is safest).
- **assignment list-vs-ndarray**: a variable reused for both a list and an ndarray.
  Annotate it with the union at first binding (e.g. \`vals: list[float] | np.ndarray\`)
  or rename — whichever preserves behavior.

## TRICKY SITES (only relevant if the named file is in YOUR cluster)
- \`families/discrete.py\` \`iterate()\` **[return] Missing return statement**: the
  \`for attempt in range(max_retries):\` loop returns on success or raises on the last
  attempt, but mypy can't prove the loop is non-empty. Add, AFTER the for-loop, a
  final \`raise RuntimeError(f"{type(self).__name__}.iterate exhausted {max_retries} "
  f"retries without a finite trajectory.")\`. This is unreachable for max_retries>=1
  and converts the degenerate max_retries<=0 silent None-return into a clear error.
  Also fix the same file's \`params.as_tuple\` (ParamSet) and \`_state_now\` (.copy)
  per the patterns above.
- \`families/continuous.py\` and \`families/delay.py\` **co_code [union-attr]**: the line
  \`repr(getattr(fn, "__code__", fn).co_code)\` — narrow to
  \`code = getattr(fn, "__code__", None)\` then
  \`repr(code.co_code) if code is not None else repr(fn)\`. Byte-identical when
  \`__code__\` exists (always, for Python functions).
- \`analysis/basins/continuation.py\` **[no-redef] "mapping"**: the early-return branch
  \`if not prev_global:\` binds \`mapping\` then returns; inline that return
  (\`return {lid: next_global + i for i, lid in enumerate(current)}, next_global + len(current)\`)
  so the later \`mapping: dict[int, int] = {}\` is the sole binding.
- \`analysis/basins/basins.py\` **[assignment] None vs Grid**: a var annotated \`Grid\`
  is later set to None. Widen its annotation to \`Grid | None\` (do not change logic).
- \`analysis/entropy/multiscale.py\` & \`analysis/surrogate/hypothesis.py\`
  **Callable default [assignment]**: an \`entropy_fn\`/stat default (e.g. \`sample_entropy\`)
  now returns a result object (ScalarResult), not \`float\`. Widen the parameter
  annotation to \`Callable[..., float]\` → \`Callable[..., Any]\` (or to the result type).
- \`derived/_base.py\` **"DerivedSystem" has no attribute "trajectory"**: \`DerivedSystem\`
  is the wrapper base; subclasses define \`trajectory\`. Declare it on the base as an
  abstract/typed stub: add (with the other methods) \`def trajectory(self, *args: Any,
  **kwargs: Any) -> Trajectory: ...\` decorated \`@abstractmethod\` (import abc) — or, if
  abstractmethod risks instantiation tests, a plain stub raising NotImplementedError.
  Read the class first; prefer the minimal change that keeps every subclass valid.

Return a structured summary of exactly what you changed.
`

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    label: { type: 'string' },
    cleared: { type: 'boolean', description: 'true if mypy prints nothing for your files now' },
    filesTouched: { type: 'array', items: { type: 'string' } },
    residual: {
      type: 'array',
      items: { type: 'string' },
      description: 'any mypy errors in your files you could NOT clear, verbatim, with a one-line reason',
    },
    behaviorNotes: {
      type: 'array',
      items: { type: 'string' },
      description: 'any change that was not a pure annotation/cast/assert (should be empty except the sanctioned discrete.py raise)',
    },
  },
  required: ['label', 'cleared', 'filesTouched', 'residual', 'behaviorNotes'],
}

const CLUSTERS = [
  { label: 'fam-continuous', files: ['families/continuous.py', 'families/protocol.py'] },
  { label: 'fam-base-discrete', files: ['families/base.py', 'families/discrete.py'] },
  { label: 'fam-stoch-delay', files: ['families/stochastic.py', 'families/delay.py', 'families/wrapped.py', 'families/_dde_lyapunov.py'] },
  { label: 'derived', files: ['derived/poincare.py', 'derived/_base.py', 'derived/tangent.py', 'derived/stroboscopic.py', 'derived/projected.py', 'derived/ensemble.py', 'derived/_variational.py'] },
  { label: 'engine-data-misc', files: ['engine/run.py', 'engine/compile.py', 'data/trajectory.py', 'data/sampling.py', 'utils/sagitta_dt.py', 'registry.py', 'viz/spec.py', '__init__.py'] },
  { label: 'fixedpoints', files: ['analysis/fixedpoints/_common.py', 'analysis/fixedpoints/fixed.py', 'analysis/fixedpoints/periodic.py', 'analysis/fixedpoints/__init__.py'] },
  { label: 'result-chaos', files: ['analysis/_result.py', 'analysis/chaos/_common.py', 'analysis/chaos/__init__.py', 'analysis/chaos/gali.py'] },
  { label: 'basins-recur-orbits', files: ['analysis/basins/_common.py', 'analysis/basins/attractors.py', 'analysis/basins/continuation.py', 'analysis/basins/basins.py', 'analysis/basins/metrics.py', 'analysis/recurrence/_common.py', 'analysis/recurrence/matrix.py', 'analysis/orbits/return_map.py', 'analysis/orbits/poincare.py', 'analysis/orbits/orbit_diagram.py'] },
  { label: 'analysis-tail', files: ['analysis/surrogate/generators.py', 'analysis/surrogate/statistics.py', 'analysis/surrogate/hypothesis.py', 'analysis/surrogate/_common.py', 'analysis/entropy/core.py', 'analysis/entropy/multiscale.py', 'analysis/entropy/lz.py', 'analysis/entropy/sample.py', 'analysis/dimensions/_common.py', 'analysis/dimensions/_scaling.py', 'analysis/dimensions/generalized.py', 'analysis/dimensions/__init__.py', 'analysis/embedding/embed.py', 'analysis/lyapunov/from_data.py', 'analysis/lyapunov/__init__.py', 'analysis/transforms/__init__.py'] },
]

phase('Annotate')

const results = await parallel(
  CLUSTERS.map((c) => () => {
    const fileList = c.files.map((f) => `${WT}/src/tsdynamics/${f}`).join('\n  ')
    const grepArgs = c.files.map((f) => `-e src/tsdynamics/${f}`).join(' ')
    const prompt = `${RULES}

## YOUR CLUSTER: ${c.label}
Your files (edit ONLY these, absolute paths):
  ${fileList}

Verify command (run repeatedly until it prints nothing):
  uv run mypy --strict --config-file ${WT}/pyproject.toml ${WT}/src/tsdynamics --cache-dir /tmp/mypycache-${c.label} 2>&1 | grep -F ${grepArgs}

Begin. Drive your files to zero mypy --strict errors, annotation-only.`
    return agent(prompt, { label: `mypy:${c.label}`, phase: 'Annotate', schema: SCHEMA })
  })
)

return { results: results.filter(Boolean) }
