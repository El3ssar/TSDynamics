# The naming glossary

Use consistent names and units when adding public analyses. This guide describes
the current conventions and the tests that check signatures. Function-specific
parameters remain documented in each function's API reference.

## 1. The first positional argument — three roles, two names

Put the primary input first:

| Input | Name | Examples |
|---|---|---|
| A system or a function accepting both systems and data | `system` | `fixed_points(system)`, `return_map(system)` |
| Recorded states or a signal | `data` | `correlation_dimension(data)`, `embed(data)` |
| An earlier analysis result | Name the result | `kaplan_yorke_dimension(spectrum)`, `tipping_points(result)` |

A name describes the input role, not a guarantee that every system family is
supported. Declare supported subjects when registering an analysis, and document
any dimension, autonomy or differentiability requirements. Existing sampling
and binary-distance functions have specific names recorded in the signature tests.

## 2. The canonical keyword glossary

| Concept | Name | Meaning / unit |
|---|---|---|
| Initial condition | `ic` | State vector of length `system.dim` |
| Random seed | `seed` | Integer or `None` |
| Discarded transient | `transient` | Time for flows; iterations for maps |
| Flow horizon | `final_time` | Time in the model's units; check whether a function takes an endpoint or elapsed duration |
| Primary iteration count | `n` | Count used by the analysis; `run` uses `steps` for maps |
| Sampling or numerical step | `dt` | Time units; its numerical role depends on the solver |
| Component selection | `components` | Name, index or selection, as accepted by the function |
| Embedding dimension | `dimension` | Number of delay coordinates |
| Embedding delay | `delay` | Samples |
| Theiler window | `theiler` | Exclusion distance in sample indices |
| Neighbour count | `n_neighbors` | Count |
| Spatial search region | `region` | Per-component bounds or an accepted region object |
| Estimation algorithm | `method` | For example `method="mi"` or `method="interval"` |
| Integration kernel | `solver` | For example `solver="dop853"` |

Use `components` even when a particular analysis accepts only one selected
channel. Its signature and validation should make that restriction clear.
Avoid introducing synonyms such as `x0`, `random_state`, `burn_in`, `tau` or
`emb_dim` for the shared concepts above. The full set of checked spellings is in
`tests/test_polish_standards.py`.

## 3. Unit rules (documentation-only — not gate-enforceable)

State units in the docstring; signature tests cannot verify them.

- For a flow, `transient` is a duration in the same units as `final_time`.
- For a map, `transient` counts discarded iterations.
- `skip_crossings` counts discarded section crossings. It is separate from a
  time or iteration transient.
- `dt` can mean output spacing, numerical step or both. Adaptive ODE solvers
  separate internal steps from output sampling; fixed-step and stochastic
  schemes use `dt` as the numerical step too. DDEs have additional history
  resolution constraints.
- `delay`, `theiler`, and window strides count samples, not physical time.
- Counts such as `n_samples`, `n_seeds`, `n_scales` and `n_c` name distinct work
  quantities; they need not be shortened to `n`.
- Descriptive counts such as `points_per_value` and `crossings` identify the
  output being requested. Preserve their meaning across overloads.

## 4. Verbs — one intent, one word

| Task | Public form |
|---|---|
| Simulate a system | `system.run(...)` |
| Advance a live cursor | `system.step(...)` |
| Read the system family | `system.family` |
| Derive a section or stroboscopic view | `system.poincare(...)` |
| Compute an analysis | `ts.analysis.name(subject, ...)` |

A flow's `run` takes a time horizon; a map's takes `steps`. Old
`.integrate`, `.iterate` and `.trajectory` methods were removed in v6. Their
migration messages point to `run`.

## 5. Homonym carve-outs (the same token, a *different* concept)

Some names have a distinct mathematical role in a particular function:

| Name | Function | Meaning |
|---|---|---|
| `k` | `gali` | Number of deviation vectors; `2 <= k <= dim` |
| `k` | `lyapunov_spectrum` | Number of exponents; DDE tangent spaces can require more than the instantaneous state dimension |
| `k_max` | `lyapunov_from_data` | Length of the stretching curve |
| `step` | `windowed_rqa` | Window stride in samples |
| `max_steps` | Basin and attractor searches | Per-trajectory search limit |
| `max_delay` | Delay selection, autocorrelation and period estimation | Search ceiling in samples |
| `grid` | Planar field analyses | Grid resolution, rather than a search-region object |
| `steps` | `streamlines` | Integration-length cap for a streamline |
| `kind` | `return_map` | Maxima, minima or section-crossing construction |
| `n_cut` | `zero_one_test` | Maximum displacement lag, rather than a discarded transient |

Prior-result inputs use names such as `spectrum`, `basins` and `result`, as
listed in their signatures. The signature tests record these exceptions by
function and parameter, so an exception does not broaden the convention for
unrelated functions.

## 6. v3 → v4 token map (quick reference)

This heading is retained for older links. For current migration guidance,
consult [Top level](../reference/top-level.md#the-error-ladder-how-you-find-a-name-that-moved)
and the error raised by the installed version. Several early renames changed
again in v6:

| Older spelling | Current spelling | Scope |
|---|---|---|
| `.integrate`, `.iterate`, `.trajectory` | `.run` | Model simulation |
| `component`, `observable` | `components` | Observed channel selection |
| `method` on `run` | `solver` | Numerical integration kernel |
| `method` on `return_map` | `kind` | Type of recurring observable |
| `tau`, `lag` | `delay` | Delay coordinates |
| `m`, `emb_dim` | `dimension` | Delay-coordinate count |
| `max_lag` | `max_delay` | Search ceiling |
| `theiler_window` | `theiler` | Temporal pair exclusion |
| `min_neighbors` | `n_neighbors` | Neighbour count |
| `n_exp` | `k` | Number of Lyapunov exponents |

These are migration examples, not interchangeable aliases. Check the current
signature before updating an older script.

## 7. For the gate (`WS-NAMEGATE`) and signature (`WS-CONV`) authors

`tests/test_polish_standards.py` checks registered analysis signatures for the
first-input names, shared parameter spellings and explicit exceptions described
above. `tests/test_calling_convention.py` covers a focused subset. Add or update
an exact exception when a name represents a different concept; do not exempt an
entire function to avoid choosing a consistent name.

Tests can check a spelling, but reviews and numerical tests must check units,
accepted families, array orientation and the meaning of a horizon.

## 8. Adding a new analysis — the naming checklist

1. Name the first argument for its input role.
2. Reuse the shared names and state their units.
3. Give function-specific parameters descriptive names; add an exact exception
   if one conflicts with a shared spelling for a different concept.
4. Keep time, iteration, sample and crossing counts distinct.
5. Register supported inputs and add numerical tests for the new behavior.
6. Run the signature checks and the analysis's focused tests.
