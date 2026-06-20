# The naming glossary

> **One word per concept.** This page is the *frozen* vocabulary for the public
> API: the single canonical spelling — and unit — for every recurring idea that
> appears in more than one function signature. It is the contract that the
> calling-convention unification (`WS-CONV`) implements and that the CI naming
> gate (`WS-NAMEGATE`) enforces. A build fails if a public signature reintroduces
> a banned spelling for a concept defined here.

The analysis layer grew one stream at a time, each authored independently, so the
*same* concept ended up spelled up to **five** ways — `transient` / `burn_in` /
`n_cut`; `final_time` / `steps` / `n` / `n_rescale`; `delay` / `tau` / `lag` /
`max_lag`; and a first argument named nine different things for the one role "the
thing you analyse" (`sys`, `system`, `sys_or_traj`, `source`, `map_sys`,
`observable`, `data`, `x`, `series`). Each function was locally reasonable; the
*aggregate* was obscure. `seed` and `ic` — already consistent across the whole
surface — prove a single convention is achievable. This glossary makes it the law
for everything else.

**Scope.** The glossary and the gate cover the whole public callable surface —
both `tsdynamics.analysis` (`registry.analyses`) **and** `tsdynamics.transforms`
(`registry.transforms`). It is a v4 change: spellings are corrected outright (the
v3→v4 token map in §6 lists every rename; the `_migration` shim raises a precise
"renamed to *X*" error for one release). **Every ban below targets a parameter
*name* in a public signature** — never an attribute (`system.dim` is untouched), a
return-object field, or a value. Domain-specific parameters that belong to a
single function (`radii`, `metric`, `q`, `eps`, `n_scales`, …) are out of scope —
each function owns those. The glossary fixes only the **cross-cutting**
vocabulary, plus the handful of homonyms (§5) the gate must whitelist by exact
name.

---

## 1. The first positional argument — three roles, two names

Every analysis takes its primary input as the **first positional argument**, with
exactly one canonical name per kind:

| Input kind | Canonical name | Accepts |
|---|---|---|
| A dynamical system (flow / map / derived wrapper) | **`system`** | any `System` (`ContinuousSystem`, `DiscreteMap`, `PoincareMap`, …) |
| A measured series / point set | **`data`** | `Trajectory`, `ndarray` shaped `(N,)` or `(N, dim)` |
| A prior analysis result | *named by kind* (whitelisted — §5) | the upstream result object |

**Choosing `system` vs `data`.** A function whose job is to *integrate or iterate
the dynamics itself* takes `system` (it produces its own trajectory internally —
`gali`, `expansion_entropy`, `fixed_points`, `find_attractors`). A function whose
job is to *operate on an already-measured signal* takes `data` (`recurrence_matrix`,
`correlation_dimension`, `permutation_entropy`, the transforms).

**Dual-input functions take `system` as the first argument** and coerce / accept a
measured series through internal handling or a `data=` overload — they never invent
a third name. The three current offenders are pinned:

| Function | v4 first arg | v3 first arg |
|---|---|---|
| `zero_one_test` | **`system`** (integrates internally like its sibling `gali`; series via coercion) | `observable` |
| `poincare_section` | **`system`** (a `Trajectory` is accepted and coerced) | `sys_or_traj` |
| `return_map` | **`system`** (a `Trajectory` / 1-D series is accepted and coerced) | `source` |

**Banned first-arg spellings** (all → `system` or `data`):

| Banned | Canonical | Reason |
|---|---|---|
| `sys` | `system` | abbreviation |
| `sys_or_traj` | `system` | the dual nature is handled by coercion, not the name |
| `map_sys` | `system` | a map *is* a `System` |
| `observable` (as 1st arg) | `system` / `data` | the *quantity observed* is the `component` keyword, not the input |
| `source` | `system` / `data` | vague |
| `x` | `data` | the entropy/transforms convention; unify on `data` |
| `series` | `data` | synonym |

---

## 2. The canonical keyword glossary

One spelling, one unit, everywhere the concept appears. The **Bans** column lists
the spellings the naming gate rejects for that concept (subject to the exact
homonym whitelist in §5). Bans marked *†* do not occur in v3 today — they are
pre-emptive, kept so the concept can never drift into them.

| Concept | Canonical | Type / unit | Bans |
|---|---|---|---|
| Initial condition | **`ic`** | array-like, length = `system.dim` | `x0`†, `initial`†, `u0`†, `y0`† |
| RNG seed | **`seed`** | `int \| None` | `random_state`†, `rng`† |
| Discard-transient amount | **`transient`** | **time** (flows) / **steps** (maps) — see §3 | `burn_in`, `n_transient`†, `warmup`† |
| Integration horizon (flows) | **`final_time`** | `float`, in the system's time units | `T`†, `t_final`†, `tmax`† |
| Iteration / sampling horizon | **`n`** | `int` — see §3 | `steps`, `n_rescale` |
| Step size | **`dt`** | `float`, time units | `h`† |
| Observed component | **`component`** | `int` index or `str` name (`None` = all) | `components`, `observable`, `coord`†, `col`† |
| Embedding dimension | **`dimension`** | `int` | `m`, `emb_dim`†, `dim` (as a *parameter*) |
| Embedding delay / lag | **`delay`** | `int`, in samples | `tau`, `lag`, `max_lag` (→ `max_delay`, §5) |
| Theiler window | **`theiler`** | `int`, in samples | `theiler_window`, `w`† |
| Nearest-neighbour count | **`n_neighbors`** | `int` | `min_neighbors`, `num_neighbors`† |
| Spatial region | **`region`** | `Box` / `Ball` / `Grid`, or `region([(lo, hi, n), …])` | `grid`, `box`, `domain`†, `bounds`† |
| Algorithm / kernel selector | **`method`** | `str` | `kind`, `mode`†, `estimator`†, `scheme`† |

`seed` and `ic` are already universal — they are listed so the gate treats them as
locked, not because they need fixing. `method` is **always allowed** (it is the
canonical variant/kernel selector — solver kernels on `integrate`/`lyapunov_spectrum`,
algorithm variants on `fixed_points`/`optimal_delay`/`power_spectral_density`); it
is never a banned token and needs no per-site whitelist.

---

## 3. Unit rules (documentation-only — not gate-enforceable)

A signature gate sees parameter *names*, not their meaning, so the rules in this
section are **documentation-only**: they bind authors and reviewers, not the CI
naming gate. (The gate-enforceable rules are in §7.)

**`transient` follows the unit of the call's horizon.**

- Paired with `final_time` (a flow) → `transient` is a **time** (same units as
  `final_time`): `lyapunov_spectrum(lorenz, final_time=300, transient=50)` discards
  the first 50 time units.
- Paired with `n` (a map / discrete view) → `transient` is a count of **steps**:
  `bifurcation_diagram(logistic, "r", vals, n=200, transient=500)` discards the
  first 500 iterations.
- A few flow estimators parametrise their run by a *count of rescaling cycles*
  (canonical `n`, replacing v3's `n_rescale`) rather than `final_time`; there
  `transient` is in **protocol steps**, following the count horizon (e.g.
  `max_lyapunov`).
- `transient` **never** means "section crossings." A Poincaré / return view that
  wants to discard the first few section hits uses the dedicated `skip_crossings`
  keyword (§5), so the unit of `transient` stays unambiguous.

**`final_time` (flows) vs `n` (maps).** A flow's run length is a time →
`final_time`. A map's / discrete view's run length is a count of iterations → `n`.
Analyses that serve *both* families expose both horizons and pick by the input
(`gali`, `expansion_entropy` carry `final_time` *and* `steps`→`n`). The unified
`lyapunov_spectrum` follows its documented signature
(`lyapunov_spectrum(system, *, k=None, final_time=None, transient=None, dt=None,
ic=None, method=None, seed=None)`) — `final_time` is its headline horizon.

**`n` vs `n_<thing>`.** Bare `n` is the *primary* run length / number of produced
units — iterations of a map, points recorded per orbit, surrogate realisations to
draw. A prefixed `n_<thing>` is a *count of that specific thing* and is a distinct,
function-owned parameter: `n_samples`, `n_seeds`, `n_radii`, `n_scales`,
`n_internal`, `n_c`, `n_ref`, `n_ks`. These are **not** the iteration horizon and
are not renamed to `n` (§7 lists them so the gate does not mistake them for bare-`n`
violators).

**`dt` is a time step.** The sampling / integration step. A window *stride*
(`windowed_rqa`) is a different concept and is **not** `dt` (§5).

---

## 4. Verbs — one intent, one word

| Intent | Canonical | Permanent aliases |
|---|---|---|
| Produce a trajectory (any family) | **`system.run(...)`** | `.integrate` (flows), `.iterate` (maps), `.trajectory` |

`run` dispatches on `is_discrete`, so `Lorenz().run(final_time=100, dt=0.01)` and
`Henon().run(n=5000)` are the one verb a newcomer learns. `.integrate` /
`.iterate` / `.trajectory` survive permanently as thin aliases (no breakage); docs
and examples standardise on `run`. The codemod rewrites only `.integrate` /
`.iterate` → `.run` — `.trajectory` is kept as-is, not rewritten.

---

## 5. Homonym carve-outs (the same token, a *different* concept)

These `(function, parameter)` pairs are **deliberately allowed** because each names
a genuinely different concept there. The naming gate whitelists **exactly** these
pairs; the same token used for a *banned* concept anywhere else still fails.

| Token | Allowed at (exact) | Means | Why it is not the banned concept |
|---|---|---|---|
| **`k`** | `gali.k` | GALI **order** — number of deviation vectors evolved (`2 ≤ k ≤ dim`) | an *order*, not a neighbour count |
| **`k`** | `lyapunov_spectrum.k` | number of **Lyapunov exponents** to compute (was `n_exp`; *may exceed `dim`* for DDEs — a function-space tangent) | a count of *exponents*, not neighbours |
| **`k_max`** | `lyapunov_from_data.k_max` | length of the divergence / stretching curve `S(k)`, evolved over `k = 0 … k_max` | an *abscissa horizon* of the scaling curve, not a neighbour count |
| **`step`** | `windowed_rqa.step` | window **stride** in samples | a stride between windows, not the integration time step `dt` |
| **`horizon`** | `nonlinear_prediction_error.horizon` | prediction **lead-time** in samples | how far ahead to predict, not the run-length horizon (`n` / `final_time`) |
| **`max_steps`** | `find_attractors`, `basins_of_attraction`, `continuation`, `basin_fractions` | integration **safety cap** — max steps before a trajectory is declared lost / non-recurrent | a divergence/abort bound, not the run length `n` |
| **`max_delay`** | `optimal_delay`, `mutual_information`, `estimate_period`, `autocorrelation` | the **search ceiling** for a delay scan (supersedes `max_lag`) | a bound on the delay search, distinct from a single embedding `delay`; the `lag`/`tau` tokens stay banned |
| **`fs`** | the transforms / spectral functions (`power_spectral_density`, `spectral_entropy`, `spectral_centroid`, `dominant_frequency`, `butter_filter`, `extract_features`) | sampling **frequency** (Hz) — accepted *alongside* `dt` (`fs = 1/dt`, mutually exclusive) | a signal-processing convenience, not a competing spelling of the time step |

**New tokens WS-CONV introduces** (not v3 renames — they do not exist yet, so a
reviewer should not expect to find them in v3 code):

- **`skip_crossings`** (`poincare_section`, `return_map`): number of section
  crossings to discard before recording. It replaces the crossing-counting *use* of
  `transient` on those two views so that `transient` keeps a single unit (§3).
- **`region([(lo, hi, n), …])`** in `tsdynamics.data`: a builder for the `region`
  argument, replacing the three-parallel-array `Grid(lo, hi, counts)` constructor.

**Prior-result first arguments** (the third role in §1). These are named by the
*kind* of result they consume and are intentionally **not** unified onto a single
spelling; the gate whitelists each exact `(function, first-arg)` pair:

| Function | First arg | Consumes |
|---|---|---|
| `kaplan_yorke_dimension` | `spectrum` | a Lyapunov spectrum (array / `LyapunovSpectrum`) |
| `uncertainty_exponent`, `wada_property`, `basin_entropy` | `basins` | a `BasinsResult` |
| `resilience`, `tipping_points` | `result` | a `BasinsResult` / `ContinuationResult` |

**Domain-owned, out of scope** (not renamed, not banned — each function owns them):
`zero_one_test.n_cut` (the mean-square-displacement *lag ceiling*, **not** a
transient — default `N // 10`), and the `n_<thing>` counts in §3.

---

## 6. v3 → v4 token map (quick reference)

Mechanical renames the migration shim and codemod (`WS-MIGRATE`) cover. The old
name raises a precise "renamed to *X* in v4.0" error for one release.

| v3 | v4 | Scope |
|---|---|---|
| `sys`, `sys_or_traj`, `map_sys`, `source`, `observable` (1st arg) | `system` | `lyapunov_spectrum`, `max_lyapunov`, `orbit_diagram`, `periodic_orbits`, `poincare_section`, `return_map`, `zero_one_test` |
| `x`, `series` (1st arg) | `data` | entropy family, transforms, `lyapunov_from_data` |
| `observable` (kwarg) | `component` | `return_map` |
| `components` | `component` | `orbit_diagram` |
| `burn_in` | `transient` | `periodic_orbit` |
| `steps`, `n_rescale` | `n` *(or `final_time` for flows)* | `gali`, `expansion_entropy`, `max_lyapunov`, `poincare_section`, `return_map` |
| `tau`, `lag` | `delay` | entropy family, `time_reversal_asymmetry`, `nonlinear_prediction_error` |
| `max_lag` | `max_delay` | `optimal_delay`, `mutual_information`, `estimate_period`, `autocorrelation` |
| `m`, `emb_dim` | `dimension` | entropy family, `lyapunov_from_data`, `nonlinear_prediction_error` |
| `theiler_window` | `theiler` | dimensions / recurrence family |
| `min_neighbors` | `n_neighbors` | `lyapunov_from_data` |
| `kind` (variant selector) | `method` | `return_map` (`max`/`min`/`poincare`), `detrend` (`linear`/`constant`) |
| `grid`, `box` (region arg) | `region` | `basins_of_attraction`, `fixed_points`, `periodic_orbits` |
| `n_exp` | `k` | `lyapunov_spectrum` |
| `.integrate` / `.iterate` | `.run` | every family (`.trajectory` kept, not rewritten) |
| `orbit_diagram` | `bifurcation_diagram` | top-level (alias kept) |

> `n_cut` (`zero_one_test`) is **not** in this table: it is a domain-owned lag
> ceiling (§5), not a transient — it is neither renamed nor banned.

---

## 7. For the gate (`WS-NAMEGATE`) and signature (`WS-CONV`) authors

**Gate-enforceable rules** (decidable from `inspect.signature` alone):

1. **First argument** of every public callable is `system` or `data`, *or* the
   `(function, first-arg)` pair is in the prior-result whitelist (§5).
2. **No parameter name** is a banned spelling from §2, *unless* the
   `(function, parameter)` pair is in the homonym whitelist (§5). Build the
   banned→canonical map straight from the §2 Bans column; build the whitelist from
   the §5 exact-pair tables. `method` is never banned (no whitelist needed).
3. The `n_<thing>` domain counts (`n_samples`, `n_seeds`, `n_radii`, `n_scales`,
   `n_internal`, `n_c`, `n_ref`, `n_ks`) are **not** `n` violations — they are
   distinct names, so a prefix check on bare `n` already passes them; they are
   listed here only so the gate author does not special-case them by mistake.
4. The gate sweeps **both** `registry.analyses` and `registry.transforms`.

**Documentation-only rules** (a signature gate cannot see units or semantics, so
do **not** try to encode these — they are enforced in review, not CI): the
`transient`-follows-horizon rule, the `final_time`(flows) / `n`(maps) split, and
the `n` vs `n_<thing>` meaning (all §3).

**For `WS-CONV`:** the §5 whitelist references names that do not exist in v3 yet
(`lyapunov_spectrum.k`, `skip_crossings`) — these are the *targets* WS-CONV
creates; the gate whitelists the new names, so the entries are correct despite
being absent from the v3 signatures.

---

## 8. Adding a new analysis — the naming checklist

When you write a new public analysis, before it can merge:

1. **First argument** is `system` (you integrate/iterate it) or `data` (you
   consume a measured series). Nothing else.
2. **Every recurring concept** in §2 uses its canonical spelling and documented
   unit. Function-specific parameters are free to be named whatever reads best.
3. **A new homonym?** If you genuinely need a §2 token for a *different* concept,
   add a row to §5 (an exact `(function, parameter)` pair) and the matching
   whitelist entry to the naming gate — don't silently reuse it.
4. **`transient` and the horizon agree on units** (§3): `final_time`+`transient`
   in time for a flow; `n`+`transient` in steps for a map.
5. **The naming gate** (`tests/test_polish_standards.py`, registry-driven) sweeps
   your function automatically — no test edits needed. If it fails, it names the
   offending parameter and the canonical spelling.

The goal is simple: a user who has learned one analysis can guess the signature
of the next.
