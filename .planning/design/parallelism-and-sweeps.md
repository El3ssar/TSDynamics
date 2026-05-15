# Parallelism & parameter sweeps

Stub. Full design lands when R2 / M3 are being built.

## Things this doc needs to cover (when it grows up)

- The `sweep(system, param_grid, observable, ...)` API surface — exact signature,
  return type, caching semantics.
- How `observable` is specified. Probably a callable `(traj) -> float | ndarray`
  plus a small registry of named observables (`"lyapunov_max"`, `"poincare_count"`,
  `"period"`, …).
- How parameters in the grid are passed to the system. Use `system.with_params(...)`
  per cell; do not mutate the input system.
- Caching: key derived from `(class, structural-hash, params-hash,
  observable-hash, t-grid-hash)`. Storage on disk under `~/.cache/tsdynamics/sweeps/`.
  Format: probably `zarr` — chunked, parallel-safe, naturally maps to grids.
- Rust kernel boundary: outer loop in Rust (`rayon`), each cell calls back into the
  Python `observable` (with GIL) only for evaluation. The integration step itself
  happens in Rust (post-N2) or in JiTCODE-via-cffi (pre-N2).
- Progress reporting: `tqdm`-compatible callback, optional, falls back to silent.
- Cancellation: Ctrl-C must stop sweeps cleanly without corrupting the disk cache.

## Open questions (revisit during R2)

- One sweep type or many? Linear / log / logit / custom-list grids?
- Should observables be allowed to be vector-valued and return ragged results
  (e.g., Poincaré crossings — variable count per cell)? Probably yes; store as a
  `list[ndarray]` rather than a regular array.
- Multi-parameter sweeps (codim-N grids): N-d cartesian product, or N-d
  user-supplied list of param dicts? Probably both, with the latter being the
  general fallback.
