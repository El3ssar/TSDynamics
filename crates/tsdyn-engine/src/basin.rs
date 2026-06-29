//! The basin / attractor recurrence finite-state machine, in Rust (stream
//! `perf/basin-march`).
//!
//! [`basin_march`] runs the **entire** per-initial-condition recurrence FSM of the
//! Python `_AttractorMapper` (stepping + cell-binning + the shared-label early-out)
//! inside one engine call, so a `basins_of_attraction` image is computed without
//! the ~10^6–10^7 per-`dt` Python→FFI round-trips and the per-step Python FSM
//! overhead the released loop pays. It drives the *same* sequential algorithm
//! over the *same* engine stepper: each cell-check advances a flow by one `dt` (an
//! [`integrate_grid`] over the two-node grid `[t, t+dt]`, byte-for-byte the
//! per-`dt` integration the Python `system.step(dt)` runs) or iterates a map once
//! ([`map_step_once`](crate::map::map_step_once)).
//!
//! For **flows** this is **bit-identical** to the Python mapper: both advance the
//! same engine stepper, so the label image and the located attractor set match
//! byte-for-byte. For **maps** the Python mapper iterates the pure-Python `_step`
//! while this kernel runs the lowered IR; the two differ by ULPs, so a
//! boundary-straddling iterate can bin into a neighbouring cell. In practice the
//! map label image is still **empirically byte-identical** across the catalogue,
//! but the guarantee is only *same-attractor* — a chaotic map's located cells may
//! sit a few cells apart.
//!
//! # Layering
//!
//! The kernel is agnostic about *which* evaluator (interpreter vs JIT) and *which*
//! solver kernel it drives — exactly like [`integrate`](crate::integrate) — so it
//! takes a `&dyn Evaluator` and a `solver_factory` closure. The binding layer
//! (`tsdyn-core`) builds the evaluator, resolves the method, and hands both here.
//! That keeps the cranelift toolchain edge in `tsdyn-core` (the engine does not
//! depend on `tsdyn-jit`).
//!
//! # Why sequential (not parallel)
//!
//! The shared, order-dependent cell labelling (`att_cells` / `bas_cells`, mutated
//! by every seed and read by later seeds) is the dominant work-saver: a later seed
//! settles cheaply the instant it reaches an already-labelled cell. Lifting that
//! dependency to march every seed independently inflates the work ~34× (measured on
//! the two-well Duffing), so the kernel keeps the seeds strictly serial — the win
//! here is eliminating the per-`dt` FFI and the per-step Python, not threads.
//!
//! # Determinism
//!
//! No RNG, no rayon: the seed order is the caller's, the FSM is deterministic, and
//! the per-cell-check numerics are the engine's fixed stepper. A repeated run on
//! the same inputs returns byte-identical labels and cell maps.

use std::collections::{HashMap, HashSet};

use tsdyn_ir::Evaluator;
use tsdyn_solvers::Solver;

use crate::integrate::{integrate_grid, IntegrateConfig};
use crate::map::map_step_once;

/// The label a seed gets when it leaves the region / never settles (mirrors the
/// Python `DIVERGED = -1`).
pub const DIVERGED: i64 = -1;

/// Why a basin march could not be set up (a shape / grid validation failure). The
/// march itself never errors — a diverged trajectory is a [`DIVERGED`] label, not
/// an error — so this only covers caller-side mistakes the binding maps to
/// `ValueError`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BasinError {
    /// A buffer length / grid invariant disagrees with the tape or itself.
    BadShape(String),
}

impl core::fmt::Display for BasinError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BasinError::BadShape(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for BasinError {}

/// The six finite-state-machine thresholds, ported one-to-one from the Python
/// `_AttractorMapper` constructor.
#[derive(Clone, Copy, Debug)]
pub struct MarchConfig {
    /// Hard cap on steps per initial condition before giving up ([`DIVERGED`]).
    pub max_steps: usize,
    /// `consecutive_recurrences` (`mx_fnd`): new attractor after this many
    /// consecutive steps into already-visited (this-trajectory) cells.
    pub mx_fnd: usize,
    /// `attractor_locate_steps` (`mx_loc`): extra steps integrated once an
    /// attractor is declared, to flesh out its cell set.
    pub mx_loc: usize,
    /// `attractor_revisits` (`mx_att`): steps in a known attractor's cells before
    /// assigning to it.
    pub mx_att: usize,
    /// `basin_revisits` (`mx_bas`): steps in a known basin's cells before
    /// inheriting that basin.
    pub mx_bas: usize,
    /// `lost_steps` (`mx_lost`): consecutive steps outside the region before
    /// declaring divergence.
    pub mx_lost: usize,
}

/// A regular cell tessellation of the box `[lo, hi]` into `counts[i]` cells per
/// axis — the Rust port of the Python `_CellGrid`.
///
/// [`index`](CellGrid::index) bins a state to a **flat** row-major cell index (over
/// `counts`), or `None` when the state lies outside the box / is non-finite. The
/// float-domain range check, the closed-top-face `min(rel, n − 1e-9)` clamp, and
/// the finite guard are ported **exactly** from `_CellGrid.index`, so the cell
/// sequence is identical to the Python mapper's.
pub struct CellGrid {
    lo: Vec<f64>,
    /// Per-axis cell width `delta[i] = (hi[i] − lo[i]) / counts[i]`.
    delta: Vec<f64>,
    counts: Vec<i64>,
    /// Number of axes.
    pub dim: usize,
}

impl CellGrid {
    /// Build a cell grid over `[lo, hi]` with `counts[i]` cells on axis `i`.
    ///
    /// Validates the Python `_CellGrid.__init__` invariants (agreeing lengths,
    /// `counts >= 1`, `hi > lo` componentwise) so a malformed grid is a clean
    /// [`BasinError::BadShape`] rather than a panic.
    pub fn new(lo: Vec<f64>, hi: Vec<f64>, counts: Vec<i64>) -> Result<Self, BasinError> {
        let dim = lo.len();
        if hi.len() != dim || counts.len() != dim {
            return Err(BasinError::BadShape(
                "CellGrid lo, hi, and counts must agree in length".to_string(),
            ));
        }
        if dim == 0 {
            return Err(BasinError::BadShape(
                "CellGrid must have at least one axis".to_string(),
            ));
        }
        if counts.iter().any(|&c| c < 1) {
            return Err(BasinError::BadShape(
                "CellGrid counts must be >= 1".to_string(),
            ));
        }
        let mut delta = vec![0.0; dim];
        for i in 0..dim {
            // Mirror the Python `np.any(hi <= lo)` guard exactly (a `<=` test, so a
            // NaN bound — `NaN <= lo` is false — slips through here as it does in
            // NumPy, surfacing later as a non-finite `delta`/`rel` and an out-of-box
            // `None` rather than a spurious raise).
            if hi[i] <= lo[i] {
                return Err(BasinError::BadShape(
                    "CellGrid requires hi > lo componentwise".to_string(),
                ));
            }
            delta[i] = (hi[i] - lo[i]) / counts[i] as f64;
        }
        Ok(CellGrid {
            lo,
            delta,
            counts,
            dim,
        })
    }

    /// Bin point `u` to its **flat** cell index, or `None` if outside / non-finite.
    ///
    /// Ported verbatim from `_CellGrid.index`: compute `rel = (u − lo) / delta`,
    /// reject any non-finite component, reject `rel < 0` or `rel > counts` (the
    /// closed box `[lo, hi]`), then floor `min(rel, counts − 1e-9)` (the closed
    /// top face belongs to the last cell) into a per-axis integer. The per-axis
    /// indices fold into one row-major flat index over `counts`.
    #[inline]
    #[allow(clippy::needless_range_loop)] // indexes four parallel arrays by axis
    pub fn index(&self, u: &[f64]) -> Option<u64> {
        let mut flat: u64 = 0;
        for i in 0..self.dim {
            // `rel` in the float domain first — an arbitrarily large but finite
            // coordinate is rejected cleanly rather than overflowing the int cast.
            let rel = (u[i] - self.lo[i]) / self.delta[i];
            if !rel.is_finite() {
                return None;
            }
            let n = self.counts[i] as f64;
            if rel < 0.0 || rel > n {
                return None;
            }
            // The closed top face (rel == n) belongs to the last cell.
            let clamped = rel.min(n - 1e-9);
            let k = clamped.floor() as i64;
            // Defensive: a floored value is in `[0, counts-1]`, but guard the cast.
            if k < 0 || k >= self.counts[i] {
                return None;
            }
            flat = flat * self.counts[i] as u64 + k as u64;
        }
        Some(flat)
    }
}

/// The per-cell-check advance: either a flow stepper or a map iterator.
///
/// The FSM is identical for both; only "advance one step" differs — a flow
/// integrates one `dt` segment, a map applies `f` once.
enum MarchDriver<'e, F>
where
    F: Fn() -> Box<dyn Solver>,
{
    /// A continuous flow: advance one `dt` via [`integrate_grid`] over `[t, t+dt]`,
    /// re-seeding a fresh solver each segment (`solver_factory`) — byte-for-byte
    /// the per-`dt` `integrate_grid` the Python `ContinuousSystem.step(dt)` runs.
    Flow {
        ev: &'e dyn Evaluator,
        solver_factory: F,
        dt: f64,
    },
    /// A discrete map: apply `f` once via [`map_step_once`].
    Map { ev: &'e dyn Evaluator },
}

/// The live integration point a [`MarchDriver`] carries across cell checks.
struct MarchState {
    u: Vec<f64>,
    t: f64,
    /// The evaluator's working buffer (interpreter register file), reused so the
    /// loop allocates nothing on the map path.
    scratch: Vec<f64>,
    next: Vec<f64>,
}

impl<'e, F> MarchDriver<'e, F>
where
    F: Fn() -> Box<dyn Solver>,
{
    fn dim(&self) -> usize {
        match self {
            MarchDriver::Flow { ev, .. } => ev.dim(),
            MarchDriver::Map { ev } => ev.dim(),
        }
    }

    fn n_scratch(&self) -> usize {
        match self {
            MarchDriver::Flow { ev, .. } => ev.n_scratch(),
            MarchDriver::Map { ev } => ev.n_scratch(),
        }
    }

    /// Advance the live point one cell-check step in place; return `false` if the
    /// trajectory blew up (a diverged integration, a non-finite map iterate, or any
    /// non-finite component) — the Python `_advance() is None` contract.
    #[inline]
    fn advance(&self, st: &mut MarchState, p: &[f64]) -> bool {
        match self {
            MarchDriver::Flow {
                ev,
                solver_factory,
                dt,
            } => {
                let dim = ev.dim();
                let tf = st.t + dt;
                // Mirror `OdeStepper::advance` / the batch `integrate_dense` exactly:
                // the two-node grid is `[t, tf]` and the first step is `tf - t`
                // (the grid-derived step, NOT the raw `dt`). A *fresh* solver +
                // state per segment, so the adaptive controller re-seeds each `dt`
                // — the released per-`dt` numerics, bit-for-bit.
                let t_eval = [st.t, tf];
                let mut solver = solver_factory();
                let cfg = IntegrateConfig::new(tf - st.t);
                match integrate_grid(*ev, &mut *solver, &st.u[..dim], p, &t_eval, &cfg) {
                    Ok(out) => {
                        // `out` is the flat `(2, dim)` buffer; the last row is the
                        // advanced state.
                        let last = &out[dim..2 * dim];
                        if !last.iter().all(|x| x.is_finite()) {
                            return false;
                        }
                        st.u[..dim].copy_from_slice(last);
                        st.t = tf;
                        true
                    }
                    // A diverged / collapsed integration is "gone for good" — the
                    // Python `_advance` catches ConvergenceError → None.
                    Err(_) => false,
                }
            }
            MarchDriver::Map { ev } => {
                let dim = ev.dim();
                // One map application; `t = 0.0` (maps are autonomous in the IR),
                // matching the engine map loop and the reference path. A non-finite
                // iterate is divergence (`false`), the map step's overflow contract.
                if map_step_once(*ev, &st.u[..dim], p, &mut st.scratch, &mut st.next) {
                    st.u[..dim].copy_from_slice(&st.next[..dim]);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Reseat the live point to `(u, 0.0)` for a fresh initial condition (the
    /// `_reinit` of the Python mapper — a flow restarts at `t = 0`, a map has no
    /// time).
    fn reinit(&self, st: &mut MarchState, ic: &[f64]) {
        let dim = self.dim();
        st.u[..dim].copy_from_slice(&ic[..dim]);
        st.t = 0.0;
    }
}

/// The mutable, order-dependent labelling state shared across every seed — the
/// heart of the amortisation. Each seed reads AND writes these maps, so a later
/// seed inherits an earlier seed's label the instant it reaches a labelled cell.
struct LabelStore {
    /// cell → attractor id (the recurrent cell set of a located attractor).
    att_cells: HashMap<u64, i64>,
    /// cell → attractor id (a transient cell that led to that attractor).
    bas_cells: HashMap<u64, i64>,
    /// attractor id → the point cloud sampled while on it.
    att_points: HashMap<i64, Vec<Vec<f64>>>,
    next_id: i64,
}

impl LabelStore {
    fn new() -> Self {
        LabelStore {
            att_cells: HashMap::new(),
            bas_cells: HashMap::new(),
            att_points: HashMap::new(),
            next_id: 1,
        }
    }

    /// Mark a trajectory's transient cells as `att_id`'s basin (the Python
    /// `_label_basin`): a trail cell that is not already an attractor cell becomes
    /// a basin cell, **without** overwriting an existing basin label
    /// (`setdefault`).
    fn label_basin(&mut self, trail: &[u64], att_id: i64) {
        for &cell in trail {
            if !self.att_cells.contains_key(&cell) {
                self.bas_cells.entry(cell).or_insert(att_id);
            }
        }
    }
}

/// Run the recurrence FSM for one seed; return its attractor id or [`DIVERGED`].
///
/// A line-for-line port of `_AttractorMapper.map_ic` (+ `_locate_attractor`), over
/// the shared [`LabelStore`]. The counter logic, the `att`/`bas`/`visited` order,
/// the lost-counter, and the locate-then-label sequence match the Python exactly,
/// so the per-seed classification is bit-identical.
fn map_ic<F>(
    driver: &MarchDriver<'_, F>,
    st: &mut MarchState,
    grid: &CellGrid,
    p: &[f64],
    cfg: &MarchConfig,
    store: &mut LabelStore,
    ic: &[f64],
) -> i64
where
    F: Fn() -> Box<dyn Solver>,
{
    driver.reinit(st, ic);
    // this-trajectory visited cells (membership only — the Python `visited` dict's
    // step-index values are never read back, so a set captures the same semantics).
    let mut visited: HashSet<u64> = HashSet::new();
    let mut trail: Vec<u64> = Vec::new();
    let (mut c, mut att_hit, mut bas_hit, mut lost) = (0usize, 0usize, 0usize, 0usize);

    for _it in 0..cfg.max_steps {
        if !driver.advance(st, p) {
            // the step blew up — the trajectory is gone for good.
            return DIVERGED;
        }
        let cell = match grid.index(&st.u) {
            None => {
                // finite but outside the region: a transient excursion may return.
                lost += 1;
                c = 0;
                att_hit = 0;
                bas_hit = 0;
                if lost >= cfg.mx_lost {
                    return DIVERGED;
                }
                continue;
            }
            Some(cell) => cell,
        };
        lost = 0;

        if let Some(&known) = store.att_cells.get(&cell) {
            att_hit += 1;
            c = 0;
            bas_hit = 0;
            if att_hit >= cfg.mx_att {
                store.label_basin(&trail, known);
                return known;
            }
            continue;
        }

        if let Some(&known) = store.bas_cells.get(&cell) {
            bas_hit += 1;
            c = 0;
            att_hit = 0;
            if bas_hit >= cfg.mx_bas {
                store.label_basin(&trail, known);
                return known;
            }
            continue;
        }

        // an unlabelled cell
        att_hit = 0;
        bas_hit = 0;
        if visited.contains(&cell) {
            c += 1;
            if c >= cfg.mx_fnd {
                return locate_attractor(driver, st, grid, p, cfg, store, &trail);
            }
        } else {
            visited.insert(cell);
            c = 0;
        }
        trail.push(cell);
    }

    DIVERGED
}

/// Recurrence detected: integrate on to map the attractor's cells/points, assign a
/// new id, and convert the trail's transient cells to basin cells.
///
/// A port of `_AttractorMapper._locate_attractor`: integrate `mx_loc` more steps
/// recording each cell + live state; if the locate run left the region with no
/// cells, roll the id back and return [`DIVERGED`]; otherwise label the located
/// cells as the new attractor, drop them from `bas_cells`, store the point cloud,
/// and `setdefault` the trail's non-attractor cells into the basin.
fn locate_attractor<F>(
    driver: &MarchDriver<'_, F>,
    st: &mut MarchState,
    grid: &CellGrid,
    p: &[f64],
    cfg: &MarchConfig,
    store: &mut LabelStore,
    trail: &[u64],
) -> i64
where
    F: Fn() -> Box<dyn Solver>,
{
    let new_id = store.next_id;
    store.next_id += 1;
    let dim = driver.dim();
    // Unique located cells (the Python `att_cells` is a `set`, so only membership
    // matters — every located cell gets the same id, so order is irrelevant
    // downstream).
    let mut att_cells: Vec<u64> = Vec::new();
    let mut seen: HashSet<u64> = HashSet::new();
    let mut points: Vec<Vec<f64>> = Vec::new();

    for _ in 0..cfg.mx_loc {
        if !driver.advance(st, p) {
            break;
        }
        match grid.index(&st.u) {
            None => break,
            Some(cell) => {
                if seen.insert(cell) {
                    att_cells.push(cell);
                }
                points.push(st.u[..dim].to_vec());
            }
        }
    }

    if att_cells.is_empty() {
        // Could not pin the attractor down (left the region while locating).
        store.next_id -= 1;
        return DIVERGED;
    }

    for &cell in &att_cells {
        store.att_cells.insert(cell, new_id);
        store.bas_cells.remove(&cell);
    }
    store.att_points.insert(new_id, points);

    // transient cells that led here become basin cells (not the attractor's).
    for &cell in trail {
        if !store.att_cells.contains_key(&cell) {
            store.bas_cells.entry(cell).or_insert(new_id);
        }
    }
    new_id
}

/// The per-seed labels + the final labelling maps a [`basin_march`] returns.
///
/// The Python wiring reconstructs the mapper's `_att_cells` / `_bas_cells` /
/// `_att_points` from these so `merge_map`, `attractor_set` and the basin painting
/// run unchanged.
#[derive(Clone, Debug)]
pub struct BasinMarchOutcome {
    /// Per-seed attractor id (or [`DIVERGED`]), in input seed order.
    pub labels: Vec<i64>,
    /// Final `att_cells`: `(flat_cell, attractor_id)` pairs.
    pub att_cells: Vec<(u64, i64)>,
    /// Final `bas_cells`: `(flat_cell, attractor_id)` pairs.
    pub bas_cells: Vec<(u64, i64)>,
    /// Per attractor id, its sampled point cloud flattened row-major `(m, dim)`
    /// with the row count, so the binding can rebuild `att_points`.
    pub att_points: Vec<(i64, usize, Vec<f64>)>,
    /// System dimension (the point-cloud row width).
    pub dim: usize,
}

/// Run the recurrence FSM over every seed sequentially — the generic core.
///
/// `driver` advances one cell-check (a flow `dt` step or a map application);
/// `seeds` is a row-major `(n_seeds, dim)` buffer; `p` the live control parameters.
/// The seeds are processed strictly in order so the shared label store accumulates
/// exactly as the Python mapper's does (see the module note on why this is not
/// parallelised).
fn run_march<F>(
    driver: &MarchDriver<'_, F>,
    grid: &CellGrid,
    seeds: &[f64],
    p: &[f64],
    cfg: &MarchConfig,
) -> BasinMarchOutcome
where
    F: Fn() -> Box<dyn Solver>,
{
    let dim = driver.dim();
    let n_seeds = seeds.len().checked_div(dim).unwrap_or(0);
    let mut store = LabelStore::new();
    let mut labels = Vec::with_capacity(n_seeds);

    let mut st = MarchState {
        u: vec![0.0; dim],
        t: 0.0,
        scratch: vec![0.0; driver.n_scratch()],
        next: vec![0.0; dim],
    };

    for s in 0..n_seeds {
        let ic = &seeds[s * dim..(s + 1) * dim];
        let label = map_ic(driver, &mut st, grid, p, cfg, &mut store, ic);
        labels.push(label);
    }

    finish(store, labels, dim)
}

/// Pack the accumulated [`LabelStore`] into the FFI-shaped [`BasinMarchOutcome`].
fn finish(store: LabelStore, labels: Vec<i64>, dim: usize) -> BasinMarchOutcome {
    let att_cells: Vec<(u64, i64)> = store.att_cells.into_iter().collect();
    let bas_cells: Vec<(u64, i64)> = store.bas_cells.into_iter().collect();
    let att_points: Vec<(i64, usize, Vec<f64>)> = store
        .att_points
        .into_iter()
        .map(|(id, pts)| {
            let m = pts.len();
            let mut flat = Vec::with_capacity(m * dim);
            for row in pts {
                flat.extend_from_slice(&row);
            }
            (id, m, flat)
        })
        .collect();

    BasinMarchOutcome {
        labels,
        att_cells,
        bas_cells,
        att_points,
        dim,
    }
}

/// Validate a row-major `(n_seeds, dim)` seed buffer against `dim`.
fn check_seeds(seeds: &[f64], dim: usize) -> Result<(), BasinError> {
    if dim == 0 {
        return Err(BasinError::BadShape("system dimension is zero".to_string()));
    }
    if !seeds.len().is_multiple_of(dim) {
        return Err(BasinError::BadShape(format!(
            "seeds buffer length {} is not a multiple of dim = {dim}",
            seeds.len()
        )));
    }
    Ok(())
}

/// Run the basin recurrence FSM for a **flow** (continuous system).
///
/// `ev` is the built ODE evaluator; `solver_factory` builds a fresh kernel per
/// `dt` segment (the binding resolves the method and threads the tolerances in).
/// Each cell check advances one `dt` byte-for-byte as the Python
/// `ContinuousSystem.step(dt)` does, so the located cells/labels are bit-identical
/// to the Python `_AttractorMapper` driven over the same flow.
#[allow(clippy::too_many_arguments)]
pub fn basin_march_flow<F>(
    ev: &dyn Evaluator,
    solver_factory: F,
    p: &[f64],
    dt: f64,
    grid_lo: &[f64],
    grid_hi: &[f64],
    grid_counts: &[i64],
    seeds: &[f64],
    cfg: MarchConfig,
) -> Result<BasinMarchOutcome, BasinError>
where
    F: Fn() -> Box<dyn Solver>,
{
    if !(dt.is_finite() && dt > 0.0) {
        return Err(BasinError::BadShape(format!(
            "flow cell-check dt must be finite and positive, got {dt}"
        )));
    }
    let dim = ev.dim();
    check_seeds(seeds, dim)?;
    if p.len() < ev.n_param() {
        return Err(BasinError::BadShape(format!(
            "parameter vector has length {}, need n_param = {}",
            p.len(),
            ev.n_param()
        )));
    }
    let grid = CellGrid::new(grid_lo.to_vec(), grid_hi.to_vec(), grid_counts.to_vec())?;
    if grid.dim != dim {
        return Err(BasinError::BadShape(format!(
            "grid dimension {} does not match system dimension {dim}",
            grid.dim
        )));
    }
    let driver = MarchDriver::Flow {
        ev,
        solver_factory,
        dt,
    };
    Ok(run_march(&driver, &grid, seeds, p, &cfg))
}

/// Run the basin recurrence FSM for a **map** (discrete system).
///
/// As [`basin_march_flow`], but each cell check applies the lowered `_step` map
/// once (no solver, no `dt`). The map tape folds its parameters in, so `p` is
/// empty; a non-finite iterate is divergence, matching the Python map step's
/// overflow→`ConvergenceError`→`DIVERGED`.
pub fn basin_march_map(
    ev: &dyn Evaluator,
    p: &[f64],
    grid_lo: &[f64],
    grid_hi: &[f64],
    grid_counts: &[i64],
    seeds: &[f64],
    cfg: MarchConfig,
) -> Result<BasinMarchOutcome, BasinError> {
    let dim = ev.dim();
    check_seeds(seeds, dim)?;
    if p.len() < ev.n_param() {
        return Err(BasinError::BadShape(format!(
            "parameter vector has length {}, need n_param = {}",
            p.len(),
            ev.n_param()
        )));
    }
    let grid = CellGrid::new(grid_lo.to_vec(), grid_hi.to_vec(), grid_counts.to_vec())?;
    if grid.dim != dim {
        return Err(BasinError::BadShape(format!(
            "grid dimension {} does not match system dimension {dim}",
            grid.dim
        )));
    }
    // `MarchDriver::Map` never calls the solver factory; supply a never-used one to
    // satisfy the generic bound.
    let driver: MarchDriver<'_, fn() -> Box<dyn Solver>> = MarchDriver::Map { ev };
    Ok(run_march(&driver, &grid, seeds, p, &cfg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::VmEval;
    use tsdyn_ir::TapeBuilder;
    use tsdyn_solvers::explicit::Rk45;
    use tsdyn_vm::Interpreter;

    fn cfg() -> MarchConfig {
        MarchConfig {
            max_steps: 500,
            mx_fnd: 8,
            mx_loc: 5,
            mx_att: 2,
            mx_bas: 10,
            mx_lost: 20,
        }
    }

    /// `x ← a x − x³` — two stable fixed points at `±sqrt(a−1)` for `1 < a < 2`
    /// (`a` folded in as a constant; a lowered map has `n_param == 0`).
    fn cubic_map(a: f64) -> Interpreter {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let ac = b.constant(a);
        let ax = b.mul(ac, x);
        let x2 = b.mul(x, x);
        let x3 = b.mul(x2, x);
        let nx = b.sub(ax, x3);
        Interpreter::new(b.finish(&[nx], &[], 1, 0).unwrap())
    }

    #[test]
    fn cell_grid_index_matches_python_closed_box() {
        let g = CellGrid::new(vec![-1.0, -1.0], vec![1.0, 1.0], vec![2, 2]).unwrap();
        // centre of lower-left cell.
        assert_eq!(g.index(&[-0.5, -0.5]), Some(0));
        // upper-right corner is the closed top face → last cell (1,1) → flat 3.
        assert_eq!(g.index(&[1.0, 1.0]), Some(3));
        // outside.
        assert_eq!(g.index(&[1.5, 0.0]), None);
        // non-finite.
        assert_eq!(g.index(&[f64::NAN, 0.0]), None);
        assert_eq!(g.index(&[f64::INFINITY, 0.0]), None);
    }

    #[test]
    fn cubic_map_finds_two_attractors() {
        // a = 1.5 → fixed points at ±sqrt(0.5) ≈ ±0.707; the recurrence FSM should
        // discover two attractors and split the line into two basins.
        let ev = VmEval::new(cubic_map(1.5));
        // Seeds either side of 0 settle to the two fixed points.
        let seeds = vec![-0.5, 0.5, -0.9, 0.9, -0.1, 0.1];
        let out = basin_march_map(&ev, &[], &[-2.0], &[2.0], &[400], &seeds, cfg()).unwrap();
        assert_eq!(out.labels.len(), 6);
        // Two distinct positive labels discovered (the fixed points at ±√0.5).
        let mut ids: Vec<i64> = out.labels.iter().copied().filter(|&l| l >= 1).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            2,
            "expected two attractors, got labels {:?}",
            out.labels
        );
        // Each sign settles consistently: all negatives share one id, all positives
        // the other, and the two ids differ.
        assert_eq!(out.labels[0], out.labels[2]);
        assert_eq!(out.labels[0], out.labels[4]);
        assert_eq!(out.labels[1], out.labels[3]);
        assert_eq!(out.labels[1], out.labels[5]);
        assert_ne!(out.labels[0], out.labels[1]);
    }

    #[test]
    fn deterministic_across_runs() {
        let ev = VmEval::new(cubic_map(1.5));
        let seeds: Vec<f64> = (0..40).map(|i| -1.0 + 0.05 * i as f64).collect();
        let run = || {
            basin_march_map(&ev, &[], &[-2.0], &[2.0], &[400], &seeds, cfg())
                .unwrap()
                .labels
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn flow_decay_settles_to_one_attractor() {
        // dx/dt = -x, dv/dt = -v → spirals to the origin; one attractor at (0,0).
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let v = b.state(1);
        let dx = b.neg(x);
        let dv = b.neg(v);
        let tape = b.finish(&[dx, dv], &[], 2, 0).unwrap();
        let ev = VmEval::new(Interpreter::new(tape));
        let seeds = vec![0.5, 0.5, -0.5, -0.5, 0.3, -0.4];
        let out = basin_march_flow(
            &ev,
            || Box::new(Rk45::with_tolerances(1e-6, 1e-9)),
            &[],
            0.2,
            &[-1.0, -1.0],
            &[1.0, 1.0],
            &[50, 50],
            &seeds,
            MarchConfig {
                max_steps: 2000,
                mx_fnd: 20,
                mx_loc: 10,
                mx_att: 2,
                mx_bas: 10,
                mx_lost: 20,
            },
        )
        .unwrap();
        // Every seed settles (no DIVERGED) and at least one attractor is located.
        // (Distinct nearby origin cells may be located as separate ids; the Python
        // `merge_map` then unions them — that is a post-kernel step, so the kernel
        // need only locate the attractor cells near the origin.)
        let ids: Vec<i64> = out.labels.iter().copied().filter(|&l| l >= 1).collect();
        assert_eq!(
            ids.len(),
            out.labels.len(),
            "a seed diverged: {:?}",
            out.labels
        );
        assert!(!out.att_points.is_empty(), "no attractor located");
        // Every located point cloud sits near the origin (the only attractor).
        for (_id, m, flat) in &out.att_points {
            assert!(*m > 0);
            for chunk in flat.chunks(out.dim) {
                let r2: f64 = chunk.iter().map(|x| x * x).sum();
                assert!(r2 < 0.25, "attractor point far from origin: {chunk:?}");
            }
        }
    }

    #[test]
    fn rejects_dim_mismatched_grid() {
        let ev = VmEval::new(cubic_map(1.5));
        let err = basin_march_map(
            &ev,
            &[],
            &[-2.0, -2.0],
            &[2.0, 2.0],
            &[10, 10],
            &[0.1],
            cfg(),
        )
        .unwrap_err();
        assert!(matches!(err, BasinError::BadShape(_)));
    }
}
