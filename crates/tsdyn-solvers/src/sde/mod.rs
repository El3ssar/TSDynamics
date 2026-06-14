//! The stochastic (SDE) solver family — diagonal-Itô kernels (stream E-SDE).
//!
//! One file per scheme, each ending in a [`register_sde_kernel!`](crate::register_sde_kernel)
//! so it is discoverable by name with no central table to edit (the same
//! no-central-table design as the [`Solver`](crate::Solver) registry; ROADMAP §4d):
//!
//! | name | scheme | strong order | uses ∂g/∂u |
//! |------|--------|--------------|------------|
//! | `euler_maruyama` | Euler–Maruyama | 0.5 | no |
//! | `milstein` | Milstein (diagonal) | 1.0 | yes |
//!
//! # Why a separate trait from [`Solver`](crate::Solver)
//!
//! An SDE step is not an ODE step with the frozen [`Solver::step`](crate::Solver::step)
//! signature: it needs **two** right-hand sides (a drift `f` and a diffusion
//! `g`, each a `dim → dim` [`Evaluator`]) and a **Wiener increment** per step,
//! while `Solver::step` carries one evaluator and no randomness. Rather than
//! widen the frozen `Solver` trait (forbidden in a feature stream), the SDE
//! family gets its own [`SdeKernel`] trait — exactly as the discrete-map family
//! drives its own engine loop instead of shoehorning maps into `Solver`.
//!
//! # The RNG lives in the engine, not here
//!
//! `tsdyn-solvers` depends only on `tsdyn-ir`; the seeded RNG / Wiener substrate
//! ([`SplitMix64`](https://docs.rs/tsdyn-engine) / `fill_wiener`) lives in
//! `tsdyn-engine`, *downstream* of this crate. So a kernel cannot draw its own
//! noise — and must not, because the parallel-equals-serial determinism contract
//! (ROADMAP §4c) requires the engine to seed each trajectory's stream by index.
//! The seam is therefore: the **engine draws the diagonal Wiener increment and
//! hands the kernel a pre-drawn `dw` slice**; the kernel is a pure, deterministic
//! function of `(u, t, p, dw, h)`. This also makes each scheme's update formula
//! exactly unit-testable with a hand-chosen `dw` (see the kernel test suites).
//!
//! [`Evaluator`]: tsdyn_ir::Evaluator

mod euler_maruyama;
mod milstein;

#[cfg(test)]
mod testkit;

pub use euler_maruyama::EulerMaruyama;
pub use milstein::Milstein;

use crate::caps::Caps;
use crate::solver::{SolverState, StepOutcome};
use tsdyn_ir::Evaluator;

/// One stochastic integration kernel for a diagonal-Itô SDE
/// `dX_k = f_k(X, t) dt + g_k(X, t) dW_k` (independent `dW_k`; ROADMAP §11).
///
/// Implemented by [`EulerMaruyama`] and [`Milstein`]; driven by the engine's SDE
/// loop (`tsdyn-engine`'s `sde` module) as `&mut dyn SdeKernel`, selected by a
/// `method=` string through this module's [`make`]. See the [module docs](self)
/// for why this is a separate trait from [`Solver`](crate::Solver) and why the
/// Wiener increment arrives pre-drawn.
///
/// # `Send`
///
/// Like [`Solver`](crate::Solver), the [`Send`] supertrait lets the engine build
/// one kernel per ensemble worker and move it across threads; a kernel is a plain
/// struct of `Vec<f64>` stage buffers, so this costs nothing. It is deliberately
/// **not** `Sync` — each worker owns its own `&mut` kernel (the evaluators, by
/// contrast, are shared `&dyn Evaluator` and so are [`Sync`](tsdyn_ir::Evaluator)).
pub trait SdeKernel: Send {
    /// The kernel's registry name — the string a user passes as `method=`.
    ///
    /// Must be unique across registered SDE kernels; a clash is caught by
    /// [`duplicates`] / the registry uniqueness test (link-time registration
    /// cannot reject it), exactly as for the [`Solver`](crate::Solver) registry.
    fn name(&self) -> &'static str;

    /// This kernel's [capabilities](Caps).
    ///
    /// All SDE kernels report [`ProblemKind::Sde`](crate::ProblemKind::Sde).
    /// [`needs_jacobian`](Caps::needs_jacobian) is `true` for a scheme that reads
    /// the **diffusion** Jacobian `∂g/∂u` (Milstein) — the signal to the
    /// engine/Python layer to lower the diffusion tape with its Jacobian.
    fn caps(&self) -> Caps;

    /// Advance the integration state `st` in place by one diagonal-Itô step of
    /// width `h`, using the pre-drawn diagonal Wiener increment `dw`.
    ///
    /// - `drift` evaluates `f(u, p, t)` (one output per component);
    /// - `diffusion` evaluates `g(u, p, t)` (the per-component diagonal noise
    ///   coefficients) — and, for a Jacobian-using scheme, its `∂g/∂u` via
    ///   [`Evaluator::eval_jac`];
    /// - [`st.u`](SolverState::u)/[`t`](SolverState::t) are the current point
    ///   (advanced in place), [`p`](SolverState::p) the parameters;
    /// - `dw[k] ~ N(0, h)` is the engine-drawn increment for component `k`
    ///   (`dw.len() == st.u.len() == drift.dim()`); the kernel never draws its
    ///   own noise (see the [module docs](self)).
    ///
    /// The kernel reads/writes only `st.u` and advances `st.t` by `h`; its stage
    /// buffers (`f`, `g`, `∂g/∂u`, evaluator scratch) are its own `&mut self`
    /// fields, allocated on first use and reused. Unlike an ODE kernel it does
    /// **not** use [`st.scratch`](SolverState::scratch): it drives two
    /// evaluators, each needing its own scratch width, so a single shared buffer
    /// cannot serve both — the engine may leave `st.scratch` empty. The kernel
    /// returns [`StepOutcome::Accepted`] with `h_next == h` (these schemes are
    /// fixed-step); the engine performs the finiteness check and the divergence
    /// policy, so a kernel need not self-diagnose a blow-up.
    fn step(
        &mut self,
        drift: &dyn Evaluator,
        diffusion: &dyn Evaluator,
        st: &mut SolverState,
        dw: &[f64],
        h: f64,
    ) -> StepOutcome;
}

/// A single registered SDE kernel: its name, [`Caps`], and a factory building a
/// fresh boxed [`SdeKernel`] (so each worker gets its own stage buffers).
///
/// Submitted into the [`inventory`](crate::inventory) collection by
/// [`register_sde_kernel!`]; `make` is a plain `fn` pointer so the whole value is
/// `const`-built in a static initializer. Mirrors
/// [`SolverRegistration`](crate::SolverRegistration) for the SDE family.
pub struct SdeRegistration {
    /// The unique `method=` name.
    pub name: &'static str,
    /// The kernel's capabilities.
    pub caps: Caps,
    /// Factory: a fresh boxed instance per call.
    pub make: fn() -> Box<dyn SdeKernel>,
}

inventory::collect!(SdeRegistration);

/// Iterate every registered SDE kernel, in unspecified order.
pub fn registered() -> impl Iterator<Item = &'static SdeRegistration> {
    inventory::iter::<SdeRegistration>()
}

/// The registration for `name`, or `None` if no SDE kernel registered under it.
pub fn find(name: &str) -> Option<&'static SdeRegistration> {
    registered().find(|r| r.name == name)
}

/// Build a fresh boxed [`SdeKernel`] for `name`, or `None` if unknown.
///
/// The engine/Python layer turns the `None` into a clear "unknown SDE method"
/// error listing [`available`] names.
pub fn make(name: &str) -> Option<Box<dyn SdeKernel>> {
    find(name).map(|r| (r.make)())
}

/// Every registered SDE kernel name, sorted — for "unknown method, available: …"
/// messages and for the Python layer to mirror.
pub fn available() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = registered().map(|r| r.name).collect();
    names.sort_unstable();
    names
}

/// Any SDE kernel names registered by more than one kernel, sorted and unique.
///
/// Empty in the healthy state; the tripwire for the link-time registry (which
/// cannot reject a clashing name), asserted by the registry uniqueness test —
/// the SDE analogue of [`duplicates`](crate::duplicates).
pub fn duplicates() -> Vec<&'static str> {
    let mut seen: Vec<&'static str> = Vec::new();
    let mut dups: Vec<&'static str> = Vec::new();
    for name in registered().map(|r| r.name) {
        if seen.contains(&name) {
            if !dups.contains(&name) {
                dups.push(name);
            }
        } else {
            seen.push(name);
        }
    }
    dups.sort_unstable();
    dups
}

/// Register an SDE kernel by name from its own module — the one line a kernel
/// file adds, with no central table to edit (the SDE twin of
/// [`register_solver!`](crate::register_solver)).
///
/// `$make` is a non-capturing closure (or `fn`) returning a boxed [`SdeKernel`].
/// It expands to an [`inventory`](crate::inventory) submission referenced through
/// `$crate::inventory`, so a registering crate needs only a dependency on
/// `tsdyn-solvers`.
///
/// ```ignore
/// use tsdyn_solvers::{Caps, ProblemKind, ProblemKinds, register_sde_kernel};
/// use tsdyn_solvers::sde::EulerMaruyama;
/// register_sde_kernel!(
///     "euler_maruyama",
///     Caps::explicit(ProblemKinds::of(ProblemKind::Sde)),
///     || Box::new(EulerMaruyama::new())
/// );
/// ```
#[macro_export]
macro_rules! register_sde_kernel {
    ($name:expr, $caps:expr, $make:expr) => {
        $crate::inventory::submit! {
            $crate::sde::SdeRegistration {
                name: $name,
                caps: $caps,
                make: $make,
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProblemKind, SolverKind};

    #[test]
    fn both_builtin_kernels_are_registered_and_unique() {
        let names = available();
        assert!(names.contains(&"euler_maruyama"), "got {names:?}");
        assert!(names.contains(&"milstein"), "got {names:?}");
        assert!(
            duplicates().is_empty(),
            "duplicate SDE kernel names: {:?}",
            duplicates()
        );
    }

    #[test]
    fn make_builds_by_name_and_reports_sde_caps() {
        let em = make("euler_maruyama").expect("euler_maruyama must be registered");
        assert_eq!(em.name(), "euler_maruyama");
        let c = em.caps();
        assert_eq!(c.kind, SolverKind::Explicit);
        assert!(c.supports(ProblemKind::Sde));
        assert!(
            !c.needs_jacobian,
            "Euler–Maruyama needs no diffusion Jacobian"
        );

        let mil = make("milstein").expect("milstein must be registered");
        assert_eq!(mil.name(), "milstein");
        assert!(mil.caps().needs_jacobian, "Milstein reads ∂g/∂u");
    }

    #[test]
    fn unknown_name_is_none() {
        assert!(make("no_such_scheme").is_none());
        assert!(find("no_such_scheme").is_none());
    }
}
