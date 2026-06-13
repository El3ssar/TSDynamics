//! The solver name registry — auto-populated, no central table (ROADMAP §4d).
//!
//! Each kernel registers itself *from its own file* with
//! [`register_solver!`](crate::register_solver), which submits a
//! [`SolverRegistration`] into a link-time [`inventory`] collection.  Looking a
//! kernel up by name walks that collection — there is no hand-edited dispatch
//! table, so two solver streams (E3, E4, E-SDE) never touch a shared file and
//! never merge-conflict.  This is the Rust half of D4; the Python half
//! (`tsdynamics.solvers` directory scan + `tsdynamics.plugins` entry points)
//! mirrors it for out-of-tree plugins.
//!
//! Registration happens at static-init time, so it requires no startup call:
//! any kernel linked into the final binary (interpreter wheel, test binary, …)
//! is discoverable. A kernel that is *not* linked is simply absent — which is
//! why [`register_solver!`] must live in the kernel's own always-compiled module.
//!
//! # Two hazards for downstream streams to own
//!
//! 1. **Clashing names are silent.** [`inventory`] has no notion of a unique key:
//!    two kernels registering the same name both land in the collection and
//!    [`find`] returns whichever the linker ordered first, shadowing the other
//!    with no error. The two solver streams (E3, E4) run in parallel and each
//!    adds `register_solver!` lines, so this is a real collision risk. It is
//!    caught not at registration but by [`duplicates`] (and the registry
//!    uniqueness test) — which see every kernel linked into the binary, so a
//!    clash fails CI. The engine should also call [`duplicates`] once at startup.
//! 2. **Dead-stripping in the cdylib (hand-off to E7 / I-WHEEL).** Link-time
//!    registration relies on the kernel object code surviving into the final
//!    artifact. When `tsdyn-core` becomes the `cdylib` wheel (stream E7), a
//!    linker that garbage-collects sections can drop a kernel whose symbols are
//!    otherwise unreferenced, leaving it silently unregistered. E7/I-WHEEL must
//!    add a "registry is non-empty in the built wheel" smoke test; if stripping
//!    bites, the fix is a `force_link()` reference per kernel or a
//!    `whole-archive`/`-u` link flag. F2 freezes the mechanism and flags this; it
//!    cannot be verified until the cdylib exists.

use crate::caps::Caps;
use crate::solver::Solver;

/// A single registered kernel: its name, its [`Caps`], and a factory that builds
/// a fresh boxed [`Solver`] instance (so each integration/worker gets its own,
/// with its own stage buffers).
///
/// Submitted into the [`inventory`] collection by [`register_solver!`]; the
/// `make` field is a plain `fn` pointer so the whole value is `const`-built in a
/// static initializer.
pub struct SolverRegistration {
    /// The unique `method=` name.
    pub name: &'static str,
    /// The kernel's capabilities (for `method=` resolution / auto-stiffness).
    pub caps: Caps,
    /// Factory: a fresh boxed instance per call.
    pub make: fn() -> Box<dyn Solver>,
}

inventory::collect!(SolverRegistration);

/// Iterate every registered kernel, in unspecified order.
pub fn registered() -> impl Iterator<Item = &'static SolverRegistration> {
    inventory::iter::<SolverRegistration>()
}

/// The registration for `name`, or `None` if no kernel registered under it.
pub fn find(name: &str) -> Option<&'static SolverRegistration> {
    registered().find(|r| r.name == name)
}

/// Build a fresh boxed [`Solver`] for `name`, or `None` if unknown.
///
/// The engine/Python layer turns the `None` into a clear "unknown method"
/// error listing [`available`] names.
pub fn make(name: &str) -> Option<Box<dyn Solver>> {
    find(name).map(|r| (r.make)())
}

/// Every registered name, sorted — for "unknown method, available: …" messages
/// and for the Python layer to mirror.
pub fn available() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = registered().map(|r| r.name).collect();
    names.sort_unstable();
    names
}

/// Any names registered by more than one kernel, sorted and de-duplicated.
///
/// Empty when every kernel name is unique (the healthy state).  Because
/// [`inventory`] cannot reject a clashing registration (see the module docs),
/// this is the tripwire: the engine should call it once at startup and the
/// registry uniqueness test asserts it is empty in CI, so a name collision
/// between two solver streams fails loudly instead of silently shadowing a
/// kernel.
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

/// Register a solver kernel by name from its own module — the one line a kernel
/// file adds, with no central table to edit (ROADMAP §4d).
///
/// `$make` is a non-capturing closure (or `fn`) returning a boxed [`Solver`].
/// It expands to an [`inventory`] submission referenced through
/// `$crate::inventory` (re-exported at the crate root), so the registering crate
/// (E3/E4/E-SDE, or an out-of-tree kernel crate) needs only a dependency on
/// `tsdyn-solvers`, not on `inventory` itself.
///
/// ```ignore
/// use tsdyn_solvers::{Caps, ProblemKind, ProblemKinds, register_solver};
/// register_solver!("rk4", Caps::explicit(ProblemKinds::of(ProblemKind::Ode)),
///                  || Box::new(Rk4::new()));
/// ```
#[macro_export]
macro_rules! register_solver {
    ($name:expr, $caps:expr, $make:expr) => {
        $crate::inventory::submit! {
            $crate::SolverRegistration {
                name: $name,
                caps: $caps,
                make: $make,
            }
        }
    };
}
