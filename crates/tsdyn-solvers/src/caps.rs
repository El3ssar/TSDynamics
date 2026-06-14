//! Solver capability flags — the metadata that drives `method=` resolution and
//! the auto-stiffness layer (ROADMAP §4d).
//!
//! A [`Caps`] value answers the questions the engine asks when picking and
//! configuring a kernel: is it explicit or implicit, does it control its own
//! step size, does it require the analytic Jacobian, which problem families
//! ([`ProblemKind`]) can it integrate, and whether it carries a native
//! dense-output interpolant ([`Caps::dense`]).  Capabilities are pure data,
//! constructed in `const` context so a kernel can register them in a static
//! initializer (see [`crate::register_solver!`]).

/// A single problem family an evaluator/solver can describe or handle.
///
/// The discriminants are powers of two so a *set* of kinds packs into the
/// [`ProblemKinds`] bitset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ProblemKind {
    /// Ordinary differential equation `du/dt = f(u, p, t)`.
    Ode = 1 << 0,
    /// Delay differential equation (history-dependent RHS).
    Dde = 1 << 1,
    /// Stochastic differential equation (drift + diffusion; diagonal-Itô, §11).
    Sde = 1 << 2,
    /// Discrete map `u_{n+1} = f(u_n, p, n)`.
    Map = 1 << 3,
}

/// A set of [`ProblemKind`]s, packed as a bitset.
///
/// Built in `const` context by `|`-combining kinds:
/// `ProblemKinds::of(ProblemKind::Ode).with(ProblemKind::Sde)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ProblemKinds(u8);

impl ProblemKinds {
    /// The empty set — supports nothing (rarely useful except as a builder seed).
    pub const fn none() -> Self {
        ProblemKinds(0)
    }

    /// The singleton set `{kind}`.
    pub const fn of(kind: ProblemKind) -> Self {
        ProblemKinds(kind as u8)
    }

    /// This set with `kind` added (const-friendly chaining).
    pub const fn with(self, kind: ProblemKind) -> Self {
        ProblemKinds(self.0 | kind as u8)
    }

    /// The union of two sets.
    pub const fn union(self, other: ProblemKinds) -> Self {
        ProblemKinds(self.0 | other.0)
    }

    /// Whether `kind` is a member.
    pub const fn contains(self, kind: ProblemKind) -> bool {
        self.0 & (kind as u8) != 0
    }

    /// Whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl From<ProblemKind> for ProblemKinds {
    fn from(kind: ProblemKind) -> Self {
        ProblemKinds::of(kind)
    }
}

impl core::ops::BitOr for ProblemKind {
    type Output = ProblemKinds;
    fn bitor(self, rhs: ProblemKind) -> ProblemKinds {
        ProblemKinds::of(self).with(rhs)
    }
}

impl core::ops::BitOr<ProblemKind> for ProblemKinds {
    type Output = ProblemKinds;
    fn bitor(self, rhs: ProblemKind) -> ProblemKinds {
        self.with(rhs)
    }
}

/// Whether a kernel is explicit or implicit — the primary axis the
/// auto-stiffness layer switches on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SolverKind {
    /// Explicit method (no algebraic solve per step): RK4, DP45, DOP853, …
    Explicit,
    /// Implicit method (solves a (non)linear system per step, typically with the
    /// Jacobian): Rosenbrock, SDIRK/TR-BDF2, BDF, …
    Implicit,
}

/// What a solver can do.  See the [module docs](self).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Caps {
    /// Explicit vs implicit.
    pub kind: SolverKind,
    /// Performs its own embedded error estimate and step-size adaption.
    pub adaptive: bool,
    /// Requires the evaluator to provide the analytic Jacobian
    /// ([`Evaluator::has_jacobian`](tsdyn_ir::Evaluator::has_jacobian)).
    pub needs_jacobian: bool,
    /// The problem families this kernel can integrate.
    pub supports: ProblemKinds,
    /// Carries a *native* dense-output interpolant — a continuous extension of
    /// the just-taken step exposed through
    /// [`Solver::interpolate`](crate::Solver::interpolate).
    ///
    /// `false` (the default for every constructor) means the kernel offers no
    /// interpolant of its own; the engine's event/dense-output layer then falls
    /// back to a cubic-Hermite continuous extension built from the step
    /// endpoints and their derivatives, which needs only the
    /// [`Evaluator`](tsdyn_ir::Evaluator).  So this flag is purely an
    /// *optimisation/accuracy* signal — every kernel gets O(h⁴) dense output
    /// regardless — and existing kernels need no change (ROADMAP §13d: the
    /// event/dense-output capability is additive, behind this flag).
    pub dense: bool,
}

impl Caps {
    /// An explicit kernel for the given families (not adaptive, no Jacobian).
    /// Refine with [`adaptive`](Caps::adaptive) / [`with_jacobian`](Caps::with_jacobian).
    pub const fn explicit(supports: ProblemKinds) -> Self {
        Caps {
            kind: SolverKind::Explicit,
            adaptive: false,
            needs_jacobian: false,
            supports,
            dense: false,
        }
    }

    /// An implicit kernel for the given families.  Implicit kernels default to
    /// `needs_jacobian = true` (the usual case); for a Jacobian-free variant, or
    /// an explicit kernel that *does* want the Jacobian, set the field directly
    /// with struct-update syntax: `Caps { needs_jacobian: false, ..c }`.
    pub const fn implicit(supports: ProblemKinds) -> Self {
        Caps {
            kind: SolverKind::Implicit,
            adaptive: false,
            needs_jacobian: true,
            supports,
            dense: false,
        }
    }

    /// Mark this kernel as adaptive (own error control + step adaption).
    pub const fn adaptive(mut self) -> Self {
        self.adaptive = true;
        self
    }

    /// Mark this kernel as carrying a native dense-output interpolant
    /// ([`dense`](Caps::dense)) — it implements
    /// [`Solver::interpolate`](crate::Solver::interpolate), so the engine uses
    /// that continuous extension for event detection / dense output instead of
    /// the endpoint cubic-Hermite fallback.
    pub const fn with_dense(mut self) -> Self {
        self.dense = true;
        self
    }

    /// Whether this kernel can integrate problems of `kind`.
    pub const fn supports(&self, kind: ProblemKind) -> bool {
        self.supports.contains(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn problem_kinds_set_algebra() {
        let s = ProblemKind::Ode | ProblemKind::Sde;
        assert!(s.contains(ProblemKind::Ode));
        assert!(s.contains(ProblemKind::Sde));
        assert!(!s.contains(ProblemKind::Dde));
        assert!(!s.contains(ProblemKind::Map));
        assert!(!s.is_empty());
        assert!(ProblemKinds::none().is_empty());
    }

    #[test]
    fn const_construction_in_static() {
        // The whole point: a kernel registers Caps in a static initializer.
        static C: Caps = Caps::explicit(ProblemKinds::of(ProblemKind::Ode)).adaptive();
        assert_eq!(C.kind, SolverKind::Explicit);
        assert!(C.adaptive);
        assert!(!C.needs_jacobian);
        assert!(C.supports(ProblemKind::Ode));
    }

    #[test]
    fn implicit_defaults_to_needing_jacobian() {
        let c = Caps::implicit(ProblemKinds::of(ProblemKind::Ode));
        assert_eq!(c.kind, SolverKind::Implicit);
        assert!(c.needs_jacobian);
    }

    #[test]
    fn dense_defaults_off_and_is_opt_in() {
        // Every kernel built through the constructors gets `dense = false`, so
        // the engine's endpoint-Hermite fallback drives them and existing
        // kernels need no change (ROADMAP §13d).
        assert!(!Caps::explicit(ProblemKinds::of(ProblemKind::Ode)).dense);
        assert!(!Caps::implicit(ProblemKinds::of(ProblemKind::Ode)).dense);
        // `with_dense()` is the explicit opt-in, composable in const context.
        static C: Caps = Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
            .adaptive()
            .with_dense();
        assert!(C.dense);
        assert!(C.adaptive);
        // Idempotent and order-independent with the other builders.
        let ode = ProblemKinds::of(ProblemKind::Ode);
        assert!(Caps::explicit(ode).with_dense().with_dense().dense);
        assert_eq!(
            Caps::explicit(ode).adaptive().with_dense(),
            Caps::explicit(ode).with_dense().adaptive()
        );
    }
}
