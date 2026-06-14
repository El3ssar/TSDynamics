//! Euler–Maruyama — the first-order (strong order 0.5) explicit SDE scheme.
//!
//! For a diagonal-Itô SDE `dX_k = f_k(X, t) dt + g_k(X, t) dW_k`, one step of
//! width `h` from `X^n` is
//!
//! ```text
//!   X_k^{n+1} = X_k^n + f_k(X^n, t) · h + g_k(X^n, t) · ΔW_k,
//! ```
//!
//! where `ΔW_k ~ N(0, h)` is the engine-drawn increment for component `k`
//! (passed in as `dw[k]`). It is the SDE analogue of the explicit Euler ODE
//! method: one drift evaluation and one diffusion evaluation per step, no
//! Jacobian. When the noise is **additive** (`g` independent of `X`) Euler–
//! Maruyama coincides with [`Milstein`](super::Milstein), since the Milstein
//! correction `½ g g' (ΔW² − h)` vanishes with `g' = 0`.

use crate::{
    register_sde_kernel, Caps, Evaluator, ProblemKind, ProblemKinds, SolverState, StepOutcome,
};

use super::SdeKernel;

/// The Euler–Maruyama diagonal-Itô kernel.
///
/// Stage buffers (`f`, `g`, and the two evaluators' scratch) grow to size on the
/// first step and are reused thereafter — no per-step allocation, mirroring the
/// explicit RK kernels.
#[derive(Default)]
pub struct EulerMaruyama {
    /// Drift `f(u, t)` for the current step.
    f: Vec<f64>,
    /// Diagonal diffusion `g(u, t)` for the current step.
    g: Vec<f64>,
    /// Scratch for the drift evaluator (its register file; empty for the JIT).
    drift_scratch: Vec<f64>,
    /// Scratch for the diffusion evaluator.
    diff_scratch: Vec<f64>,
}

impl EulerMaruyama {
    /// A fresh kernel with unallocated buffers (sized on the first step).
    pub fn new() -> Self {
        EulerMaruyama::default()
    }

    /// Size the stage buffers for `dim` state components and the two evaluators'
    /// scratch widths. Cheap no-op once the sizes are right.
    fn ensure(&mut self, dim: usize, drift: &dyn Evaluator, diffusion: &dyn Evaluator) {
        if self.f.len() != dim {
            self.f = vec![0.0; dim];
            self.g = vec![0.0; dim];
        }
        if self.drift_scratch.len() != drift.n_scratch() {
            self.drift_scratch = vec![0.0; drift.n_scratch()];
        }
        if self.diff_scratch.len() != diffusion.n_scratch() {
            self.diff_scratch = vec![0.0; diffusion.n_scratch()];
        }
    }
}

impl SdeKernel for EulerMaruyama {
    fn name(&self) -> &'static str {
        "euler_maruyama"
    }

    fn caps(&self) -> Caps {
        // Explicit, no diffusion Jacobian needed (that is Milstein's).
        Caps::explicit(ProblemKinds::of(ProblemKind::Sde))
    }

    fn step(
        &mut self,
        drift: &dyn Evaluator,
        diffusion: &dyn Evaluator,
        st: &mut SolverState,
        dw: &[f64],
        h: f64,
    ) -> StepOutcome {
        let SolverState {
            u,
            t,
            p,
            scratch: _,
        } = st;
        let dim = u.len();
        debug_assert_eq!(
            dw.len(),
            dim,
            "Wiener increment length must equal the dimension"
        );
        self.ensure(dim, drift, diffusion);

        drift.eval(u, p, *t, &mut self.drift_scratch, &mut self.f);
        diffusion.eval(u, p, *t, &mut self.diff_scratch, &mut self.g);

        for k in 0..dim {
            u[k] += self.f[k] * h + self.g[k] * dw[k];
        }
        *t += h;
        StepOutcome::Accepted { h_next: h }
    }
}

register_sde_kernel!(
    "euler_maruyama",
    Caps::explicit(ProblemKinds::of(ProblemKind::Sde)),
    || Box::new(EulerMaruyama::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sde::testkit::{ConstDiffusion, LinearDrift, ProportionalDiffusion};

    #[test]
    fn caps_are_explicit_sde_without_jacobian() {
        let s = EulerMaruyama::new();
        assert_eq!(s.name(), "euler_maruyama");
        let c = s.caps();
        assert!(c.supports(ProblemKind::Sde));
        assert!(!c.needs_jacobian);
        assert!(!c.adaptive);
    }

    #[test]
    fn step_applies_the_exact_euler_maruyama_formula() {
        // dX = θ(μ − X) dt + σ dW, one step from X0 with a hand-chosen ΔW so the
        // update is exact and independent of any RNG.
        let (theta, mu, sigma) = (1.5, 2.0, 0.4);
        let drift = LinearDrift::new(theta, mu); // f(x) = θ(μ − x)
        let diffusion = ConstDiffusion::new(sigma); // g(x) = σ
        let mut s = EulerMaruyama::new();

        let mut st = SolverState {
            u: vec![0.5],
            t: 0.0,
            p: vec![],
            scratch: vec![],
        };
        let dw = [0.3];
        let h = 0.1;
        s.step(&drift, &diffusion, &mut st, &dw, h);

        let want = 0.5 + theta * (mu - 0.5) * h + sigma * dw[0];
        assert!(
            (st.u[0] - want).abs() < 1e-15,
            "u = {}, want {want}",
            st.u[0]
        );
        assert!((st.t - h).abs() < 1e-15, "time must advance by h");
    }

    #[test]
    fn multiplicative_noise_uses_g_at_the_current_state() {
        // dX = μX dt + σX dW (geometric Brownian motion). Euler–Maruyama evaluates
        // g(X) = σX at the *current* state, so the noise term is σ X0 ΔW.
        let (m, sigma) = (0.05, 0.3);
        let drift = LinearDrift::new(-m, 0.0); // f(x) = -(-m)(0 - x) = m x
        let diffusion = ProportionalDiffusion::new(sigma); // g(x) = σx
        let mut s = EulerMaruyama::new();

        let mut st = SolverState {
            u: vec![2.0],
            t: 0.0,
            p: vec![],
            scratch: vec![],
        };
        let dw = [-0.2];
        let h = 0.05;
        s.step(&drift, &diffusion, &mut st, &dw, h);

        let want = 2.0 + m * 2.0 * h + sigma * 2.0 * dw[0];
        assert!(
            (st.u[0] - want).abs() < 1e-15,
            "u = {}, want {want}",
            st.u[0]
        );
    }

    #[test]
    fn zero_noise_reduces_to_explicit_euler() {
        // With ΔW = 0 the scheme is the deterministic explicit Euler step.
        let drift = LinearDrift::new(1.0, 0.0); // f(x) = -x
        let diffusion = ConstDiffusion::new(0.7);
        let mut s = EulerMaruyama::new();

        let mut st = SolverState {
            u: vec![1.0, -2.0],
            t: 0.0,
            p: vec![],
            scratch: vec![],
        };
        let h = 0.01;
        s.step(&drift, &diffusion, &mut st, &[0.0, 0.0], h);
        // f(x) = θ(μ − x) = 1·(0 − x) = −x ⇒ u_new = u + h·(−u).
        assert!((st.u[0] - (1.0 - h)).abs() < 1e-15, "u0 = {}", st.u[0]);
        assert!(
            (st.u[1] - (-2.0 + 2.0 * h)).abs() < 1e-15,
            "u1 = {}",
            st.u[1]
        );
    }
}
