//! Milstein — the order-1.0 (strong) explicit SDE scheme for diagonal noise.
//!
//! For a diagonal-Itô SDE `dX_k = f_k(X, t) dt + g_k(X, t) dW_k`, the Milstein
//! step adds to Euler–Maruyama the first-order Itô–Taylor correction
//!
//! ```text
//!   X_k^{n+1} = X_k^n + f_k · h + g_k · ΔW_k
//!                     + ½ g_k · (∂g_k/∂X_k) · (ΔW_k² − h),
//! ```
//!
//! where `ΔW_k ~ N(0, h)` is the engine-drawn increment for component `k`. The
//! correction needs only the **diagonal** of the diffusion Jacobian `∂g_k/∂X_k`:
//! with independent per-component Wiener processes the off-diagonal Lévy-area
//! terms vanish, which is exactly why the diagonal-noise contract (ROADMAP §11)
//! makes order 1.0 reachable without simulating iterated stochastic integrals.
//!
//! Raising the strong order from 0.5 to 1.0 costs one extra thing over
//! Euler–Maruyama: the diffusion evaluator must carry its Jacobian. The kernel
//! therefore reports [`Caps::needs_jacobian`] `= true`, the signal to the
//! engine/Python layer to lower the diffusion tape with `∂g/∂u` (the
//! `with_diffusion_jacobian` path). For **additive** noise (`g' ≡ 0`) the
//! correction is zero and Milstein reduces to Euler–Maruyama.

use crate::{
    register_sde_kernel, Caps, Evaluator, ProblemKind, ProblemKinds, SolverState, StepOutcome,
};

use super::SdeKernel;

/// The Milstein diagonal-Itô kernel.
///
/// Beyond [`EulerMaruyama`](super::EulerMaruyama)'s buffers it keeps the
/// row-major `dim × dim` diffusion Jacobian `∂g/∂u` (only its diagonal is read).
/// All buffers grow on the first step and are reused.
#[derive(Default)]
pub struct Milstein {
    /// Drift `f(u, t)` for the current step.
    f: Vec<f64>,
    /// Diagonal diffusion `g(u, t)` for the current step.
    g: Vec<f64>,
    /// Row-major `dim × dim` diffusion Jacobian `∂g/∂u`; only `∂g_k/∂u_k` is used.
    gjac: Vec<f64>,
    /// Scratch for the drift evaluator.
    drift_scratch: Vec<f64>,
    /// Scratch for the diffusion evaluator.
    diff_scratch: Vec<f64>,
}

impl Milstein {
    /// A fresh kernel with unallocated buffers (sized on the first step).
    pub fn new() -> Self {
        Milstein::default()
    }

    fn ensure(&mut self, dim: usize, drift: &dyn Evaluator, diffusion: &dyn Evaluator) {
        if self.f.len() != dim {
            self.f = vec![0.0; dim];
            self.g = vec![0.0; dim];
            self.gjac = vec![0.0; dim * dim];
        }
        if self.drift_scratch.len() != drift.n_scratch() {
            self.drift_scratch = vec![0.0; drift.n_scratch()];
        }
        if self.diff_scratch.len() != diffusion.n_scratch() {
            self.diff_scratch = vec![0.0; diffusion.n_scratch()];
        }
    }
}

impl SdeKernel for Milstein {
    fn name(&self) -> &'static str {
        "milstein"
    }

    fn caps(&self) -> Caps {
        // Explicit, but reads the diffusion Jacobian ∂g/∂u — flag it so the
        // engine/Python layer lowers the diffusion tape with its Jacobian.
        Caps {
            needs_jacobian: true,
            ..Caps::explicit(ProblemKinds::of(ProblemKind::Sde))
        }
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
        // A *hard* assert, not debug-only: Milstein reads ∂g/∂u, and whether the
        // diffusion tape carries one is the *caller's* lowering choice (the
        // `with_diffusion_jacobian` path), not an engine-controlled invariant. In
        // a release wheel a debug-only check would compile out and let
        // `eval_jac` run on a Jacobian-less evaluator — a caller bug (per the
        // `Evaluator` contract) that would silently corrupt the correction term
        // or panic deep in the VM. Fail loudly and early instead, matching the
        // engine's hard assert on a bad `dt`.
        assert!(
            diffusion.has_jacobian(),
            "Milstein requires the diffusion tape to carry ∂g/∂u \
             (lower with with_diffusion_jacobian=True)"
        );
        self.ensure(dim, drift, diffusion);

        drift.eval(u, p, *t, &mut self.drift_scratch, &mut self.f);
        // One pass yields g and ∂g/∂u together.
        diffusion.eval_jac(
            u,
            p,
            *t,
            &mut self.diff_scratch,
            &mut self.g,
            &mut self.gjac,
        );

        for k in 0..dim {
            let dwk = dw[k];
            let gk = self.g[k];
            // Diagonal entry ∂g_k/∂u_k of the row-major dim×dim Jacobian.
            let dgk = self.gjac[k * dim + k];
            u[k] += self.f[k] * h + gk * dwk + 0.5 * gk * dgk * (dwk * dwk - h);
        }
        *t += h;
        StepOutcome::Accepted { h_next: h }
    }
}

register_sde_kernel!(
    "milstein",
    Caps {
        needs_jacobian: true,
        ..Caps::explicit(ProblemKinds::of(ProblemKind::Sde))
    },
    || Box::new(Milstein::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sde::testkit::{
        ConstDiffusion, EulerMaruyamaRef, LinearDrift, ProportionalDiffusion,
    };
    use crate::sde::EulerMaruyama;

    #[test]
    fn caps_flag_the_diffusion_jacobian() {
        let s = Milstein::new();
        assert_eq!(s.name(), "milstein");
        let c = s.caps();
        assert!(c.supports(ProblemKind::Sde));
        assert!(c.needs_jacobian, "Milstein reads ∂g/∂u");
    }

    #[test]
    #[should_panic(expected = "diffusion tape to carry")]
    fn jacobian_less_diffusion_fails_loudly_even_in_release() {
        // The release-mode safety net: a Milstein kernel handed a diffusion
        // evaluator with no Jacobian must abort with a clear error, not read
        // `eval_jac` and produce silent garbage. `LinearDrift` reports
        // `has_jacobian() == false`, so it stands in for a no-Jacobian diffusion.
        let drift = LinearDrift::new(1.0, 0.0);
        let no_jac_diffusion = LinearDrift::new(0.5, 0.0);
        let mut st = SolverState {
            u: vec![1.0],
            t: 0.0,
            p: vec![],
            scratch: vec![],
        };
        Milstein::new().step(&drift, &no_jac_diffusion, &mut st, &[0.1], 0.01);
    }

    #[test]
    fn step_applies_the_exact_milstein_formula_for_multiplicative_noise() {
        // GBM: dX = μX dt + σX dW, so g(x) = σx, g'(x) = σ. The Milstein
        // correction ½ g g' (ΔW² − h) = ½ (σX)(σ)(ΔW² − h) is exercised.
        let (m, sigma) = (0.05, 0.3);
        let drift = LinearDrift::new(-m, 0.0); // f(x) = m x
        let diffusion = ProportionalDiffusion::new(sigma); // g(x) = σx, g' = σ
        let mut s = Milstein::new();

        let mut st = SolverState {
            u: vec![2.0],
            t: 0.0,
            p: vec![],
            scratch: vec![],
        };
        let dw = [0.25];
        let h = 0.05;
        s.step(&drift, &diffusion, &mut st, &dw, h);

        let x0 = 2.0;
        let g = sigma * x0;
        let gprime = sigma;
        let want = x0 + m * x0 * h + g * dw[0] + 0.5 * g * gprime * (dw[0] * dw[0] - h);
        assert!(
            (st.u[0] - want).abs() < 1e-15,
            "u = {}, want {want}",
            st.u[0]
        );
    }

    #[test]
    fn additive_noise_reduces_to_euler_maruyama() {
        // g(x) = σ constant ⇒ g' = 0 ⇒ the Milstein correction vanishes, so a
        // Milstein step must equal an Euler–Maruyama step bit-for-bit.
        let (theta, mu, sigma) = (1.2, -0.5, 0.4);
        let drift = LinearDrift::new(theta, mu);
        let diffusion = ConstDiffusion::new(sigma);

        let dw = [0.31, -0.22];
        let h = 0.07;

        let mut mil = Milstein::new();
        let mut st_mil = SolverState {
            u: vec![0.3, 1.1],
            t: 0.0,
            p: vec![],
            scratch: vec![],
        };
        mil.step(&drift, &diffusion, &mut st_mil, &dw, h);

        let mut em = EulerMaruyama::new();
        let mut st_em = SolverState {
            u: vec![0.3, 1.1],
            t: 0.0,
            p: vec![],
            scratch: vec![],
        };
        em.step(&drift, &diffusion, &mut st_em, &dw, h);

        assert_eq!(
            st_mil.u[0].to_bits(),
            st_em.u[0].to_bits(),
            "additive noise: Milstein ≠ EM"
        );
        assert_eq!(st_mil.u[1].to_bits(), st_em.u[1].to_bits());
    }

    #[test]
    fn correction_is_the_only_difference_from_euler_maruyama() {
        // For multiplicative noise the Milstein update minus the plain EM update
        // must be exactly the correction term ½ g g' (ΔW² − h), per component.
        let drift = LinearDrift::new(0.0, 0.0); // f ≡ 0, isolate the noise terms
        let sigma = 0.5;
        let diffusion = ProportionalDiffusion::new(sigma);
        let dw = [0.4, -0.15];
        let h = 0.03;

        let mut st_mil = SolverState {
            u: vec![1.5, 0.8],
            t: 0.0,
            p: vec![],
            scratch: vec![],
        };
        Milstein::new().step(&drift, &diffusion, &mut st_mil, &dw, h);

        // EulerMaruyamaRef is the formula written out by hand (no Jacobian).
        let u_em = EulerMaruyamaRef::step(&drift, &diffusion, &[1.5, 0.8], &dw, h);

        for k in 0..2 {
            let g = sigma * [1.5, 0.8][k];
            let correction = 0.5 * g * sigma * (dw[k] * dw[k] - h);
            assert!(
                (st_mil.u[k] - (u_em[k] + correction)).abs() < 1e-15,
                "component {k}: {} vs {}",
                st_mil.u[k],
                u_em[k] + correction
            );
        }
    }
}
