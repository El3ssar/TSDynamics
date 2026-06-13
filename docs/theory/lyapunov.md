---
description: The math behind the Lyapunov routines — variational equations, the QR/Benettin algorithm, time-weighted averaging, and the Kaplan–Yorke conjecture.
---

<span class="ts-kicker">Theory · 02</span>

# Lyapunov exponents

## Definition

For a flow $\dot{\mathbf{y}} = f(\mathbf{y})$, an infinitesimal
perturbation $\mathbf{w}$ evolves under the **variational equations**

$$
\dot{\mathbf{w}} = J(\mathbf{y}(t))\, \mathbf{w},
\qquad J_{ij} = \frac{\partial f_i}{\partial y_j},
$$

linearized along the trajectory. The Lyapunov exponents are the long-time
average exponential growth rates of volumes spanned by such perturbations:

$$
\lambda_i = \lim_{t \to \infty} \frac{1}{t} \ln \frac{\|\mathbf{w}_i(t)\|}{\|\mathbf{w}_i(0)\|},
$$

ordered $\lambda_1 \ge \lambda_2 \ge \dots$ For a map
$\mathbf{x}_{n+1} = f(\mathbf{x}_n)$ the same holds with the Jacobian
product $J(\mathbf{x}_{n-1}) \cdots J(\mathbf{x}_0)$ in place of the
flow's fundamental matrix. A positive $\lambda_1$ is the practical
definition of chaos; flows additionally carry one exactly-zero exponent
along the trajectory direction.

## The QR / re-orthonormalization algorithm

Naively evolving $k$ deviation vectors fails: all of them collapse onto
the fastest-growing direction, and their norms overflow. The standard fix
(Benettin, Galgani, Giorgilli & Strelcyn 1980) is to re-orthonormalize
periodically with a QR decomposition: after each interval, factor the
evolved frame $W = QR$, continue with $Q$, and accumulate
$\ln |R_{ii}|$ — the log of how much the $i$-th orthogonal direction
stretched. Then

$$
\lambda_i \approx \frac{1}{T} \sum_{\text{intervals}} \ln |R_{ii}|.
$$

Comparisons across method variants (Geist, Parlitz & Lauterborn 1990)
established this as the robust default, and it is what TSDynamics uses for
maps (`DiscreteMap.lyapunov_spectrum`, `TangentSystem`) with the exact
`_jacobian` at every iterate.

## What each family actually solves

- **ODEs** — JiTCODE differentiates the right-hand side symbolically and
  compiles state + variational equations into one C module
  (`jitcode_lyap`); the integrator returns *local* exponents per sampling
  interval. Because the adaptive stepper makes interval lengths uneven,
  TSDynamics averages them **weighted by interval duration** — an
  unweighted mean would bias the estimate toward whatever the step
  controller did. A `burn_in` discards the transient during which the
  deviation frame aligns with the attractor's Oseledets subspaces.

- **Maps** — pure QR as above, in a single forward pass alongside the
  trajectory.

- **DDEs** — the tangent space of a delay system is the
  **infinite-dimensional** history space $C([-\tau_{\max}, 0])$; there is
  a full spectrum of infinitely many exponents. `jitcdde_lyap`
  approximates the leading few by evolving perturbations of the history
  spline, with the same weighted averaging (weights come from the
  solver). This is why `n_exp` must be chosen consciously, why the
  estimates converge more slowly than ODE ones, and why `TangentSystem`
  refuses delay systems outright.

## The two-trajectory estimator

`max_lyapunov` implements the older and simpler estimator (Benettin,
Galgani & Strelcyn 1976): evolve the system and a copy displaced by
$d_0$, measure the separation $d$ after a short interval, accumulate
$\ln(d/d_0)$, renormalize the displacement back to $d_0$, repeat. It
needs no Jacobian at all — only the ability to step and to overwrite a
state — making it the right tool for non-smooth systems, at the price of
estimating only $\lambda_1$.

## The Kaplan–Yorke conjecture

Kaplan & Yorke (1979) conjectured that the information dimension of an
attractor equals the **Lyapunov dimension**

$$
D_{KY} = j + \frac{\sum_{i=1}^{j} \lambda_i}{|\lambda_{j+1}|},
\qquad j = \max\Big\{ k : \sum_{i=1}^{k} \lambda_i \ge 0 \Big\}
$$

— the dimension at which an interpolated volume neither grows nor
shrinks. For Lorenz, $D_{KY} \approx 2 + 0.906/14.57 \approx 2.06$.
`kaplan_yorke_dimension` computes exactly this, returning `0.0` when all
exponents are negative and `len(spectrum)` when the sum never turns
negative (the spectrum is incomplete — compute more exponents).

## References

- G. Benettin, L. Galgani, J.-M. Strelcyn, *Kolmogorov entropy and
  numerical experiments*, Phys. Rev. A **14**, 2338 (1976).
- G. Benettin, L. Galgani, A. Giorgilli, J.-M. Strelcyn, *Lyapunov
  characteristic exponents for smooth dynamical systems and for
  Hamiltonian systems; a method for computing all of them*, Meccanica
  **15**, 9–30 (1980).
- K. Geist, U. Parlitz, W. Lauterborn, *Comparison of different methods
  for computing Lyapunov exponents*, Prog. Theor. Phys. **83**, 875
  (1990).
- J. L. Kaplan, J. A. Yorke, *Chaotic behavior of multidimensional
  difference equations*, in Functional Differential Equations and
  Approximation of Fixed Points, Lecture Notes in Mathematics **730**,
  Springer (1979).

## See also

- [Analysis · Lyapunov spectra](../analysis/lyapunov.md) — the API these equations sit behind
- [Compilation pipeline](compilation.md) — how the variational equations get compiled
