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
growth rates of the tangent propagator. Let $\Phi(t)$ map an initial
perturbation to its value at time $t$, with $\Phi(0)=I$, and let
$\sigma_i(\Phi(t))$ be its ordered singular values. When the long-time limits
exist, the spectrum is

$$
\lambda_i = \lim_{t \to \infty} \frac{1}{t} \ln \sigma_i\!\left(\Phi(t)\right),
$$

ordered $\lambda_1 \ge \lambda_2 \ge \dots$ For a map
$\mathbf{x}_{n+1} = f(\mathbf{x}_n)$ the same holds with the Jacobian
product $J(\mathbf{x}_{n-1}) \cdots J(\mathbf{x}_0)$ in place of the
flow's fundamental matrix. A positive $\lambda_1$ is the practical
indicator of sensitive dependence, interpreted on a bounded invariant orbit.
A bounded non-equilibrium orbit of an autonomous flow has a neutral direction
along the trajectory. Equilibria and explicitly time-dependent forcing do not
have that guaranteed zero exponent. Every computed spectrum is a finite-time
estimate, so a small positive value alone need not establish asymptotic chaos.

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
maps (the compiled QR tangent-map kernel behind `TangentSystem`) with the exact
`_jacobian` at every iterate.

## What each family actually solves

- **ODEs** — the right-hand side is differentiated symbolically and the
  *extended* variational system (state ⊕ tangent vectors) is integrated on the
  Rust engine, QR re-orthonormalizing per chunk; the integrator returns *local*
  exponents per sampling interval. Because the adaptive stepper makes interval
  lengths uneven, TSDynamics averages them **weighted by interval duration** —
  an unweighted mean would bias the estimate toward whatever the step
  controller did. A `burn_in` discards the transient during which the
  deviation frame aligns with the attractor's Oseledets subspaces.

- **Maps** — pure QR as above, in a single forward pass alongside the
  trajectory.

- **DDEs** — the tangent space of a delay system is the
  **infinite-dimensional** history space $C([-\tau_{\max}, 0])$; there is
  a full spectrum of infinitely many exponents. The DDE estimator
  approximates the leading few by building an *extended* delay system — the base
  state plus deviation states whose dynamics are the symbolic variational
  equations — and integrating it on the engine in delay-window chunks, with a
  function-space QR over the deviation history segment. This is why `k` must
  be chosen consciously, why the estimates converge more slowly than ODE ones,
  and why `TangentSystem` refuses delay systems outright.

## The two-trajectory estimator

`lyapunov_spectrum` falls back to the older and simpler estimator when the
system carries no Jacobian to differentiate (Benettin,
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
`kaplan_yorke_dimension` evaluates this interpolation, returning `0.0` when all
exponents are negative. A complete finite-dimensional spectrum whose sum never
turns negative saturates at the state dimension. A truncated spectrum whose
cumulative sum stays nonnegative does not determine the crossing and cannot
support a dimension estimate; request more exponents. A delay system has
infinitely many directions, so its leading finite spectrum is always partial.
For an explicitly forced flow, include the forcing phase in an autonomous
extension before interpreting an attractor dimension.

The dimension is a conjectural relation to geometric properties of an attractor,
not a certificate of a measured fractal dimension. A raw exponent array cannot
carry its missing directions or forcing assumptions; preserve a
`LyapunovSpectrum` result when those checks matter.

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
