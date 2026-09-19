"""The library's solver-tolerance defaults — the one place the numbers live.

Every ``rtol=`` / ``atol=`` default in TSDynamics is a name defined here.  Before
v6 the pair ``1e-6`` / ``1e-9`` was duplicated as bare literals across sixteen
call sites in five subpackages (``engine.run``, ``engine.events``,
``families.continuous``, ``derived.tangent``, ``derived._crossings``,
``analysis.basins.attractors``) plus the docstrings that quoted them.  That is a
defect in its own right — it is exactly how the defaults would silently drift
apart, and it made "what is the default tolerance?" a question with sixteen
independent answers.  Hoisting them here makes the answer singular and makes a
deliberate per-driver exception *visible* rather than accidental.

This module is deliberately a **leaf**: it imports nothing from
:mod:`tsdynamics`, so both the family layer (:mod:`tsdynamics.families`, which
imports the engine only lazily inside methods) and the engine layer
(:mod:`tsdynamics._engine.run`) can import it at module scope with no cycle.  It
sits beside :mod:`tsdynamics._utils.grids`, which hoists the output grid for the
same reason.

Why the ODE default is ``1e-9`` / ``1e-12``
-------------------------------------------
.. versionchanged:: 6.0
    The ODE default tightened from ``rtol=1e-6`` / ``atol=1e-9`` to
    ``rtol=1e-9`` / ``atol=1e-12``.

Before v6 the adaptive stepper had no dense output: it was *forced to land on
every requested output sample*, so a fine ``dt`` silently subsidised accuracy far
beyond what ``rtol`` asked for (Lorenz to ``T=10`` at ``rtol=1e-6`` delivered
``1.3e-3`` at ``dt=10`` but ``4.3e-10`` at ``dt=0.001`` — a ``3e6x`` spread, and
``rtol=1e-4``/``1e-6``/``1e-8``/``1e-10`` returned *bit-identical* arrays on a
fine grid).  v6 gave the kernels native continuous extensions, so ``dt`` is
honestly an output grid and ``rtol`` honestly sets accuracy — but a user who
never touched ``rtol`` therefore *lost* the subsidy.  Measured at the library
defaults (``dt=0.02``, ``T=5``, error at the final time versus SciPy ``DOP853``
at ``rtol=1e-13``/``atol=1e-16``), the old default now delivers e.g. Lorenz
``2.2e-4``, Thomas ``3.8e-4``, Halvorsen ``1.5e-3``, Chua ``1.1e-1``.

``1e-9``/``1e-12`` restores and exceeds the pre-v6 delivered accuracy — a
measured **median 1459x** improvement over a fifteen-system catalogue sample
(Lorenz ``2.2e-4 -> 1.9e-7``, Halvorsen ``1.5e-3 -> 2.9e-7``) — for a **median
1.74x** wall-clock cost on that sample.  Chaotic systems amplify integration
error exponentially, and they are this library's core subject, so the trade is
taken.  Users who want the old cost/accuracy point pass ``rtol=1e-6``
explicitly.

Why three drivers keep their own, looser number
-----------------------------------------------
The bump compensates for dense output, so it is only owed to the surfaces dense
output actually changed: the adaptive explicit kernels (``rk45``/``tsit5``/
``dop853``) sampling a grid with interior points.  Three internal drivers are
*not* in that blast radius, and for each the tighter tolerance was measured to
buy nothing while costing 2-3x.  Their defaults are therefore named here rather
than left to accidentally track the global one:

- **The DDE method of steps** (:data:`DDE_RTOL` / :data:`DDE_ATOL`) still lands
  on every output sample, so it never lost the subsidy.  Measured on all six
  built-in DDEs at the DDE default ``dt=0.02``, ``T=10``: five of six return a
  **bit-identical** final state at ``rtol=1e-3`` and at ``rtol=1e-9`` (the
  tolerance is inert because ``dt`` already bounds the step below the natural
  error), and the sixth (``IkedaDelay``) costs 1.4x.
- **The basin cell march** (:data:`BASIN_RTOL` / :data:`BASIN_ATOL`) runs
  thousands of two-node ``[t, t+dt]`` integrations per image, which dense output
  provably never touched (no interior sample).  Its output is a *topological*
  classification — which cell the orbit settles in — at a cell size many orders
  above ``1e-6``.  Measured: 2.27x (smooth two-well Duffing, 60x60) and 3.01x
  (fractal magnetic pendulum, 35x35 slice) slower for **0.00 %** of basin labels
  changing in either image.
- **The DDE Lyapunov estimator** (:data:`DDE_LYAPUNOV_RTOL` /
  :data:`DDE_LYAPUNOV_ATOL`) integrates the extended variational DDE on that same
  method-of-steps march.  Measured on all six built-in DDEs, ``1e-7``/``1e-9``
  and ``1e-9``/``1e-12`` agree to within the estimator's own finite-time scatter
  at identical cost.

The section-crossing march (:mod:`tsdynamics.derived._crossings`) is a fourth
non-blast-radius driver, but it forces the **fixed-step** ``rk4`` kernel, which
has no error control at all — so its tolerances are inert (measured: the bump
leaves a 400-crossing Rössler section **bit-identical** at 1.01x cost).  Having
nothing to trade, it simply follows the global default.
"""

from __future__ import annotations

__all__ = [
    "BASIN_ATOL",
    "BASIN_RTOL",
    "DDE_ATOL",
    "DDE_LYAPUNOV_ATOL",
    "DDE_LYAPUNOV_RTOL",
    "DDE_RTOL",
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
]

#: Default relative tolerance for every adaptive ODE solver surface —
#: ``integrate`` / ``run`` / ``ensemble`` / ``reinit`` + ``step`` / the events
#: seam / the ODE Lyapunov (tangent) engine / the section-crossing march.
#: Tightened from ``1e-6`` in v6; see the module docstring.
DEFAULT_RTOL: float = 1e-9

#: Default absolute tolerance, the companion of :data:`DEFAULT_RTOL`.  Kept three
#: decades below it (the released ratio), so a component decaying through zero is
#: still controlled absolutely rather than chasing a meaningless relative error.
#: Tightened from ``1e-9`` in v6.
DEFAULT_ATOL: float = 1e-12

#: Default relative tolerance for the DDE method of steps
#: (:class:`~tsdynamics.families.delay.DelaySystem`).  Deliberately looser than
#: :data:`DEFAULT_RTOL`: the DDE march lands on every output sample, so ``dt``
#: bounds the step and the tolerance is inert for five of the six built-in DDEs
#: (bit-identical results across six decades of ``rtol``).
DDE_RTOL: float = 1e-3

#: Default absolute tolerance for the DDE method of steps; see :data:`DDE_RTOL`.
DDE_ATOL: float = 1e-3

#: Tolerances for the engine DDE Lyapunov estimator
#: (:func:`tsdynamics.families._dde_lyapunov.dde_lyapunov_spectrum`), which
#: integrates the extended variational DDE on the same method-of-steps march.
DDE_LYAPUNOV_RTOL: float = 1e-7

#: Companion of :data:`DDE_LYAPUNOV_RTOL`.
DDE_LYAPUNOV_ATOL: float = 1e-9

#: Tolerances for the basin cell march — the per-``dt`` flow steps
#: :class:`~tsdynamics.analysis.basins.attractors._AttractorMapper` and its Rust
#: twin (:func:`tsdynamics._engine.run.basin_march`) drive per cell check.  The
#: two paths are contractually **bit-identical**, so these constants must be used
#: by both or the equivalence breaks.  Deliberately looser than
#: :data:`DEFAULT_RTOL`; see the module docstring for the measurement.
BASIN_RTOL: float = 1e-6

#: Companion of :data:`BASIN_RTOL`.
BASIN_ATOL: float = 1e-9


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
