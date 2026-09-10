"""
Curated test samples and per-system test inputs.

The bulk suite is registry-driven (see ``conftest.py``); this module holds the
hand-picked subsets and inputs that cannot be derived from the registry:

- ``INTEGRATION_SAMPLE`` — representative ODE systems compiled + integrated in
  the regular ``slow`` tier.  The full 120-system sweep runs nightly under
  ``-m full``.  A guard test in ``test_registry.py`` enforces that every ODE
  category keeps at least two representatives here.
- ``MAP_LYAPUNOV_EXCLUDE`` — maps whose Jacobian is singular or undefined
  along the orbit, excluded from spectrum-shape tests.
- ``DDE_HISTORIES`` — non-equilibrium history callables per DDE system
  (a constant history at a fixed point makes every exponent ≈ 0).  A guard
  test asserts completeness, so adding a DDE without a history fails loudly.
- ``SDE_SAMPLES`` — per-SDE-system integration inputs (a reproducible ``seed``
  and an in-basin ``ic``) for the registry-driven stochastic sweep, the
  diagonal-Itô analogue of ``DDE_HISTORIES``.  Empty today (no built-in SDE
  systems yet); a guard test keeps it in lock-step with the ``sde`` family so a
  future built-in SDE without an entry fails loudly.
- ``DYNAMICS_ICS`` — on-attractor initial conditions for the universal
  per-system dynamical gate (``tests/test_catalogue_dynamics.py``).  The third
  member of the ``DDE_HISTORIES`` / ``SDE_SAMPLES`` family: a per-system test
  *input* that cannot be derived from the registry, here because the system's
  basin does not contain the gate's generic starting point.
"""

from __future__ import annotations

import numpy as np

# --- ODE integration sample (slow tier) ------------------------------------
# Oregonator is excluded: stiff, needs very tight tolerances.
INTEGRATION_SAMPLE: list[str] = [
    # chaotic_attractors
    "Lorenz",
    "Rossler",
    "Halvorsen",
    "HyperRossler",
    "SprottA",
    # chem_bio_systems
    "HindmarshRose",
    "CircadianRhythm",
    "ForcedVanDerPol",
    # climate_geophysics
    "RayleighBenard",
    "ArnoldBeltramiChildress",
    # coupled_systems
    "Chen",
    "LuChen",
    # exotic_systems
    "HyperCai",
    "HyperBao",
    # oscillatory_systems
    "ShimizuMorioka",
    "Aizawa",
    "Torus",
    "Lissajous2D",
    # physical_systems
    "DoublePendulum",
    "Colpitts",
    "Laser",
    # population_dynamics
    "Finance",
    "CoevolvingPredatorPrey",
    # NOTE: the `spatial_fields` category (GrayScott, SwiftHohenberg) is
    # intentionally absent — see HEAVY_FIELD_CATEGORIES below.
]

# ODE categories excluded from the per-system integration / cross-validation sweeps
# because they are high-dimensional method-of-lines PDE *fields* (a flattened 2-D
# grid → ~1k-5k coupled ODEs).  The pure-Python *reference* trajectory leg
# (test_xval_catalogue leg 4) steps them in interpreted Python (intractable), and
# the per-state `reference == engine` RHS leg (leg 3) re-lowers the tape for every
# sample state (tens of seconds at 1k-5k states).  Engine lowering + integration
# themselves are fast (a couple of seconds — the field movies are quick), so these
# systems are fully exercised by the small-grid viz field tests (N=8), which lower
# + integrate the same `_equations` on the engine.  Skipped by: the
# INTEGRATION_SAMPLE coverage guard (test_registry), the reference==engine leg
# (test_xval_catalogue leg 3), and the nightly full integration sweep.
HEAVY_FIELD_CATEGORIES: set[str] = {"spatial_fields"}

# --- ODE systems excluded from the exhaustive integration sweeps ------------
# Systems that cannot be integrated by adaptive solvers in bounded time would
# hang the nightly full sweep rather than fail it, so they are skipped here.
# An extension hook: currently empty (the whole catalogue integrates).
HARD_TO_INTEGRATE: dict[str, str] = {}

# --- Maps excluded from Lyapunov-spectrum shape tests -----------------------
MAP_LYAPUNOV_EXCLUDE: dict[str, str] = {
    "Bogdanov": "singular Jacobian at the origin",
    "Ulam": "orbit visits points where the Jacobian degenerates",
    "Gingerbreadman": "piecewise-linear; sign(x) Jacobian undefined at kinks",
}


# --- DDE histories (non-equilibrium, drive the dynamics) --------------------
def _mg_history(s: float) -> list[float]:
    return [1.0 + 0.1 * np.sin(0.2 * s)]


def _ikeda_history(s: float) -> list[float]:
    return [0.1 + 0.05 * np.cos(0.3 * s)]


def _sprott_history(s: float) -> list[float]:
    return [0.5 + 0.1 * np.sin(0.2 * s)]


def _scroll_history(s: float) -> list[float]:
    return [0.3 + 0.05 * np.cos(0.1 * s)]


def _piece_history(s: float) -> list[float]:
    return [0.4 + 0.05 * np.sin(0.15 * s)]


def _voss_history(s: float) -> list[float]:
    return [0.15 + 0.1 * np.sin(0.2 * s)]


DDE_HISTORIES: dict[str, object] = {
    "MackeyGlass": _mg_history,
    "IkedaDelay": _ikeda_history,
    "SprottDelay": _sprott_history,
    "ScrollDelay": _scroll_history,
    "PiecewiseCircuit": _piece_history,
    "VossDelay": _voss_history,
}


# --- SDE samples (per-system seed + in-basin ic for the stochastic sweep) ----
# The diagonal-Itô analogue of DDE_HISTORIES.  Every built-in StochasticSystem
# needs a ``"<Name>": {"seed": <int>, "ic": [...]}`` entry here; a guard test in
# ``test_registry.py`` (``test_sde_samples_complete``) asserts this stays
# complete against the ``sde`` family.
SDE_SAMPLES: dict[str, dict] = {
    "OrnsteinUhlenbeck": {"seed": 0, "ic": [2.0]},
    "GeometricBrownianMotion": {"seed": 1, "ic": [1.0]},
    "DoubleWell": {"seed": 2, "ic": [-1.0]},
}


# --- On-attractor ICs for the universal dynamical gate ----------------------
# ``tests/test_catalogue_dynamics.py`` needs one deterministic, in-basin start
# per system.  Its generic rule (``default_ic`` when the class declares one,
# else a seeded draw from a small ball about the origin) covers 169 of the 177
# catalogue systems; the eight below have a basin that ball misses, so the gate
# would otherwise be asserting things about an escaping transient rather than
# about the attractor the system claims.  Each entry says why, and
# ``test_catalogue_dynamics.test_reference_ic_overrides_are_all_live`` fails if
# one is no longer needed.
#
# This is a per-system test *input*, exactly like ``DDE_HISTORIES`` — it does
# not weaken any assertion.  Every system here is still held to the full gate
# (bounded, non-degenerate, revisiting, claim-consistent) from this IC.
DYNAMICS_ICS: dict[str, list[float]] = {
    # Population / kinetic models: the state is a vector of abundances or
    # concentrations, so the attractor lives strictly inside the positive
    # orthant and a ball about the origin straddles the (unphysical) boundary.
    "TurchinHanski": [0.5, 0.5, 0.0],
    "CoevolvingPredatorPrey": [0.5, 0.3, 0.2],
    "HastingsPowell": [0.8, 0.2, 8.0],
    # Sprott's minimal chaotic flows have famously small basins: cases L and the
    # jerk system escape from almost everywhere outside them.  SprottJerk's
    # basin is tiny enough that (0, 0, 1) — Sprott's own start — is the only one
    # of a dozen candidates tried that stays bounded.
    "SprottL": [0.1, 0.1, 0.1],
    "SprottJerk": [0.0, 0.0, 1.0],
    # SprottE's x = y = 0 axis is invariant (`x' = yz`, `y' = x^2 - y`), and the
    # generic ball is close enough to it that the orbit creeps along the axis
    # with z ramping instead of reaching the attractor: lambda_1 = -3.8e-4 from
    # the ball, +0.071 to +0.086 from every other start tried.
    "SprottE": [0.1, 0.1, 0.1],
    # A three-population cancer model: from the generic ball the orbit is bounded
    # and recurrent but lands on a *fixed point* (lambda_1 = -4.39), not on the
    # strange attractor the docstring describes (lambda_1 = +0.015 from here).
    "ItikBanksTumor": [0.1, 0.1, 0.1],
    # A six-variable model of the cell-division oscillator: from the near-origin
    # ball the run never reaches it.  This is the start the docs page uses.
    # (GlycolyticOscillation and WindmiReduced look like the same case but are
    # not: they only needed a longer window — see DYNAMICS_WINDOWS — and the
    # liveness guard rejects a redundant entry here.)
    "CellCycle": [0.01, 0.01, 0.01, 0.01, 0.9, 0.9],
}

# --- Longer reference windows for slow-settling systems ---------------------
# The dynamical gate discards the first half of a 100-time-unit run as
# transient, which is many Lyapunov times for almost every catalogue flow.  A
# handful of models have an intrinsically slower approach (a three-level food
# chain relaxes on the top predator's demographic timescale), so their transient
# has not finished at t = 50 and the gate would be measuring the approach rather
# than the attractor.  ``{name: (final_time, dt)}``.
# ``test_catalogue_dynamics.test_reference_windows_are_all_live`` fails if an
# entry is no longer needed, so this cannot become a place to hide a defect.
DYNAMICS_WINDOWS: dict[str, tuple[float, float]] = {
    # lambda_1 ~ 0.010, so 100 units is only one Lyapunov time; measured
    # growth 6.68 at T=100 -> 1.05 at T=500 (settled).
    "HastingsPowell": (500.0, 0.05),
    # The tracer's radial coordinate is still spiralling outwards at t = 100
    # (return gap 1.21); by T=300 it is on its torus (return gap 7.4e-4) and
    # stays there (1.3e-3 at T=2400).
    "BlinkingRotlet": (300.0, 0.05),
    # Geophysical units: the jet speed is u = 6.3e-5, so 100 time units is
    # numerically indistinguishable from standing still.  The window (and the
    # matching coarse sampling) is the one the docs page integrates.
    "BickleyJet": (40_000.0, 16.0),
    # Slow kinetics / a slow substorm cycle: measured all-frozen at T=100 and
    # fully settled at these windows.
    "GlycolyticOscillation": (400.0, 0.05),
    "CellCycle": (600.0, 0.05),
    "WindmiReduced": (500.0, 0.05),
}
