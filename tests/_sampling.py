"""
Curated test samples and per-system test inputs.

The bulk suite is registry-driven (see ``conftest.py``); this module holds the
hand-picked subsets and inputs that cannot be derived from the registry:

- ``INTEGRATION_SAMPLE`` — representative ODE systems compiled + integrated in
  the regular ``slow`` tier.  The full 118-system sweep runs nightly under
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

# ODE categories excluded from the per-system correctness sweeps because they are
# high-dimensional method-of-lines PDE *fields* (a flattened 2-D grid → ~1k-5k
# coupled ODEs).  At the catalogue default grid, lowering the symbolic tape alone
# costs seconds-to-tens-of-seconds and a full integration / pure-Python reference
# trajectory is intractable (and the dense symbolic autogen Jacobian does not
# scale at all).  They are viz-only demos, fully exercised by the viz field-movie
# tests on small grids (N=8) — which lower + integrate the same `_equations` on the
# engine.  Consumers that skip these categories: the INTEGRATION_SAMPLE
# category-coverage guard (test_registry) and the `reference == engine` RHS leg
# (test_xval_catalogue leg 3).
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


DDE_HISTORIES: dict[str, object] = {
    "MackeyGlass": _mg_history,
    "IkedaDelay": _ikeda_history,
    "SprottDelay": _sprott_history,
    "ScrollDelay": _scroll_history,
    "PiecewiseCircuit": _piece_history,
}


# --- SDE samples (per-system seed + in-basin ic for the stochastic sweep) ----
# The diagonal-Itô analogue of DDE_HISTORIES.  Empty today: there are no
# built-in SDE systems yet (the registry now detects the ``sde`` family, but the
# catalogue under ``systems/`` has none).  When a built-in StochasticSystem
# lands, add ``"<Name>": {"seed": <int>, "ic": [...]}`` here; a guard test in
# ``test_registry.py`` asserts this stays complete against the ``sde`` family.
SDE_SAMPLES: dict[str, dict] = {}
