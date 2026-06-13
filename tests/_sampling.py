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
]

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
