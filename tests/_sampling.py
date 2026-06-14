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
]

# --- ODE systems excluded from the exhaustive integration sweeps ------------
# Systems that cannot be integrated by adaptive solvers in bounded time would
# hang the nightly full sweep rather than fail it, so they are skipped here.
# Currently empty: ``BlinkingRotlet`` (the former entry) only *looked*
# un-integrable because JiTCODE's simplify(ratio=1) codegen pass hangs on its
# large rational RHS — fixed with ``_compile_simplify = False`` on the class.
HARD_TO_INTEGRATE: dict[str, str] = {}

# Systems the diffsol backend specifically can't integrate (the default JiTCODE
# backend handles them fine). Excluded from the diffsol full-catalogue sweep.
# Currently empty: ``WindmiReduced`` (the former entry) now integrates on every
# diffsol solver after guarding the fractional power with ``abs(p)`` and
# clamping the steep ``tanh`` argument (autodiff-safe) — both no-ops on the
# physical trajectory.
DIFFSOL_SKIP: dict[str, str] = {}

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
