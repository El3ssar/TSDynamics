"""Shared, frozen benchmark configuration — the single source of truth.

Every adapter (TSDynamics, SciPy, pynamical, nolds, nolitsa, …) and the external
DynamicalSystems.jl Julia script read these *same* numbers, so a cross-library
comparison measures the libraries, not divergent problem set-ups. The Python
adapters import this module directly; the Julia script reads the JSON dump that
:func:`dump_config` writes (so Julia and Python use byte-identical parameters).

All constants live here. Nothing in this module imports a heavy dependency, so it
is cheap to import from the Julia-config dumper or a bare environment probe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Canonical systems & their parameters (the classics every library ships)
# --------------------------------------------------------------------------- #

# Lorenz (1963): the reference chaotic ODE. Classic parameters.
LORENZ_PARAMS: dict[str, float] = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}
LORENZ_IC: list[float] = [1.0, 1.0, 1.0]

# Rössler (1976): used for the Poincaré-section task (clean single-funnel band).
ROSSLER_PARAMS: dict[str, float] = {"a": 0.2, "b": 0.2, "c": 5.7}
ROSSLER_IC: list[float] = [1.0, 1.0, 1.0]

# Hénon (1976): the reference chaotic map. Classic parameters.
HENON_PARAMS: dict[str, float] = {"a": 1.4, "b": 0.3}
HENON_IC: list[float] = [0.1, 0.1]

# Logistic map: the bifurcation-diagram workhorse.
LOGISTIC_R_MIN: float = 2.8
LOGISTIC_R_MAX: float = 4.0
LOGISTIC_N_RATES: int = 1000  # number of parameter values swept
LOGISTIC_N_GENS: int = 100  # generations kept per rate
LOGISTIC_N_DISCARD: int = 200  # transient generations discarded per rate
LOGISTIC_IC: float = 0.5

# --------------------------------------------------------------------------- #
# Integration grids (speed) and accuracy probe (precision)
# --------------------------------------------------------------------------- #

DT: float = 0.01  # uniform output step for the speed integrations
T_SHORT: float = 100.0  # "short" integration horizon
T_LONG: float = 10_000.0  # "long" integration horizon

# Accuracy probe: integrate to a modest horizon with a tight tolerance and a
# high-order adaptive method, then compare every library's final state to an
# independent ultra-tight reference. Kept short (a few Lyapunov times) so the
# comparison is still meaningful for the chaotic Lorenz flow.
T_ACC: float = 8.0
ACC_RTOL: float = 1e-10
ACC_ATOL: float = 1e-10
ACC_REF_RTOL: float = 1e-13
ACC_REF_ATOL: float = 1e-13

# --------------------------------------------------------------------------- #
# From-data analysis (correlation dimension, Lyapunov-from-data)
# --------------------------------------------------------------------------- #

# A long Lorenz x(t) scalar series feeds the data-driven estimators. Every such
# library gets the SAME series and the SAME embedding parameters.
SERIES_FINAL_TIME: float = 250.0
SERIES_DT: float = 0.01
SERIES_TRANSIENT_SAMPLES: int = 5_000  # drop ~50 time units of transient
SERIES_MAX_SAMPLES: int = 15_000  # cap the series so every lib finishes

# Delay-embedding parameters shared across the from-data estimators.
EMBED_DIM: int = 5
EMBED_DELAY: int = 11  # samples (~0.11 time units, near the first ACF minimum)
THEILER: int = 100  # Theiler window (samples) excluding temporal neighbours

# Correlation dimension: nolds' estimator is O(N²) in the series length, so the
# point set is capped to a size every library can finish in seconds.
CORR_N: int = 5_000

# Lyapunov-from-data: the dt=0.01 series oversamples the divergence curve, so the
# estimators see a stride-downsampled series; LYAP_DELAY is in *downsampled*
# samples, LYAP_THEILER likewise.
LYAP_STRIDE: int = 5
LYAP_N: int = 3_000
LYAP_DELAY: int = 3
LYAP_THEILER: int = 20

# --------------------------------------------------------------------------- #
# Basins of attraction — the Newton root-finding map on z**3 - 1 (three basins).
# Defined identically in every adapter (it is not a catalogue system anywhere).
# Iterating Newton's method on f(z)=z**3-1 in the complex plane converges to one
# of the three cube roots of unity; the plane splits into a fractal 3-colouring.
# --------------------------------------------------------------------------- #

NEWTON_GRID_MIN: float = -1.5
NEWTON_GRID_MAX: float = 1.5
NEWTON_GRID_RES: int = 200  # NEWTON_GRID_RES**2 initial conditions
NEWTON_MAX_STEPS: int = 200

# --------------------------------------------------------------------------- #
# Complexity / from-data analysis (the expanded task set: entropy, DFA/Hurst,
# RQA, embedding-dimension, surrogates). Every library that supports a task gets
# the SAME input + parameters.
# --------------------------------------------------------------------------- #

ENTROPY_N: int = 3000  # Lorenz x samples for sample/permutation/multiscale entropy
ENTROPY_M: int = 2  # entropy embedding dimension (m); permutation order is M+1
RQA_N: int = 1200  # RQA is O(N²) — a shorter window every library can finish
RQA_EMBED_DIM: int = 3
RQA_EMBED_DELAY: int = 5
RQA_RECURRENCE_RATE: float = 0.05
DFA_N: int = 8000  # white-noise length for DFA/Hurst (more = steadier α)
EMBED_TARGET_DELAY: int = 8  # delay (samples) for the FNN/Cao embedding-dim task
EMBED_MAX_DIM: int = 10

# --------------------------------------------------------------------------- #
# Reference values from the literature (for the precision tables)
# --------------------------------------------------------------------------- #

# Lorenz Lyapunov spectrum (Sprott, "Chaos and Time-Series Analysis", 2003):
# λ ≈ (0.9056, 0, -14.5723); maximal exponent ≈ 0.9056 (base-e, per unit time).
LORENZ_LAMBDA_MAX: float = 0.9056
LORENZ_LAMBDA_SPECTRUM: list[float] = [0.9056, 0.0, -14.5723]

# Lorenz correlation dimension D2 ≈ 2.05 (Grassberger & Procaccia, 1983).
LORENZ_D2: float = 2.05

# Hénon (a=1.4, b=0.3) maximal Lyapunov exponent ≈ 0.419 (base-e, per iteration).
HENON_LAMBDA_MAX: float = 0.419

# --------------------------------------------------------------------------- #
# Timing protocol
# --------------------------------------------------------------------------- #

REPEAT: int = 5  # best-of-N (minimum) timing samples
REPEAT_QUICK: int = 2


def henon_fixed_points(a: float, b: float) -> list[tuple[float, float]]:
    """Analytic fixed points of the Hénon map (for the precision comparison).

    The Hénon map ``x' = 1 - a x^2 + y``, ``y' = b x`` has fixed points where
    ``y = b x`` and ``x = 1 - a x^2 + b x``, i.e. ``a x^2 + (1-b) x - 1 = 0``.

    Parameters
    ----------
    a, b : float
        Hénon parameters.

    Returns
    -------
    list of (float, float)
        The (up to two) real fixed points, sorted by ``x``.
    """
    import math

    disc = (1.0 - b) ** 2 + 4.0 * a
    if disc < 0:
        return []
    roots = sorted((-(1.0 - b) + s * math.sqrt(disc)) / (2.0 * a) for s in (+1.0, -1.0))
    return [(x, b * x) for x in roots]


def as_dict() -> dict[str, Any]:
    """Return the full config as a plain dict (for the JSON dump Julia reads)."""
    return {
        "lorenz": {"params": LORENZ_PARAMS, "ic": LORENZ_IC},
        "rossler": {"params": ROSSLER_PARAMS, "ic": ROSSLER_IC},
        "henon": {"params": HENON_PARAMS, "ic": HENON_IC},
        "logistic": {
            "r_min": LOGISTIC_R_MIN,
            "r_max": LOGISTIC_R_MAX,
            "n_rates": LOGISTIC_N_RATES,
            "n_gens": LOGISTIC_N_GENS,
            "n_discard": LOGISTIC_N_DISCARD,
            "ic": LOGISTIC_IC,
        },
        "integration": {
            "dt": DT,
            "t_short": T_SHORT,
            "t_long": T_LONG,
            "t_acc": T_ACC,
            "acc_rtol": ACC_RTOL,
            "acc_atol": ACC_ATOL,
            "acc_ref_rtol": ACC_REF_RTOL,
            "acc_ref_atol": ACC_REF_ATOL,
        },
        "series": {
            "final_time": SERIES_FINAL_TIME,
            "dt": SERIES_DT,
            "transient_samples": SERIES_TRANSIENT_SAMPLES,
            "max_samples": SERIES_MAX_SAMPLES,
            "embed_dim": EMBED_DIM,
            "embed_delay": EMBED_DELAY,
            "theiler": THEILER,
            "corr_n": CORR_N,
            "lyap_stride": LYAP_STRIDE,
            "lyap_n": LYAP_N,
            "lyap_delay": LYAP_DELAY,
            "lyap_theiler": LYAP_THEILER,
            "entropy_n": ENTROPY_N,
            "entropy_m": ENTROPY_M,
            "rqa_n": RQA_N,
            "rqa_embed_dim": RQA_EMBED_DIM,
            "rqa_embed_delay": RQA_EMBED_DELAY,
            "rqa_recurrence_rate": RQA_RECURRENCE_RATE,
            "dfa_n": DFA_N,
            "embed_target_delay": EMBED_TARGET_DELAY,
            "embed_max_dim": EMBED_MAX_DIM,
        },
        "newton": {
            "grid_min": NEWTON_GRID_MIN,
            "grid_max": NEWTON_GRID_MAX,
            "grid_res": NEWTON_GRID_RES,
            "max_steps": NEWTON_MAX_STEPS,
        },
        "references": {
            "lorenz_lambda_max": LORENZ_LAMBDA_MAX,
            "lorenz_lambda_spectrum": LORENZ_LAMBDA_SPECTRUM,
            "lorenz_d2": LORENZ_D2,
            "henon_lambda_max": HENON_LAMBDA_MAX,
            # x-coordinate of the Hénon saddle fixed point on the attractor
            # (positive root of a x² + (1-b) x − 1 = 0).
            "henon_fp_x": henon_fixed_points(HENON_PARAMS["a"], HENON_PARAMS["b"])[-1][0],
            # White noise has DFA exponent α = 0.5 and Hurst H = 0.5 (no long-range
            # correlation) — the ground truth for the DFA / Hurst tasks.
            "dfa_alpha": 0.5,
            "hurst_exp": 0.5,
            # Lorenz attractor's embedding dimension ≈ 3 (it lives in 3-D); the FNN/
            # Cao estimate is method-dependent (3–6), so this is a loose anchor.
            "lorenz_embed_dim": 3.0,
        },
        "timing": {"repeat": REPEAT, "repeat_quick": REPEAT_QUICK},
    }


def dump_config(path: str | Path) -> Path:
    """Write the config as JSON so the Julia script reads identical parameters.

    Parameters
    ----------
    path : str or Path
        Destination JSON path.

    Returns
    -------
    Path
        The written path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(as_dict(), indent=2))
    return p
