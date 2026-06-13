"""
Known-value Lyapunov tests, driven by ``known_lyapunov`` class metadata.

Any system that declares ``known_lyapunov`` (see ``SystemBase``) is checked
against its literature spectrum here — seeding the metadata on a new system
automatically adds it to this tier.  Supported metadata keys:

- ``spectrum`` + ``atol``: per-exponent comparison against literature values
- ``n_positive``: count of strictly positive exponents (for hyperchaos etc.)
- ``params``: parameter overrides (e.g. Logistic at r=4 for the exact ln 2)
- ``ic``: initial condition; ``kwargs``: forwarded to ``lyapunov_spectrum``
"""

from __future__ import annotations

import numpy as np
import pytest
from _sampling import DDE_HISTORIES

from tsdynamics import registry

_ENTRIES = [e for e in registry.all_systems() if e.known_lyapunov]
_IDS = [e.name for e in _ENTRIES]


def _compute_spectrum(entry) -> tuple[np.ndarray, dict]:
    meta = dict(entry.known_lyapunov)
    kwargs = dict(meta.get("kwargs", {}))
    overrides = meta.get("params")
    sys = entry.cls(params=dict(overrides)) if overrides else entry.cls()

    if entry.family == "dde":
        # DDE Lyapunov needs a state on the attractor: integrate from a
        # non-equilibrium history first, then seed with the final state.
        history = DDE_HISTORIES[entry.name]
        traj = sys.integrate(final_time=200.0, dt=0.5, history=history, rtol=1e-4, atol=1e-4)
        kwargs.setdefault("ic", traj.y[-1])
    elif meta.get("ic") is not None:
        kwargs.setdefault("ic", list(meta["ic"]))

    return sys.lyapunov_spectrum(**kwargs), meta


@pytest.mark.slow
@pytest.mark.parametrize("entry", _ENTRIES, ids=_IDS)
def test_known_lyapunov_values(entry) -> None:
    spectrum, meta = _compute_spectrum(entry)
    assert np.all(np.isfinite(spectrum))

    if "spectrum" in meta:
        expected = np.asarray(meta["spectrum"], dtype=float)
        atol = np.asarray(meta.get("atol", 0.1), dtype=float)
        assert spectrum.shape == expected.shape
        deviation = np.abs(spectrum - expected)
        assert np.all(deviation <= atol), (
            f"{entry.name}: spectrum {np.round(spectrum, 4)} deviates from "
            f"literature {expected} by {np.round(deviation, 4)} (atol {atol}). "
            f"Source: {meta.get('source', 'n/a')}"
        )
        # Full literature spectra are given in descending order.
        assert np.all(np.diff(expected) <= 0)

    if "n_positive" in meta:
        # A hyperchaotic system's smallest positive exponent often sits right
        # at zero, so a finite-time estimate can land marginally negative
        # (e.g. HyperBao's 2nd exponent ~ +0.01 estimated as -0.009). Count an
        # exponent as positive when it clears a small near-zero band so the
        # check tracks the qualitative count, not estimator noise at zero.
        zero_band = meta.get("zero_band", 0.02)
        n_pos = int(np.sum(spectrum > -zero_band))
        assert n_pos >= meta["n_positive"], (
            f"{entry.name}: expected >= {meta['n_positive']} positive exponents "
            f"(within ±{zero_band} of zero), got {n_pos} in {np.round(spectrum, 4)}"
        )


# ---------------------------------------------------------------------------
# Behavioural one-offs that metadata cannot express
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_lorenz_spectrum_is_dissipative_and_sorted() -> None:
    import tsdynamics as ts

    lor = ts.Lorenz(ic=[1.0, 1.0, 1.0])
    spec = lor.lyapunov_spectrum(dt=0.1, burn_in=50.0, final_time=200.0)
    assert spec[0] >= spec[1] >= spec[2]
    # divergence of Lorenz = -(sigma + 1 + beta) ≈ -13.67
    assert -20.0 < spec.sum() < -5.0


@pytest.mark.slow
def test_lorenz_partial_spectrum_n_exp_2() -> None:
    import tsdynamics as ts

    lor = ts.Lorenz(ic=[1.0, 1.0, 1.0])
    exps = lor.lyapunov_spectrum(dt=0.1, burn_in=30.0, final_time=100.0, n_exp=2)
    assert exps.shape == (2,)
    assert exps[0] > 0.0


@pytest.mark.slow
def test_logistic_stable_regime_negative_exponent() -> None:
    """Logistic at r=2 sits on a stable fixed point: LE < 0."""
    import tsdynamics as ts

    m = ts.Logistic(params={"r": 2.0})
    exps = m.lyapunov_spectrum(steps=5_000)
    assert exps[0] < 0.0


@pytest.mark.slow
def test_mackeyglass_two_exponents_finite() -> None:
    import tsdynamics as ts

    mg = ts.MackeyGlass()
    traj = mg.integrate(
        final_time=200.0,
        dt=0.5,
        history=DDE_HISTORIES["MackeyGlass"],
        rtol=1e-4,
        atol=1e-4,
    )
    exps = mg.lyapunov_spectrum(
        n_exp=2,
        dt=0.5,
        burn_in=50.0,
        final_time=300.0,
        ic=traj.y[-1],
        rtol=1e-4,
        atol=1e-4,
    )
    assert exps.shape == (2,)
    assert np.all(np.isfinite(exps))
