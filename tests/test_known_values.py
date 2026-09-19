"""
Known-value Lyapunov tests, driven by ``known_lyapunov`` class metadata.

Any system that declares ``known_lyapunov`` (see ``SystemBase``) is checked
against its literature spectrum here — seeding the metadata on a new system
automatically adds it to this tier.  Supported metadata keys:

- ``spectrum`` + ``atol``: per-exponent comparison against literature values
- ``n_positive``: the **exact** count of strictly positive exponents (chaos vs
  hyperchaos vs order) — see :func:`count_positive_exponents`
- ``params``: parameter overrides (e.g. Logistic at r=4 for the exact ln 2)
- ``ic``: initial condition; ``kwargs``: forwarded to ``lyapunov_spectrum``
"""

from __future__ import annotations

import numpy as np
import pytest
from _sampling import DDE_HISTORIES

import tsdynamics as ts
from tsdynamics import registry
from tsdynamics.analysis.results import LyapunovSpectrum

_ENTRIES = [e for e in registry.all_systems() if e.known_lyapunov]
_IDS = [e.name for e in _ENTRIES]

#: Relative half-width of the near-zero band used by
#: :func:`count_positive_exponents`, as a fraction of the spectrum's largest
#: magnitude.  A flow's zero exponent is only *numerically* near zero, and its
#: numerical size scales with the spectrum, so an absolute floor cannot serve
#: both Mackey-Glass (largest |lambda| = 0.003) and Chua (4.1).
_ZERO_BAND_FRACTION = 1e-3
_ZERO_BAND_FLOOR = 1e-6


def count_positive_exponents(spectrum, zero_band: float = 0.0) -> int:
    """Count the **strictly positive** exponents of ``spectrum``.

    An exponent counts as positive when it clears a near-zero band — the wider of
    ``zero_band`` (the per-system metadata override), an absolute floor, and a
    fraction of the spectrum's largest magnitude.  A flow's zero exponent and a
    marginally-negative estimate of it both fall *inside* the band, so they are
    not counted.

    This replaces a check that was **vacuous**: it counted every exponent above
    ``-zero_band``, i.e. every exponent that was not clearly negative, and then
    asserted the count was ``>=`` the expected one.  For any bounded 3-D flow the
    leading and the zero exponent both cleared that bar, so ``n_positive: 1``
    was satisfied by a limit cycle just as well as by a chaotic attractor.
    """
    exps = np.asarray(spectrum, dtype=float)
    scale = float(np.max(np.abs(exps))) if exps.size else 0.0
    band = max(float(zero_band), _ZERO_BAND_FLOOR, _ZERO_BAND_FRACTION * scale)
    return int(np.sum(exps > band))


def _compute_spectrum(entry) -> tuple[np.ndarray, dict]:
    meta = dict(entry.known_lyapunov)
    kwargs = dict(meta.get("kwargs", {}))
    overrides = meta.get("params")
    sys = entry.cls(params=dict(overrides)) if overrides else entry.cls()

    if entry.family == "dde":
        # DDE Lyapunov needs a state on the attractor: integrate from a
        # non-equilibrium history first, then seed with the final state.
        history = DDE_HISTORIES[entry.name]
        traj = sys.run(final_time=200.0, dt=0.5, history=history, rtol=1e-4, atol=1e-4)
        kwargs.setdefault("ic", traj.y[-1])
    elif meta.get("ic") is not None:
        kwargs.setdefault("ic", list(meta["ic"]))

    return ts.analysis.lyapunov_spectrum(sys, **kwargs), meta


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
        # The EXACT count, not ">=": the qualitative claim being made is "this
        # attractor is chaotic (1) / hyperchaotic (2)", and a spectrum with more
        # positive exponents than claimed falsifies it just as surely as one with
        # fewer. See count_positive_exponents for why ">=" over a signed band was
        # satisfied by any non-diverging flow.
        n_pos = count_positive_exponents(spectrum, meta.get("zero_band", 0.0))
        assert n_pos == meta["n_positive"], (
            f"{entry.name}: expected exactly {meta['n_positive']} positive exponents, "
            f"got {n_pos} in {np.round(spectrum, 4)}"
        )


# ---------------------------------------------------------------------------
# The counter itself must be able to FAIL (the old one could not)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spectrum", "expected"),
    [
        # A limit cycle: one exactly-zero exponent, the rest contracting. The
        # pre-fix counter scored this 1 and so PASSED an `n_positive: 1` claim.
        ([0.0, -1.0, -2.5], 0),
        # ...and so did a stable fixed point with a marginally-positive estimate
        # of a strictly negative exponent.
        ([-0.001, -1.0, -2.5], 0),
        ([0.906, -1e-4, -14.57], 1),  # Lorenz: chaotic
        ([0.43, 0.376, -3.3], 2),  # folded towel: hyperchaotic
        ([0.00278], 1),  # Mackey-Glass: tiny but genuinely positive
        ([0.0], 0),  # a single zero exponent is not positive
    ],
)
def test_count_positive_exponents_is_not_vacuous(spectrum, expected) -> None:
    """The positive-exponent counter separates order from chaos.

    Guards the fix for the vacuous ``n_positive`` check: every case here must
    score exactly ``expected``, so a regular orbit can no longer satisfy an
    ``n_positive >= 1`` claim.
    """
    assert count_positive_exponents(spectrum) == expected


def test_count_positive_exponents_honors_a_metadata_zero_band() -> None:
    """An explicit ``zero_band`` widens (never narrows) the near-zero band."""
    spectrum = [0.5, 0.01, -1.0]
    assert count_positive_exponents(spectrum) == 2
    assert count_positive_exponents(spectrum, zero_band=0.05) == 1


# ---------------------------------------------------------------------------
# The maximal exponent against the literature
#
# ``max_lyapunov`` was retired in v6: two doors onto one question answered it
# with two numbers (Hénon, one nominal horizon: 0.4233 against 0.4160), because
# ``n`` counted rescaling CYCLES at one door and ITERATIONS at the other.  The
# better half — the burn-in, and the Jacobian-free two-trajectory machine for a
# system with no RHS to differentiate — moved into ``lyapunov_spectrum``, so
# these properties are asserted of ``k=1`` now.  Every one of them still holds,
# and the literature agreement got BETTER: Lorenz 0.5 % where the old tolerance
# was 15 %.
#
# ``seed=`` is gone with it, and that is the point rather than an omission: the
# surviving path is the deterministic tangent/QR iteration, so there is no random
# perturbation direction to seed.  The old seed-reproducibility test becomes a
# determinism test.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    ("system_name", "ic", "expected"),
    [
        # Sprott (2003), Chaos and Time-Series Analysis, appendix spectra.
        ("Lorenz", [1.0, 1.0, 1.0], 0.906),
        ("Rossler", [1.0, 0.0, 0.0], 0.0714),
        # Lü, Chen, Cheng & Čelikovský (2002), Int. J. Bifurcation Chaos 12, 2917.
        ("Chen", [-0.1, 0.5, -0.6], 2.03),
    ],
)
def test_maximal_exponent_on_flows_matches_literature(system_name, ic, expected) -> None:
    """``lyapunov_spectrum(k=1)`` recovers the literature exponent at its defaults.

    Regression (v6): the pre-v6 ``max_lyapunov`` defaults averaged over
    ``n * steps_per * dt`` = 20 time units and counted the perturbation's
    random-direction alignment transient, which biased a flow ~25 % LOW — Lorenz
    came out at 0.61-0.70 against 0.906, and the only flow test was too loose to
    see it.  The tolerance here is 15 %, tight enough to fail on that bias;
    measured, the survivor lands inside 6 % on all three.
    """
    cls = getattr(ts.systems, system_name)
    value = float(ts.analysis.lyapunov_spectrum(cls(ic=ic), k=1, ic=ic)[0])
    assert value == pytest.approx(expected, rel=0.15), (
        f"{system_name}: lyapunov_spectrum(k=1) gave {value:.4f}, literature {expected}"
    )


@pytest.mark.slow
def test_the_maximal_exponent_is_deterministic() -> None:
    """Repeated calls agree exactly — there is no random direction left to vary.

    Its ancestor scattered over 0.61-0.70 across perturbation seeds (a 13 %
    spread) on the pre-v6 20-time-unit window.  The tangent iteration has no
    seed, so the requirement is now the stronger one: bit-for-bit agreement.
    """
    lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
    values = [
        float(ts.analysis.lyapunov_spectrum(lor, k=1, ic=[1.0, 1.0, 1.0])[0]) for _ in range(3)
    ]
    assert len(set(values)) == 1, values


@pytest.mark.slow
@pytest.mark.parametrize("dt", [0.05, 0.01, 0.002])
def test_the_maximal_exponent_does_not_depend_on_the_output_step(dt: float) -> None:
    """The estimate must be the same at every ``dt`` the caller chooses.

    Regression: the averaging window is a span of **time**, so a default fixed in
    *cycles* silently shrank it as ``dt`` fell — with the old ``n = 2000``
    default, Lorenz returned 0.889 at ``dt = 0.01`` but 0.747 at ``dt = 0.002``
    and 0.659 at ``dt = 0.001`` (27 % low), i.e. the very bias the v6 defaults
    were raised to remove, reachable by passing a finer step.

    Truth: 0.9076, from an independent variational (Benettin) integration of the
    Lorenz equations under ``scipy.integrate.solve_ivp`` at ``rtol=1e-11``;
    literature 0.906 (Sprott 2003).  Measured spread across these three steps is
    under 5 %.
    """
    value = float(
        ts.analysis.lyapunov_spectrum(
            ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]), k=1, ic=[1.0, 1.0, 1.0], dt=dt
        )[0]
    )
    assert value == pytest.approx(0.9076, rel=0.1), (dt, value)


@pytest.mark.slow
def test_the_maximal_exponent_on_a_limit_cycle_is_zero() -> None:
    """A periodic orbit has no positive exponent (the estimator must say so)."""
    periodic = ts.systems.Rossler(params={"a": 0.1, "b": 0.1, "c": 6.0})
    value = float(ts.analysis.lyapunov_spectrum(periodic, k=1, ic=[1.0, 1.0, 1.0])[0])
    assert abs(value) < 0.02, value


# ---------------------------------------------------------------------------
# Kaplan–Yorke dimension: the analytic identities and the edge conventions
# ---------------------------------------------------------------------------


def test_kaplan_yorke_on_literature_spectra() -> None:
    """D_KY of the published Lorenz / folded-towel spectra.

    Both are closed-form given the spectrum: Lorenz
    ``2 + 0.906/14.57 = 2.0622`` (Sprott 2003), folded towel
    ``2 + (0.43 + 0.376)/3.3 = 2.244`` (Rössler 1979, quoted as ~2.25).
    """
    assert float(ts.analysis.kaplan_yorke_dimension([0.906, 0.0, -14.57])) == pytest.approx(
        2.0 + 0.906 / 14.57, rel=1e-12
    )
    assert float(ts.analysis.kaplan_yorke_dimension([0.43, 0.376, -3.3])) == pytest.approx(
        2.0 + (0.43 + 0.376) / 3.3, rel=1e-12
    )


@pytest.mark.parametrize(
    ("spectrum", "expected"),
    [
        ([-0.5, -1.0, -2.0], 0.0),  # stable fixed point: a 0-dimensional attractor
        ([-0.5], 0.0),  # ...also with a single exponent
        ([0.0, -1.0], 1.0),  # limit cycle: exactly the 1-D orbit
        ([0.693], 1.0),  # a 1-D chaotic map fills its interval
        ([1.0, -1.0], 2.0),  # sum exactly zero (conservative) -> saturates at len
        ([0.5, 0.5], 2.0),  # cumulative sum never turns negative -> saturates
    ],
)
def test_kaplan_yorke_edge_conventions(spectrum, expected) -> None:
    """Each documented edge of the Kaplan–Yorke definition returns its convention."""
    assert float(ts.analysis.kaplan_yorke_dimension(spectrum)) == pytest.approx(expected)


@pytest.mark.parametrize("spectrum", [[], [np.nan, -1.0], [np.inf, -1.0], [0.5, -np.inf]], ids=str)
def test_kaplan_yorke_refuses_meaningless_input(spectrum) -> None:
    """An empty or non-finite spectrum raises instead of returning a number.

    An empty spectrum used to return ``0.0`` — indistinguishable from the honest
    answer for a stable fixed point — and a ``nan`` exponent (a non-converged
    estimator) sorted to the end and produced a silently wrong dimension.
    """
    from tsdynamics.analysis.lyapunov import kaplan_yorke_dimension
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        kaplan_yorke_dimension(spectrum)


# ---------------------------------------------------------------------------
# Behavioural one-offs that metadata cannot express
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_lorenz_spectrum_is_dissipative_and_sorted() -> None:
    lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
    spec = ts.analysis.lyapunov_spectrum(lor, dt=0.1, transient=50.0, final_time=200.0)
    assert spec[0] >= spec[1] >= spec[2]
    # divergence of Lorenz = -(sigma + 1 + beta) ≈ -13.67
    assert -20.0 < spec.sum() < -5.0


@pytest.mark.slow
def test_lorenz_partial_spectrum_n_exp_2() -> None:
    lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
    exps = ts.analysis.lyapunov_spectrum(lor, dt=0.1, transient=30.0, final_time=100.0, k=2)
    assert exps.shape == (2,)
    assert exps[0] > 0.0


@pytest.mark.slow
def test_logistic_stable_regime_negative_exponent() -> None:
    """Logistic at r=2 sits on a stable fixed point: LE < 0."""
    m = ts.systems.Logistic(params={"r": 2.0})
    exps = ts.analysis.lyapunov_spectrum(m, n=5_000)
    assert exps[0] < 0.0


@pytest.mark.slow
def test_mackeyglass_two_exponents_finite() -> None:
    mg = ts.systems.MackeyGlass()
    traj = mg.run(
        final_time=200.0,
        dt=0.5,
        history=DDE_HISTORIES["MackeyGlass"],
        rtol=1e-4,
        atol=1e-4,
    )
    exps = ts.analysis.lyapunov_spectrum(
        mg,
        k=2,
        dt=0.5,
        transient=50.0,
        final_time=300.0,
        ic=traj.y[-1],
        rtol=1e-4,
        atol=1e-4,
    )
    assert exps.shape == (2,)
    assert np.all(np.isfinite(exps))


# ---------------------------------------------------------------------------
# The VERDICT the repr prints, scored against the same literature (CONTRACT §4.4)
# ---------------------------------------------------------------------------
#
# §4.4 replaced an unimplementable rule (classify with `lambda_i > max(atol,
# k*sigma_i)`, where sigma is a finite-time scatter the estimator does not
# compute) with a repr-only one calibrated on the estimator's OWN realised zero.
# Its claim is a score: "current rule 1 wrong, new rule 0 wrong" over the
# catalogue's literature spectra.  A claim of that shape belongs beside the
# literature it was scored on, so it lives here rather than in the repr tests —
# and it is cheap, because it reads the DECLARED spectrum instead of estimating
# one.


def _literature_verdict(spectrum, declared_n_positive) -> str:
    """What the literature says this spectrum is, in the repr's own vocabulary."""
    n_pos = declared_n_positive
    if n_pos is None:
        n_pos = count_positive_exponents(spectrum)
    return {0: "regular", 1: "chaotic"}.get(n_pos, "hyperchaotic")


_LITERATURE_SPECTRA = [
    (e.name, e.family, e.known_lyapunov["spectrum"], e.known_lyapunov.get("n_positive"))
    for e in _ENTRIES
    if e.known_lyapunov.get("spectrum") is not None
]


def test_the_literature_spectra_are_actually_swept() -> None:
    """Guard against a vacuous sweep: the catalogue must still declare spectra."""
    assert len(_LITERATURE_SPECTRA) >= 10, (
        f"only {len(_LITERATURE_SPECTRA)} systems declare a literature spectrum — "
        "the verdict sweep below would certify almost nothing"
    )


@pytest.mark.parametrize(
    ("name", "family", "spectrum", "n_positive"),
    _LITERATURE_SPECTRA,
    ids=[row[0] for row in _LITERATURE_SPECTRA],
)
def test_the_repr_names_the_dynamics_the_literature_names(
    name: str, family: str, spectrum, n_positive
) -> None:
    """§4.4 — a literature spectrum prints the literature's own classification.

    ``LotkaVolterra`` is the case that motivated the rule: a shipped
    *conservative* system with spectrum ``[0, 0]``, which the pre-v6 rule called
    **chaotic**.  ``FoldedTowel`` is the other end — two positive exponents must
    read ``hyperchaotic``, not ``chaotic``.
    """
    result = LyapunovSpectrum(
        np.asarray(spectrum, dtype=float),
        meta={"system": name, "analysis": "lyapunov_spectrum"},
    )
    headline = repr(result).splitlines()[0]
    expected = _literature_verdict(spectrum, n_positive)
    assert expected in headline, f"{name}: expected {expected!r} in {headline!r}"
    for wrong in {"regular", "chaotic", "hyperchaotic"} - {expected}:
        if wrong == "chaotic" and expected == "hyperchaotic":
            continue  # "hyperchaotic" contains "chaotic" as a substring
        assert wrong not in headline, f"{name}: {headline!r} also claims {wrong!r}"


@pytest.mark.parametrize(
    ("name", "family", "spectrum", "n_positive"),
    _LITERATURE_SPECTRA,
    ids=[row[0] for row in _LITERATURE_SPECTRA],
)
def test_kaplan_yorke_is_printed_only_when_an_exponent_is_positive(
    name: str, family: str, spectrum, n_positive
) -> None:
    """§4.4 / M26 — on a regular spectrum ``D_KY`` is the integer state dimension.

    Printing it there reads as a measurement of a fractal dimension that was
    never measured, so it is suppressed.
    """
    result = LyapunovSpectrum(
        np.asarray(spectrum, dtype=float),
        meta={"system": name, "analysis": "lyapunov_spectrum"},
    )
    headline = repr(result).splitlines()[0]
    positive = count_positive_exponents(spectrum) > 0
    assert ("D_KY" in headline) is positive, headline


def test_a_spectrum_whose_verdict_moves_across_the_band_refuses_to_name_one() -> None:
    """§4.4 — the honest answer to a too-short horizon is "indeterminate".

    The rule prints a word only when the positive count is stable across a 10x
    tolerance band.  This is the hedge that turned ``HenonHeiles`` from a
    confidently wrong verdict into a correct refusal.
    """
    hedged = LyapunovSpectrum(
        np.asarray([0.11, 0.021, -0.0003, -25.2], dtype=float),
        meta={"system": "Lorenz", "analysis": "lyapunov_spectrum"},
    )
    headline = repr(hedged).splitlines()[0]
    assert "indeterminate" in headline, headline
    assert "final_time" in repr(hedged), "the hedge must say what to do about it"
