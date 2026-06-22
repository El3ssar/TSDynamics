r"""IC policy for :func:`tsdynamics.analysis.chaos.gali` (stream **FIX-GALI-IC**).

``gali`` characterises one *specific* orbit, so it must distinguish a caller-pinned
initial condition from the default draw:

- an **explicit** ``ic`` that diverges (escapes the attractor's basin) is a user
  error to surface — ``gali`` raises :class:`~tsdynamics.errors.InvalidInputError`
  rather than silently substituting a different (random) orbit and returning a
  result for an orbit the caller never asked about;
- with ``ic=None`` (the default draw — many maps carry no ``default_ic``) an
  off-basin first attempt is *expected*, so the seeded random-IC retry survives:
  ``gali`` recovers onto the attractor and never raises.

These two behaviours are the whole of the fix; both are asserted below, on the
pure-Python ``_step``/``_jacobian`` (maps) and the RK4 variational core (flows),
so none of these tests need the Rust engine.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.analysis.chaos import GALIResult, gali
from tsdynamics.errors import InvalidInputError

# An IC far outside the Hénon attractor's basin: x' = 1 - 1.4·100 + 10 = -129,
# then squared next step → overflow to a non-finite state within the burn-in.
HENON_OFFBASIN_IC = [10.0, 10.0]


# ── (a) an explicit diverging ic RAISES (the core fix) ────────────────────────


def test_explicit_diverging_ic_raises_invalidinput():
    """A pinned ``ic`` that escapes the basin must raise, not silently re-roll."""
    with pytest.raises(InvalidInputError):
        gali(ts.Henon(), k=2, ic=HENON_OFFBASIN_IC, n=80, seed=0)


def test_explicit_diverging_ic_is_a_typeerror_subclass():
    """``InvalidInputError`` is a ``TypeError`` — legacy ``except TypeError`` catches it."""
    with pytest.raises(TypeError):  # InvalidInputError subclasses TypeError
        gali(ts.Henon(), k=2, ic=HENON_OFFBASIN_IC, n=80, seed=0)


def test_explicit_diverging_ic_message_names_the_value():
    """The error follows the value-naming standard: it names the offending ic."""
    with pytest.raises(InvalidInputError) as excinfo:
        gali(ts.Henon(), k=2, ic=HENON_OFFBASIN_IC, n=80, seed=0)
    msg = str(excinfo.value)
    assert "10.0" in msg  # the offending ic is reported back
    assert "ic" in msg


def test_explicit_diverging_ic_does_not_silently_reroll():
    """Distinct off-basin explicit ICs both raise (no hidden recovery onto a result).

    A silent re-roll would return an (identical, seed-pinned) ``GALIResult`` from a
    random orbit regardless of the diverging ``ic`` — i.e. the ``ic`` would be
    ignored.  Here every off-basin ``ic`` must raise instead.
    """
    for offbasin in ([10.0, 10.0], [-50.0, 7.0], [1e3, -1e3]):
        with pytest.raises(InvalidInputError):
            gali(ts.Henon(), k=2, ic=offbasin, n=60, seed=0)


# ── (b) ic=None still recovers via the seeded random-IC retry ─────────────────


def test_implicit_ic_none_recovers_and_never_raises():
    """With ``ic=None`` the off-basin draws are re-rolled and gali returns a result.

    ``Henon`` declares no ``default_ic``, so every call draws a random IC; many
    land outside the attractor's basin.  The retry must recover onto the attractor
    and return a finite, chaotic ``GALIResult`` every time (regression for the
    re-roll path the fix must *preserve*).
    """
    for seed in range(12):
        g = gali(ts.Henon(), k=2, n=1500, seed=seed)
        assert isinstance(g, GALIResult)
        assert np.all(np.isfinite(g.values))
        assert g.is_chaotic()  # Hénon is chaotic → GALI₂ collapses toward 0


def test_implicit_ic_none_omitted_keyword_recovers():
    """Omitting the ``ic`` keyword entirely (not even ``ic=None``) also recovers."""
    g = gali(ts.Henon(), k=2, n=1200)
    assert isinstance(g, GALIResult)
    assert np.all(np.isfinite(g.values))


# ── (c) a good explicit ic still works (map + flow) ───────────────────────────


def test_good_explicit_ic_map_works():
    """An on-attractor explicit ``ic`` for a map returns the expected chaotic decay."""
    g = gali(ts.Henon(), k=2, ic=[0.1, 0.1], n=70, seed=0)
    assert isinstance(g, GALIResult)
    assert np.all(np.isfinite(g.values))
    assert g.is_chaotic()
    assert np.all(g.values <= 1.0 + 1e-9)  # GALI_k ∈ [0, 1] (unit-column volume)


def test_good_explicit_ic_flow_works():
    """An on-attractor explicit ``ic`` for a flow returns a finite GALI series."""
    g = gali(ts.Lorenz(), k=2, ic=[1.0, 1.0, 1.0], final_time=20.0, dt=0.05, seed=0)
    assert isinstance(g, GALIResult)
    assert np.all(np.isfinite(g.values))
    assert g.is_chaotic()  # Lorenz is chaotic → GALI₂ collapses toward 0


def test_good_explicit_ic_is_honoured_exactly():
    """A good explicit ``ic`` is run as given — repeating the call reproduces it.

    The explicit path takes **no random IC draw** (only the deviation frame is
    seeded), so two calls with the same ``ic`` and ``seed`` are bit-identical:
    the orbit is the one the caller pinned, not a re-rolled random one.
    """
    g1 = gali(ts.Henon(), k=2, ic=[0.1, 0.1], n=70, seed=0)
    g2 = gali(ts.Henon(), k=2, ic=[0.1, 0.1], n=70, seed=0)
    assert np.array_equal(g1.values, g2.values)

    # A *different* on-basin ic gives a different orbit (so the ic is not ignored).
    g3 = gali(ts.Henon(), k=2, ic=[0.3, -0.1], n=70, seed=0)
    assert not np.array_equal(g1.values, g3.values)
