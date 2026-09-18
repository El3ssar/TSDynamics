"""What the v6 round-9 visibility sweep had to give back.

Round 9 hid ~560 tab-surface slots under one rule: *hiding is a DISCOVERY
change, never a REACHABILITY change* (CONTRACT §11.1).  A blind user then drove
the reduced surface and found five places where that promise was not kept, or
where the surface was telling them something untrue.  Each is pinned here, in
the user's own words, so the next sweep cannot re-break it:

1. an out-of-tree ``@ts.analysis.register`` did not produce the attribute its
   own docstring promises and ``find()`` prints;
2. the four names in ``ts.families`` had no forwarding address, so ``ts.System``
   answered with three classes that are not it;
3. ``Geometry.parts`` — named by two of ``Geometry``'s own error messages — was
   not in ``dir(g)``;
4. ``Plot.from_dict(json.loads(p.to_json()))`` raised a bare ``KeyError``;
5. CPython's own *"Did you mean …?"* was appended to our migration messages,
   offering the reader a **private** name.

Plus two answers that were wrong rather than merely hidden: the counts printed
by the "I cannot find that name" message, and an overlay drawn outside the
host's frame.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import tsdynamics as ts  # noqa: E402


@pytest.fixture
def traj():
    """A short Lorenz run — the cheapest real subject."""
    return ts.systems.Lorenz().run(final_time=2.0, dt=0.05, ic=[1.0, 1.0, 1.0])


# ── 1. registering an analysis out of tree ───────────────────────────────────


def test_registering_an_analysis_binds_it_on_the_namespace(traj):
    """``register`` promises six things; three of them are the package surface.

    Measured before the fix: the registry entry appeared and ``find(traj)``
    listed the analysis under the heading ``ts.analysis.<name>(data, ...)`` —
    and that exact line raised ``AttributeError``, whose own suggestion was the
    name it had just refused.
    """
    name = "orbit_extent_probe"
    try:

        @ts.analysis.register(subjects=("trajectory",), area="geometry", keywords="extent span")
        def orbit_extent_probe(data):
            """Peak-to-peak range of each component."""
            return np.ptp(np.asarray(data), axis=0)

        assert name in ts.registry.analyses.names()
        assert name in ts.analysis.__all__  # the promised __all__ entry
        assert name in dir(ts.analysis)  # ...therefore in TAB
        assert name in (ts.analysis.__doc__ or "")  # the promised __doc__ row
        assert ts.analysis.orbit_extent_probe is orbit_extent_probe  # the attribute
        assert np.allclose(ts.analysis.orbit_extent_probe(traj), np.ptp(traj.y, axis=0))
        assert any(f.__name__ == name for f in ts.analysis.find(traj))
    finally:
        ts.registry.analyses.unregister(name)
        ts.analysis._refresh_surface()
    assert name not in ts.analysis.__all__


# ── 2. the families home has a forwarding address ────────────────────────────


@pytest.mark.parametrize("name", ["System", "SystemBase", "ParamSet", "MetaStore"])
def test_a_families_name_is_answered_with_its_address(name):
    """These four are the names you reach for when you WRITE against the library.

    ``System`` is the Protocol you annotate and ``isinstance``-check,
    ``SystemBase`` the class you subclass to add a family.  Before this, they
    were the only entries of a public home's ``__all__`` with no row in
    ``_PUBLIC_HOMES``: ``ts.System`` offered ``DerivedSystem`` /
    ``TangentSystem`` / ``WrappedSystem``, none of which is it.
    """
    with pytest.raises(ts.errors.MovedInV6) as excinfo:
        getattr(ts, name)
    assert f"ts.families.{name}" in str(excinfo.value)
    # ...and the address RESOLVES, which is the whole point of printing it.
    assert getattr(ts.families, name) is not None


# ── 3. the arrays escape hatch lists what its errors name ────────────────────


def test_geometry_lists_parts_because_its_errors_hand_it_back(traj):
    """``g.parts`` is the remedy two ``Geometry`` errors print."""
    g = ts.viz.geometry(traj, "phase_portrait")
    assert "parts" in dir(g)
    assert len(g.parts) >= 1
    assert "g.parts" in (ts.viz.geometry.__doc__ or "")


# ── 4. the JSON round trip a user actually types ─────────────────────────────


def test_from_dict_accepts_the_envelope_to_json_writes(traj):
    """``to_json`` writes ``{schema_version, spec}``; ``from_dict`` now unwraps it.

    The obvious inverse used to fail with a bare ``KeyError: 'kind'`` — the only
    untaught error in a whole blind session.
    """
    p = ts.plot(traj)
    for payload in (p.to_dict(), json.loads(p.to_json())):
        rebuilt = ts.viz.Plot.from_dict(payload)
        assert rebuilt.kind == p.kind
        assert len(rebuilt.layers) == len(p.layers)
    with pytest.raises(ts.InvalidInputError, match="ts.viz.load"):
        ts.viz.Plot.from_dict({"not": "a plot"})
    plt.close("all")


# ── 5. our message is the last word ──────────────────────────────────────────


def _rendered(exc: BaseException) -> str:
    """Render *exc* the way a traceback would, suggestion included."""
    import traceback

    return "".join(traceback.format_exception_only(type(exc), exc))


@pytest.mark.parametrize(
    ("subject", "name"),
    [
        ("flow", "lyapunov_spectrum"),
        ("map", "lyapunov_spectrum"),
        ("traj", "dims"),
        ("traj", "recurrence"),
        ("traj", "to_plot_spec"),
        ("flow", "integrate"),
        ("wrapper", "integrate"),
    ],
)
def test_no_cpython_suggestion_is_appended_to_a_taught_message(subject, name, traj):
    """A migration message must not end in a guess that contradicts it.

    Measured before the seal: ``lorenz.lyapunov_spectrum`` — whose message names
    the free function to call — ended ``Did you mean: '_lyapunov_spectrum'?``,
    the private helper behind that very function; ``traj.dims`` (a retired
    namespace of four analyses) was answered ``Did you mean: 'dim'?``, an
    integer; and ``traj.to_plot_spec`` was answered with the dunder its own
    message calls *"not a verb you type"*.
    """
    held = {
        "flow": ts.systems.Lorenz(),
        "map": ts.systems.Henon(),
        "traj": traj,
        "wrapper": ts.systems.Rossler().poincare("y", 0.0),
    }[subject]
    with pytest.raises(AttributeError) as excinfo:
        getattr(held, name)
    assert "Did you mean: '" not in _rendered(excinfo.value)
    assert excinfo.value.name == name  # ``except AttributeError as e: e.name`` still reads
    assert not hasattr(held, name)  # ...and sealing did not make the miss a hit


# ── the counts a lost user is shown ──────────────────────────────────────────


def test_the_miss_message_counts_what_it_says_it_counts():
    """It advertised "180 built-in systems" (177) and "51 quantifiers" (49)."""
    with pytest.raises(AttributeError) as excinfo:
        getattr(ts, "SomethingNobodyHasEverTyped")  # noqa: B009 - the miss IS the subject
    text = str(excinfo.value)
    n_systems = sum(1 for name in ts.systems.__all__ if isinstance(getattr(ts.systems, name), type))
    n_analyses = len(ts.registry.analyses.names())
    assert f"the {n_systems} built-in systems" in text
    assert f"the {n_analyses} quantifiers" in text


# ── an overlay that lands inside the frame ───────────────────────────────────


class _Bistable(ts.ContinuousSystem):
    """Two wells far outside any auto-sampled nullcline window."""

    params = {"a": 1.0, "b": 0.02, "d": 0.3}
    variables = ("x", "v")

    @staticmethod
    def _equations(u, t, a, b, d):
        return [u(1), a * u(0) - b * u(0) ** 3 - d * u(1)]


def test_an_overlay_widens_the_host_window_to_enclose_itself():
    """A ``model`` host pins its sampled box; the overlay used to be clipped off."""
    system = _Bistable()
    host = ts.plot(system, "nullclines")
    before = tuple(host.x.limits)
    points = ts.analysis.fixed_points(system, region=[(-10, 10), (-10, 10)])
    xs = np.asarray(points)[:, 0]
    assert xs.max() > before[1], "the probe must actually fall outside the host window"
    points.overlay_on(host)
    assert host.x.limits[0] <= xs.min() and host.x.limits[1] >= xs.max()
    plt.close("all")


def test_an_overlay_that_already_fits_changes_nothing():
    """Widening must be a no-op in the common case, never a rescale."""
    vdp = ts.systems.VanDerPol()
    host = ts.plot(vdp, "nullclines")
    before = (tuple(host.x.limits), tuple(host.y.limits))
    ts.analysis.fixed_points(vdp, region=[(-3, 3), (-3, 3)]).overlay_on(host)
    assert (tuple(host.x.limits), tuple(host.y.limits)) == before
    plt.close("all")
