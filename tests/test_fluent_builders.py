"""Tests for the fluent derived-builder verbs (stream WS-FLUENT).

WS-ACCESSORS (#226) added the builder *verbs* (``poincare`` / ``stroboscope`` /
``tangent`` / ``project`` / ``ensemble``) and tested that each returns the right
wrapper type (see ``test_accessors.py``).  WS-FLUENT (#203) adds the two pieces
that make the fluent flow *read left-to-right and run*:

* a ``.run(...)`` alias on the derived wrappers, so the documented flow
  ``Rossler().poincare(section="y", at=0.0).run(steps=500)`` produces the section
  (byte-identical to the wrapper's ``.trajectory(...)``);
* forcing-period **inference** for ``stroboscope()`` — the period is read from
  the system's drive (``omega`` / ``drive_frequency`` / ``forcing_period``), with
  an explicit ``period=`` override retained and a clear error when none can be
  inferred.

These assert the WS-FLUENT contract specifically; the wrapper-type / binding
guarantees live in ``test_accessors.py`` and are not duplicated here.  The
builders run / iterate the system, so the module needs the compiled extension.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.derived import StroboscopicMap
from tsdynamics.families._accessors import infer_forcing_period
from tsdynamics.systems import Duffing, Lorenz, Rossler

# Engine-backed (runs the flow); gates the module and auto-tags it ``engine``.
pytest.importorskip("tsdynamics._rust")


# --------------------------------------------------------------------------- #
# the ``.run(...)`` alias on derived wrappers (left-to-right fluent flow)
# --------------------------------------------------------------------------- #


def test_poincare_run_produces_a_section():
    """``sys.poincare(...).run(steps=N)`` returns the crossing section."""
    pm = Rossler().poincare(section="y", at=0.0, direction=+1)
    assert hasattr(pm, "run")
    section = pm.run(steps=25, transient=5, ic=[1.0, 1.0, 1.0])
    # one row per requested crossing, full state width
    assert section.y.shape == (25, Rossler().dim)


def test_poincare_run_matches_trajectory():
    """``.run`` is a byte-identical alias of the wrapper's ``.trajectory``."""
    pm = Rossler().poincare(section="y", at=0.0, direction=+1)
    ic = [0.5, 0.5, 0.5]
    via_run = pm.run(steps=30, transient=8, ic=ic)
    via_traj = pm.trajectory(30, transient=8, ic=ic)
    assert np.array_equal(via_run.y, via_traj.y)
    assert np.array_equal(via_run.t, via_traj.t)


def test_stroboscope_run_matches_trajectory():
    """The stroboscopic wrapper's ``.run`` alias matches ``.trajectory`` too."""
    strobe = Duffing().stroboscope()
    ic = [0.1, 0.0, 0.0]
    via_run = strobe.run(steps=20, transient=3, ic=ic)
    via_traj = strobe.trajectory(20, transient=3, ic=ic)
    assert np.array_equal(via_run.y, via_traj.y)


def test_run_alias_lives_on_derived_base():
    """The alias is defined once on the wrapper base, not per subclass."""
    from tsdynamics.derived._base import DerivedSystem

    assert "run" in vars(DerivedSystem)
    # and it is genuinely a delegating alias (not a re-implementation)
    assert "trajectory" in DerivedSystem.run.__doc__


def test_fluent_chain_reads_left_to_right():
    """The headline flow from the design dossier works end to end."""
    section = Rossler().poincare(section="y", at=0.0).run(steps=40, ic=[1.0, 1.0, 1.0])
    assert section.y.shape[0] == 40


# --------------------------------------------------------------------------- #
# stroboscope() infers its forcing period
# --------------------------------------------------------------------------- #


def test_stroboscope_infers_period_from_omega():
    """A forced system's period is inferred as ``2*pi/omega`` from its drive."""
    duf = Duffing()
    strobe = duf.stroboscope()
    assert isinstance(strobe, StroboscopicMap)
    assert strobe.period == pytest.approx(2.0 * np.pi / duf.omega)


def test_stroboscope_explicit_period_overrides_inference():
    """An explicit ``period=`` is honoured and never inferred over."""
    strobe = Duffing().stroboscope(period=3.21)
    assert strobe.period == 3.21


def test_stroboscope_inference_tracks_a_changed_omega():
    """Inference reads the drive *live*: a changed ``omega`` changes the period."""
    duf = Duffing()
    duf.omega = 2.0
    strobe = duf.stroboscope()
    assert strobe.period == pytest.approx(2.0 * np.pi / 2.0)
    # with_params path agrees with the constructor path it sugars
    duf2 = Duffing().with_params(omega=2.0)
    assert duf2.stroboscope().period == pytest.approx(np.pi)


def test_stroboscope_inferred_equals_explicit_constructor_path():
    """Inferred stroboscope is identical to the hand-built one (no behaviour add)."""
    duf = Duffing()
    inferred = duf.stroboscope()
    explicit = StroboscopicMap(duf, 2.0 * np.pi / duf.omega)
    assert inferred.period == pytest.approx(explicit.period)
    assert type(inferred) is type(explicit)


def test_stroboscope_unforced_system_raises_clearly():
    """A system with no drive hook cannot infer a period — clear error."""
    with pytest.raises(ts.errors.InvalidParameterError) as excinfo:
        Lorenz().stroboscope()
    msg = str(excinfo.value)
    assert "period" in msg
    # the message must point the user at the explicit escape hatch
    assert "period=" in msg


def test_stroboscope_unforced_system_works_with_explicit_period():
    """The explicit override keeps unforced systems usable (no regression)."""
    strobe = Lorenz().stroboscope(period=1.5)
    assert isinstance(strobe, StroboscopicMap)
    assert strobe.period == 1.5


# --------------------------------------------------------------------------- #
# the period-inference helper (priority order + extension hooks)
# --------------------------------------------------------------------------- #


def test_infer_forcing_period_from_omega():
    """``omega`` is read as an angular drive frequency."""
    assert infer_forcing_period(Duffing()) == pytest.approx(2.0 * np.pi / Duffing().omega)


def test_infer_forcing_period_prefers_drive_frequency_hook():
    """A ``drive_frequency`` attribute hook wins over the ``omega`` convention."""

    class DriveHookSystem(ts.ContinuousSystem):
        params = {"omega": 1.0}
        dim = 1
        drive_frequency = 4.0  # explicit hook, distinct from omega

        @staticmethod
        def _equations(y, t, omega):
            return [-y(0)]

    period = infer_forcing_period(DriveHookSystem())
    assert period == pytest.approx(2.0 * np.pi / 4.0)


def test_infer_forcing_period_prefers_explicit_period_hook():
    """A ``forcing_period`` hook is used verbatim (no ``2*pi`` conversion)."""

    class PeriodHookSystem(ts.ContinuousSystem):
        params = {"omega": 1.0}
        dim = 1
        forcing_period = 7.5  # used as-is

        @staticmethod
        def _equations(y, t, omega):
            return [-y(0)]

    assert infer_forcing_period(PeriodHookSystem()) == pytest.approx(7.5)


def test_infer_forcing_period_no_hook_raises_keyerror():
    """No recognised hook → KeyError (the caller wraps it into a friendly error)."""
    with pytest.raises(KeyError):
        infer_forcing_period(Lorenz())


def test_infer_forcing_period_rejects_nonpositive_omega():
    """A present-but-invalid drive value is rejected, not silently used."""

    class BadDriveSystem(ts.ContinuousSystem):
        params = {"omega": -1.0}
        dim = 1

        @staticmethod
        def _equations(y, t, omega):
            return [-y(0)]

    with pytest.raises(ts.errors.InvalidParameterError):
        infer_forcing_period(BadDriveSystem())
