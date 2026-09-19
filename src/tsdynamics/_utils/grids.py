"""Output-grid helpers shared by every family and the engine seam.

The one definition of the uniform output grid an ``integrate`` call samples on.
Both the family base classes (:mod:`tsdynamics.families`) and the engine run
layer (:mod:`tsdynamics._engine.run`) build the same grid, so it lives here in
the leaf ``utils`` package (it imports only NumPy) and both layers consume it —
rather than each carrying a byte-identical private copy that could silently
drift apart.

It owns the numeric guards over the same march for the same reason:
:func:`validate_max_step` is consumed by the SciPy-backed reference integrator
and by the event seam, which would otherwise each carry their own copy of one
sentence.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["make_output_grid", "validate_max_step"]


def validate_max_step(max_step: float | None) -> None:
    """Refuse a ``max_step`` ceiling that cannot bound anything.

    ``None`` and ``inf`` mean "no ceiling" and are inert; ``nan`` and anything
    ``<= 0`` are refused.

    The engine validates at the FFI boundary, but the *stiff* and
    ``backend="reference"`` paths route to :func:`scipy.integrate.solve_ivp`,
    which raises a bare :class:`ValueError` for ``max_step <= 0`` and — worse —
    **silently accepts** ``nan`` (every ``h > max_step`` comparison is then
    false, so the ceiling never binds).  Validating here is what makes all three
    backends reject the same values with the same type, instead of a typo'd
    ceiling raising on ``interp``/``jit`` and quietly doing nothing on
    ``reference``.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If ``max_step`` is ``nan`` or non-positive.
    """
    from tsdynamics.errors import InvalidParameterError

    if max_step is None:
        return
    value = float(max_step)
    if math.isnan(value) or value <= 0.0:
        raise InvalidParameterError(
            f"max_step must be positive (or infinite for no ceiling); got {max_step}"
        )


def make_output_grid(t0: float, tf: float, dt: float) -> np.ndarray:
    """Build a uniform output grid from ``t0`` to ``tf`` (inclusive).

    The grid is ``arange(t0, tf, dt)`` with ``tf`` appended when the last
    sample would otherwise fall short of it — so the final time is always
    sampled exactly, regardless of whether ``dt`` divides ``tf - t0``.

    The endpoint tolerance
    ----------------------
    ``tf`` is appended when ``t_arr`` is empty *or* its last sample sits below
    ``tf - 1e-12``.  The small absolute slack matters: when ``dt`` divides the
    window cleanly, the last ``arange`` sample lands on ``tf`` only up to
    floating-point error — typically a fraction of a ULP *below* it.  Comparing
    against the bare ``tf`` would then see ``t_arr[-1] < tf`` and append a second,
    sub-ULP-spaced "endpoint", giving a spurious final segment of width ~1e-16.
    The ``1e-12`` slack treats any sample already within that band of ``tf`` as
    *being* the endpoint, so a cleanly-dividing window yields no duplicate tail.
    It is an **absolute** tolerance (not relative) by deliberate design: the
    integration windows here are O(1)–O(1e3) in time units and ``dt`` is rarely
    below ~1e-6, so 1e-12 is far smaller than any meaningful step yet comfortably
    larger than ``arange`` round-off — a relative tolerance would buy nothing and
    complicate the contract.  (A pathological window with ``dt`` itself near
    1e-12 is already rejected upstream as physically meaningless.)

    This is the one chokepoint every flow family (ODE / DDE / SDE) and the
    engine run layer build their grid through, so it is also where the
    silent-footgun horizons are caught early with a domain message: a
    non-positive ``dt`` (which used to surface as a bare ``ZeroDivisionError``
    from this helper) and a window that does not run forward in time (which used
    to yield a one-sample garbage trajectory).

    Every bound must additionally be **finite**.  ``dt`` and ``final_time`` were
    guarded only by ``> 0`` / ``> t0``, which ``+inf`` satisfies: an infinite
    ``dt`` slipped through to a two-sample ``[t0, tf]`` grid (a "trajectory" of
    one giant step, silently nothing like the requested sampling), and an
    infinite ``final_time`` reached :func:`numpy.arange`, which raised the
    unhelpful ``ValueError: Maximum allowed size exceeded``.  Both are rejected
    here instead, naming the offending value.

    Parameters
    ----------
    t0, tf : float
        Start and end of the window.
    dt : float
        Output sampling interval.  Must be strictly positive.

    Returns
    -------
    ndarray
        The output times, with ``t_arr[0] == t0`` and ``t_arr[-1] == tf``.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If ``dt`` is not finite and strictly positive, if ``t0`` is not finite,
        or if ``tf`` is not finite and strictly after ``t0`` (an empty /
        backwards / unbounded window).  All subclass :class:`ValueError`, so
        ``except ValueError`` still catches them.

    Examples
    --------
    >>> make_output_grid(0.0, 1.0, 0.5)
    array([0. , 0.5, 1. ])
    """
    from tsdynamics.errors import invalid_value

    if not (math.isfinite(dt) and dt > 0):
        raise invalid_value("dt", dt, rule="must be finite and > 0 (the output sampling interval)")
    if not math.isfinite(t0):
        raise invalid_value("t0", t0, rule="must be finite (the start of the window)")
    if not math.isfinite(tf):
        raise invalid_value(
            "final_time",
            tf,
            rule="must be finite (the window has to have an end to sample)",
        )
    if not tf > t0:
        raise invalid_value(
            "final_time",
            tf,
            rule=f"must run forward in time (be > the start time {t0!r})",
            hint="check the sign and that final_time exceeds t0",
        )
    t_arr = np.arange(t0, tf, dt)
    if t_arr.size == 0 or t_arr[-1] < tf - 1e-12:
        t_arr = np.append(t_arr, tf)
    return t_arr


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
