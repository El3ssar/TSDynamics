"""The two verbs that turn one system into another.

Ruling **A3** cut the design's five derivation verbs to two, and the census is
the argument: ``project`` and ``tangent`` had **no user callers**.  ``project``
is ``traj["x", "z"]`` when you want the columns and
``ts.derived.ProjectedSystem(sys, [0, 2])`` when you want the live
low-dimensional stepping; ``tangent`` is Lyapunov machinery and lives at
``ts.derived.TangentSystem``.  ``copies`` was one word from ``ensemble`` with a
different return type — its own docstring conceded the pair had already been
renamed once for exactly that reason.

What is left is:

``poincare(plane=None, at=..., *, period=None, ...)``
    **Absorbs ``stroboscope``.**  A section plane and a strobe period are
    mathematically disjoint on their argument — a plane is an affine surface
    ``g(u) = n·u - c``, a period samples the phase *circle* — so one verb with
    two return types is honest, and the two implementations stay two classes
    because the affine section genuinely cannot express ``z mod 2pi``.

``ensemble(states)``
    Many copies of the system, run or stepped together.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DeriveMixin"]


class _Unset:
    """The sentinel ``poincare(at=...)`` uses, rendered legibly in ``help()``.

    ``at=0.0`` is the commonest crossing value, so "not given" cannot be ``0.0``
    and cannot be ``None`` either (``None`` is a legal ``plane``).  A bare
    ``object()`` worked, but printed its **memory address** into the signature of
    every one of the 151 built-in flows: ``at=<object object at 0x7fea54b8b6c0>``.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Render as ``<unset>`` wherever a signature is printed."""
        return "<unset>"


#: "The caller did not pass this" — distinct from any value they *could* pass.
_UNSET: Any = _Unset()

#: The keywords ``poincare`` forwards to the section it builds.  Closed, so a
#: typo names ``sys.poincare`` rather than ``PoincareMap.__init__`` — a class the
#: caller never mentioned.
_SECTION_KEYWORDS = ("dt", "max_time")


class DeriveMixin:
    """``poincare`` and ``ensemble`` — mixed into :class:`SystemBase`."""

    __slots__ = ()

    def poincare(
        self,
        plane: Any = None,
        at: Any = _UNSET,
        *,
        period: float | None = None,
        direction: Any = "up",
        dt: float | None = None,
        max_time: float | None = None,
        **unknown: Any,
    ) -> Any:
        """Take a section of this flow — a plane, or a strobe.

        Two sections of the same flow, one verb, selected by **which keyword you
        gave**::

            ros.poincare("y", 0.0)                  # component + crossing value
            ros.poincare(("y", 0.0, "up"))          # ... with the direction word
            ros.poincare("y", 0.0, direction="up")  # ... as a keyword
            ros.poincare(plane=([1, 0, 0], 0.0))    # an arbitrary normal
            duff.poincare(period=4.488)             # the strobe
            duff.poincare()                         # ... period inferred from the drive

        The result is a *system*: it answers ``run`` / ``step`` / ``state`` like
        any other, and ``ts.analysis.orbit_diagram(pmap, "c", values)`` over it
        is a bifurcation diagram of the flow.

        Parameters
        ----------
        plane : str, int, or tuple, optional
            The state component whose level set defines the section (a name is
            resolved against ``variables``), **or** the whole plane tuple —
            ``(axis, offset)``, ``(axis, offset, direction)``, or
            ``(normal, offset)``.
        at : float, optional
            The crossing value, when ``plane`` names a component.  Default 0.0.
        period : float, optional
            The **forcing period** of a periodically driven flow.  Giving it
            selects a stroboscopic section instead of an affine one.  Omitting
            *both* ``plane`` and ``period`` infers the period from the system's
            drive hook (``forcing_period`` / ``drive_period``, or ``omega`` /
            ``drive_frequency`` read as an angular frequency).
        direction : int or str, default ``"up"``
            Crossing direction for a plane — a sign, or ``"up"`` / ``"down"`` /
            ``"both"``.  A direction inside the plane tuple wins.
        dt : float, optional
            Detection step for the crossing march, in **time units**.
        max_time : float, optional
            How long to march before giving up on finding a crossing, in **time
            units**.

        Returns
        -------
        PoincareMap or StroboscopicMap

        Raises
        ------
        InvalidParameterError
            If both a plane and a period are given, or if neither is given and
            the system exposes no drive hook to infer a period from.
        """
        from tsdynamics.errors import InvalidParameterError

        if unknown:
            from ._kwargs import run_keyword_error

            bad = next(iter(unknown))
            raise run_keyword_error(
                self,
                bad,
                unknown[bad],
                family=getattr(self, "family", "ode"),
                accepted=("plane", "at", "period", "direction", *_SECTION_KEYWORDS),
                verb="poincare",
            )
        kwargs = {k: v for k, v in (("dt", dt), ("max_time", max_time)) if v is not None}

        gave_plane = plane is not None or at is not _UNSET
        if gave_plane and period is not None:
            raise InvalidParameterError(
                "poincare() takes a section plane or a strobe period, not both.\n"
                "A plane is an affine surface g(u) = n·u - c; a period samples the "
                "phase circle. They are different sections of the same flow.\n"
                "    duff.poincare('z', 0.0)      # the plane\n"
                "    duff.poincare(period=4.488)  # the strobe"
            )

        if gave_plane:
            from tsdynamics.derived import PoincareMap

            whole_plane = isinstance(plane, (tuple, list)) and len(plane) in (2, 3) and at is _UNSET
            resolved = plane if whole_plane else (plane, 0.0 if at is _UNSET else float(at))
            return PoincareMap(self, resolved, direction=direction, **kwargs)

        from tsdynamics.derived import PoincareMap, StroboscopicMap
        from tsdynamics.derived.stroboscopic import infer_forcing_period

        if period is None:
            # Neither a plane nor a period: a *periodically forced* flow has an
            # obvious section — its own drive — so infer it.  An autonomous flow
            # has no drive to read, and its obvious section is a plane the
            # attractor actually crosses, so fall through to ``auto_plane``.
            try:
                period = infer_forcing_period(self)
            except KeyError:
                return PoincareMap(self, None, direction=direction, **kwargs)
        return StroboscopicMap(self, period, **kwargs)

    def ensemble(self, states: Any, **horizon: Any) -> Any:
        """Many copies of this system, run or stepped together.

        Returns an :class:`~tsdynamics.derived.Ensemble` — a *system*, so it
        answers the same verbs everything else does::

            band = lor.ensemble(np.random.rand(100, 3))
            batch = band.run(final_time=10.0)     # -> TrajectoryBatch
            batch.final                           # -> (100, 3) final states
            band.step(0.01)                       # ... or drive it yourself

        Before v6 this verb *ran* the batch and returned a bare ``(n, dim)``
        array while a second verb, ``copies``, returned the lazy wrapper.  One
        object with one set of verbs replaces the pair.

        Parameters
        ----------
        states : array-like, shape (n, dim)
            One row per copy.

        Returns
        -------
        Ensemble
        """
        _HORIZON = ("final_time", "dt", "steps", "t0", "transient", "seed")
        given = [k for k in sorted(horizon) if k in _HORIZON]
        if given:
            from tsdynamics.errors import InvalidParameterError, remedy

            raise InvalidParameterError(
                f"ensemble() builds the batch; it does not run it, so it takes no "
                f"{given[0]!r}. One verb, one object (§3.2)."
                + remedy(
                    "band = system.ensemble(states)",
                    f"batch = band.run({', '.join(f'{k}=...' for k in given)})",
                    "batch.final          # the (n, dim) end states",
                )
            )
        if horizon:
            # NOT a horizon word, so the "run it instead" line would hand back a
            # call that raises.  Answer it as the ordinary unknown keyword it is.
            from ._kwargs import run_keyword_error

            bad = sorted(horizon)[0]
            raise run_keyword_error(
                self,
                bad,
                horizon[bad],
                family=getattr(self, "family", "ode"),
                accepted=("states",),
                verb="ensemble",
            )
        from tsdynamics.derived import Ensemble

        return Ensemble(self, states)
