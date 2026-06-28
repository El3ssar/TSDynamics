"""
The trajectory — the lingua franca every analysis consumes.

:class:`Trajectory` is the result of integrating or iterating a dynamical
system: a time vector ``t`` and a state array ``y`` of shape ``(T, dim)``,
plus a back-reference to the producing system and a provenance ``meta`` dict.

It lives in :mod:`tsdynamics.data` (not in the families) because it is a *data*
type: the families merely produce it, while the whole analysis layer
(dimensions, embeddings, entropy, recurrence, surrogates, …) consumes it.  It
re-exports through :mod:`tsdynamics.families` and the top-level namespace, so
``from tsdynamics import Trajectory`` and ``from tsdynamics.data import
Trajectory`` resolve to the same object.

The point-set operations (:meth:`Trajectory.minmax`,
:meth:`Trajectory.standardize`, :meth:`Trajectory.neighbors`,
:meth:`Trajectory.set_distance`) build on the geometry primitives in
:mod:`tsdynamics.data.sampling`; the KD-tree backing
:meth:`Trajectory.neighbors` is built lazily and cached per instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from scipy.spatial import cKDTree

    from tsdynamics.viz.spec import Animation, PlotSpec


# ---------------------------------------------------------------------------
# to_plot_spec routing tables (the single-panel front door)
# ---------------------------------------------------------------------------

#: Friendly ``kind=`` spellings → the internal routing key.  A *recipe* like
#: ``"delay"`` is **not** a :class:`~tsdynamics.viz.spec.PlotKind` member (the
#: enum is frozen); it routes to a producer that emits a real semantic kind
#: (a delay embedding is a ``PHASE_PORTRAIT_2D``).  Every other ``kind=`` value
#: passes through unchanged and is resolved against ``PlotKind`` directly.
_KIND_ALIASES: dict[str, str] = {
    "delay": "delay_embedding",
    "delay_embedding": "delay_embedding",
    # ``"field"`` is a *recipe* (not a frozen PlotKind value): it routes to the
    # ``spatial_field`` producer, which emits a real ``SPATIAL_FIELD`` spec — a
    # 1-D profile line or a 2-D heatmap of the system's field, reshaped via its
    # ``_field_shape``.  ``"spatial_field"`` (the kind value) routes here too.
    "field": "spatial_field",
    "spatial_field": "spatial_field",
}

#: Per-route allow-list of the extra keyword(s) accepted via ``**kind_kw`` — kept
#: off the ``to_plot_spec`` signature because each is valid for one kind only.
#: This table is the one place the per-kind options live; extending a kind's
#: options is a one-line edit here (the validation + ``plot()`` forwarding both
#: read it), so the surface grows without reshaping the signature.
_KIND_KW: dict[str, frozenset[str]] = {
    "delay_embedding": frozenset({"tau"}),
    "time_series": frozenset({"color_by"}),
    "phase_portrait_2d": frozenset({"color_by"}),
    "phase_portrait_3d": frozenset({"color_by"}),
    "spacetime": frozenset({"transpose"}),
    # ``"field"`` takes no per-kind keyword — the spatial grid comes from the
    # system's ``_field_shape`` (via meta), and the field-block selector rides on
    # the main ``components=`` argument.
    "spatial_field": frozenset(),
}

#: The keywords ``plot()`` peels off and forwards to :meth:`Trajectory.to_plot_spec`
#: (rather than leaking them to the renderer).  Derived from the routing tables so
#: it can never drift out of sync with the per-kind options above.
_PLOT_SPEC_KEYS: frozenset[str] = frozenset({"kind", "components", "animate"}).union(
    *_KIND_KW.values()
)


def _auto_route(n_components: int) -> str:
    """Pick the default routing key from the number of selected components.

    1 → time series, 2 → 2-D portrait, 3 → 3-D portrait, **4+ → spacetime image**
    (a high-dimensional field reads as a spacetime plot, never a bogus 3-D
    portrait of its first three coordinates).
    """
    if n_components > 3:
        return "spacetime"
    if n_components == 3:
        return "phase_portrait_3d"
    if n_components == 2:
        return "phase_portrait_2d"
    return "time_series"


class Trajectory:
    """
    The result of integrating or iterating a dynamical system.

    Supports tuple-unpacking for backward compatibility::

        t, y = system.integrate(final_time=100)

    Attributes
    ----------
    t : ndarray, shape (T,)
        Time points (or step indices for discrete maps).
    y : ndarray, shape (T, dim)
        State at each time point.
    system : SystemBase
        Back-reference to the system that produced this trajectory.
    meta : dict
        Provenance: system name, params snapshot, solver, tolerances, ic.

    Examples
    --------
    >>> traj = lor.integrate(final_time=100)
    >>> traj.dim
    3
    >>> traj["x"]            # named component (via the class's ``variables``)
    array([...])
    >>> traj.after(20.0)     # drop transient
    Trajectory(n_steps=..., dim=3, t=[20.0, 100.0])
    >>> t, y = traj          # tuple-unpack still works
    """

    __slots__ = ("t", "y", "system", "meta", "_kdtree")

    def __init__(
        self,
        t: np.ndarray,
        y: np.ndarray,
        system: Any,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.t = np.asarray(t)
        self.y = np.asarray(y)
        self.system = system
        self.meta = dict(meta) if meta else {}
        self._kdtree: cKDTree | None = None

    # --- compatibility / convenience ---

    def __iter__(self) -> Iterator[np.ndarray]:
        """Allow ``t, y = trajectory``."""
        return iter((self.t, self.y))

    def __getitem__(self, key: Any) -> Any:
        """
        Component access by name, or joint row slicing.

        - ``traj["x"]`` → 1-D component array (requires the system class to
          declare ``variables``).
        - ``traj[["x", "z"]]`` → ``(T, 2)`` array.
        - anything else (int/slice/mask) slices ``t`` and ``y`` together and
          returns a new :class:`Trajectory`.
        """
        if isinstance(key, str):
            return self.y[:, self._component_index(key)]
        if isinstance(key, list | tuple) and key and all(isinstance(k, str) for k in key):
            return self.y[:, [self._component_index(k) for k in key]]
        if isinstance(key, int | np.integer):
            # Keep the result a well-formed Trajectory (one row), not a
            # corrupted one built from scalars.
            return Trajectory(
                np.atleast_1d(self.t[key]),
                np.atleast_2d(self.y[key]),
                self.system,
                meta=self.meta,
            )
        return Trajectory(self.t[key], self.y[key], self.system, meta=self.meta)

    @property
    def variables(self) -> tuple[str, ...] | None:
        """Component names declared by the system (instance attr or class ClassVar)."""
        if self.system is None:
            return None
        # Instance lookup falls back to the ClassVar for the built-in families,
        # and also honours per-instance names (e.g. WrappedSystem).
        return getattr(self.system, "variables", None)

    def _component_index(self, name: str) -> int:
        names = self.variables
        if names is None:
            raise KeyError(
                f"{type(self.system).__name__ if self.system else 'This system'} declares no "
                f"`variables`; use integer indexing or add e.g. variables = ('x', 'y', 'z') "
                f"to the system class."
            )
        try:
            return names.index(name)
        except ValueError:
            raise KeyError(f"Unknown component {name!r}. Declared variables: {names}") from None

    @property
    def dim(self) -> int:
        """State-space dimension."""
        return int(self.y.shape[1])

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return len(self.t)

    def component(self, i: int | str) -> np.ndarray:
        """
        Return a single state component.

        Parameters
        ----------
        i : int or str
            Component index, or component name when the system declares
            ``variables``.

        Returns
        -------
        ndarray, shape (T,)
        """
        if isinstance(i, str):
            i = self._component_index(i)
        return self.y[:, i]

    def after(self, t0: float) -> Trajectory:
        """
        Drop the initial transient.

        Parameters
        ----------
        t0 : float
            Keep only time points ``t >= t0``.

        Returns
        -------
        Trajectory
        """
        mask = self.t >= t0
        return Trajectory(self.t[mask], self.y[mask], self.system, meta=self.meta)

    # --- visualization seam ---

    def to_plot_spec(
        self,
        kind: str | None = None,
        *,
        components: int | str | Sequence[int | str] | None = None,
        animate: bool | dict[str, Any] | Animation = False,
        **kind_kw: Any,
    ) -> PlotSpec:
        """
        Describe this trajectory as a backend-agnostic :class:`PlotSpec`.

        This is the **one front door** for trajectory plotting — every common
        view goes through here, so the parameterised ``viz.producers`` builders
        stay an internal detail.

        Auto-dispatch
            With ``kind=None`` the semantic kind follows the number of selected
            components (after applying ``components=``): 1 → ``TIME_SERIES``,
            2 → ``PHASE_PORTRAIT_2D``, 3 → ``PHASE_PORTRAIT_3D``, and **4+ →
            ``SPACETIME``** (a Lorenz-96-style field image, *not* a misleading
            3-D portrait of the first three coordinates).  A discrete-map orbit
            draws with a ``SCATTER`` mark (a point sequence) rather than a line.
            A Poincaré-section trajectory (carrying ``meta["plot_kind"]`` of
            ``"poincare_section"``) is recognised and drawn as its in-plane
            scatter.

        Selecting components
            ``components=`` picks what to draw — a name, an index, or a sequence
            of them: ``components="x"`` (a single time series), ``components=
            ["y0", "y1", "y2"]`` (a 3-D portrait of three chosen channels).  The
            auto-dispatch then keys off how many you selected.

        Overriding the kind
            ``kind=`` forces any member of the closed
            :class:`~tsdynamics.viz.spec.PlotKind` vocabulary — e.g.
            ``kind="time_series"`` to overlay component-vs-time on a 3-D
            trajectory, or ``kind="spacetime"`` to image it.  The recipe
            ``kind="delay"`` builds a delay-coordinate embedding ``x(t)`` vs
            ``x(t - tau)``; pass ``tau`` (in **time units**) via ``**kind_kw``.

        Per-kind options (``**kind_kw``)
            Options valid for one kind only are accepted as keywords rather than
            cluttering the signature — ``tau`` (required for ``kind="delay"``,
            converted from time units to samples via ``meta["dt"]``),
            ``color_by={"time", "speed"}`` (time series / phase portraits),
            ``transpose`` (spacetime).  Passing one to the wrong kind raises
            :class:`~tsdynamics.errors.InvalidParameterError`.

        The :mod:`tsdynamics.viz` import is local to this method (lazy), and the
        spec carries no rendering code, so building a spec (or importing
        :mod:`tsdynamics`) never imports matplotlib / Plotly.

        Parameters
        ----------
        kind : str, optional
            Override the auto-dispatched kind (a ``PlotKind`` value, or the
            ``"delay"`` recipe).  ``None`` auto-dispatches.
        components : int or str or sequence of int/str, optional
            Which state components to draw (names or indices).  ``None`` uses all.
        animate : bool or dict or Animation, optional
            Turn the spec into a reveal animation.  ``True`` uses sensible per-kind
            defaults (a moving head on portraits / spacetime, off for a plain time
            series); a dict overrides individual
            :class:`~tsdynamics.viz.spec.Animation` fields; an
            :class:`~tsdynamics.viz.spec.Animation` is used as-is.  Tweak further
            with the chainable ``spec.animate()`` / ``.trail()`` / ``.head()`` /
            ``.camera()`` / ``.clock()`` methods.
        **kind_kw
            Per-kind options (see above).

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import PlotKind

        all_names = self.variables or tuple(f"y{i}" for i in range(self.dim))

        # A Poincaré section carries its intent in meta; honour it before the
        # dimensionality dispatch (only for the unmodified default view).
        if (
            kind is None
            and components is None
            and not kind_kw
            and str(self.meta.get("plot_kind", "")) == PlotKind.POINCARE_SECTION
        ):
            return self._with_animation(self._poincare_section_spec(all_names), animate)

        # The ``"field"`` / ``"spatial_field"`` recipe routes before component
        # resolution: here ``components=`` selects a *field block* (e.g. Gray–
        # Scott's "u"/"v"), not a state component, so it must not be resolved
        # against the per-cell labels.
        if kind is not None and _KIND_ALIASES.get(kind, kind) == "spatial_field":
            self._validate_kind_kw("spatial_field", kind_kw)
            field = self._spatial_field_spec(components)
            return self._with_animation(field, animate)

        sel = self._resolve_components(components, all_names)
        sel_names = tuple(all_names[i] for i in sel)
        n_sel = len(sel)
        discrete = self._is_discrete()

        # Resolve the routing key: a friendly alias (``"delay"``) first, else the
        # auto kind from the number of selected components.
        route = _KIND_ALIASES.get(kind, kind) if kind is not None else _auto_route(n_sel)
        self._validate_kind_kw(route, kind_kw)

        if route == "delay_embedding":
            delay = self._delay_spec(sel, sel_names, kind_kw, explicit=components is not None)
            return self._with_animation(delay, animate)

        spec_kind = PlotKind(route)  # "delay" never reaches here (aliased above)
        ys = self.y[:, sel]

        if spec_kind == PlotKind.SPACETIME:
            image = self._spacetime_spec(ys, sel_names, transpose=bool(kind_kw.get("transpose")))
            return self._with_animation(image, animate)

        color_by = kind_kw.get("color_by")
        if spec_kind == PlotKind.TIME_SERIES:
            series = self._time_series_spec(sel, ys, sel_names, discrete, color_by)
            return self._with_animation(series, animate)

        portrait = self._phase_portrait_spec(spec_kind, sel, ys, sel_names, discrete, color_by)
        return self._with_animation(portrait, animate)

    def _with_animation(
        self, spec: PlotSpec, animate: bool | dict[str, Any] | Animation
    ) -> PlotSpec:
        """Stamp an :class:`Animation` onto ``spec`` per the ``animate`` request.

        ``False``/``None`` leaves the spec static.  ``True`` uses per-kind defaults
        (head on except for a plain time series); a dict overrides individual
        :class:`~tsdynamics.viz.spec.Animation` fields; an ``Animation`` is used
        verbatim.  The spec carries ``meta["dt"]``, so time-unit trails / the clock
        resolve at render time.

        A :data:`~tsdynamics.viz.spec.PlotKind.SPATIAL_FIELD` spec always animates
        in the **frames** model — the field *movie*: each frame is the spatial
        state at that instant (a 1-D profile line or a 2-D heatmap), so consecutive
        frames carry genuinely different data.  No comet window, no head marker; the
        field itself is the motion.
        """
        if animate is False or animate is None:
            return spec
        from dataclasses import replace as _dc_replace

        from tsdynamics.viz.spec import Animation as _Animation
        from tsdynamics.viz.spec import PlotKind

        if spec.kind == PlotKind.SPATIAL_FIELD:
            # The spatial-field movie is the frames model — every frame is a fresh
            # spatial snapshot.  Force ``mode="frames"`` and the field defaults (no
            # comet window, no head), letting the caller's own keys still win.
            field_defaults: dict[str, Any] = {
                "mode": "frames",
                "head": False,
                "trail_kind": None,
            }
            if isinstance(animate, _Animation):
                # Use dataclasses.replace — never mutate the caller's Animation.
                spec.animation = _dc_replace(
                    animate, mode="frames", head=False, trail_kind=None, trail_length=None
                )
            elif isinstance(animate, dict):
                spec.animation = _Animation(**{**field_defaults, **animate})
            else:  # truthy (e.g. ``True``)
                spec.animation = _Animation(**field_defaults)
            return spec

        head_default = spec.kind != PlotKind.TIME_SERIES
        # Default to a *windowed* comet (a moving trail + head over the full faint
        # curve) — smooth, small, and rotatable in plotly; a persistent "draw-it-in"
        # is one ``.trail(length=None)`` away.  A tight window (≈ 1/10 of the series,
        # capped) keeps the comet crisp and the exported HTML small.
        defaults: dict[str, Any] = {
            "head": head_default,
            "trail_kind": "steps",
            "trail_length": float(max(2, min(self.n_steps // 10, 200))),
        }
        if isinstance(animate, _Animation):
            # Copy — never store (and later mutate) the caller's Animation by
            # reference, so a subsequent ``.trail()``/``.head()`` on the spec does
            # not reach back and mutate the caller's object (matches the
            # SPATIAL_FIELD branch, which already copies via ``replace``).
            spec.animation = _dc_replace(animate)
        elif isinstance(animate, dict):
            spec.animation = _Animation(**{**defaults, **animate})
        else:  # truthy (e.g. ``True``)
            spec.animation = _Animation(**defaults)
        return spec

    def _resolve_components(
        self, components: int | str | Sequence[int | str] | None, names: tuple[str, ...]
    ) -> list[int]:
        """Resolve a ``components=`` selector to a list of column indices.

        ``None`` selects every component; a lone name/index is wrapped; names
        resolve against ``names`` (the declared or generated ``y0…`` labels).
        """
        from tsdynamics.errors import InvalidParameterError

        if components is None:
            return list(range(self.dim))
        items: Sequence[int | str]
        items = (components,) if isinstance(components, (int, str, np.integer)) else components
        out: list[int] = []
        for c in items:
            if isinstance(c, str):
                if c not in names:
                    raise InvalidParameterError(
                        f"unknown component {c!r}; available components: {list(names)}"
                    )
                out.append(names.index(c))
            else:
                idx = int(c)
                if not -self.dim <= idx < self.dim:
                    raise InvalidParameterError(
                        f"component index {idx} out of range for a {self.dim}-D trajectory"
                    )
                out.append(idx % self.dim)
        if not out:
            raise InvalidParameterError("components= selected no channels; pass at least one.")
        return out

    @staticmethod
    def _validate_kind_kw(route: str | None, kind_kw: dict[str, Any]) -> None:
        """Reject per-kind options passed to the wrong kind (and require ``tau``)."""
        from tsdynamics.errors import InvalidParameterError

        allowed = _KIND_KW.get(route or "", frozenset())
        unknown = set(kind_kw) - allowed
        if unknown:
            raise InvalidParameterError(
                f"kind={route!r} does not accept keyword(s) {sorted(unknown)}; "
                f"allowed here: {sorted(allowed) or '(none)'}"
            )
        if route == "delay_embedding" and "tau" not in kind_kw:
            raise InvalidParameterError("kind='delay' requires tau=<delay in time units>")

    def _delay_tau_samples(self, tau: float) -> int:
        """Convert a delay ``tau`` in **time units** to an integer sample lag."""
        from tsdynamics.errors import InvalidParameterError

        if not np.isfinite(float(tau)) or float(tau) <= 0:
            raise InvalidParameterError(f"delay tau must be a positive, finite time, got {tau!r}.")
        dt = self.meta.get("dt")
        dt_f = float(dt) if dt is not None else None
        if dt_f is None or not np.isfinite(dt_f) or dt_f <= 0:
            diffs = np.diff(self.t)
            dt_f = float(np.median(diffs)) if diffs.size else None
        if dt_f is None or dt_f <= 0:
            raise InvalidParameterError(
                "cannot convert delay tau to samples: the trajectory carries no "
                "'dt' in meta and its time grid is degenerate."
            )
        samples = max(1, int(round(float(tau) / dt_f)))
        if samples >= self.n_steps:
            raise InvalidParameterError(
                f"delay tau={tau} (→ {samples} samples at dt={dt_f:g}) must be shorter "
                f"than the series length {self.n_steps}."
            )
        return samples

    def _delay_spec(
        self,
        sel: list[int],
        sel_names: tuple[str, ...],
        kind_kw: dict[str, Any],
        *,
        explicit: bool,
    ) -> PlotSpec:
        """Build the ``x(t)`` vs ``x(t - tau)`` delay embedding (a ``PHASE_PORTRAIT_2D``).

        A delay embedding reconstructs **one** scalar observable; with no
        ``components=`` it embeds the first component, but an explicit
        multi-component selection is rejected rather than silently dropped.
        """
        from tsdynamics.errors import InvalidParameterError
        from tsdynamics.viz import producers

        if explicit and len(sel) != 1:
            raise InvalidParameterError(
                "kind='delay' embeds a single component; select exactly one via "
                f"components= (got {len(sel)})."
            )
        samples = self._delay_tau_samples(kind_kw["tau"])
        return producers.delay_embedding(self, tau=samples, component=sel[0], label=sel_names[0])

    def _time_series_spec(
        self,
        sel: list[int],
        ys: np.ndarray,
        sel_names: tuple[str, ...],
        discrete: bool,
        color_by: Any,
    ) -> PlotSpec:
        """Overlay one component-vs-time layer per selected component."""
        from tsdynamics.viz.spec import Axis, Layer, Legend, PlotKind, PlotSpec

        if color_by is not None:
            from tsdynamics.viz import producers

            return producers.time_series(self, components=sel, color_by=color_by)
        mark = PlotKind.SCATTER if discrete else PlotKind.LINE
        layers = [
            Layer(mark, {"x": self.t, "y": ys[:, k]}, label=sel_names[k]) for k in range(len(sel))
        ]
        return PlotSpec(
            kind=PlotKind.TIME_SERIES,
            ndim=1,
            title=self._title(),
            x=Axis(label="t"),
            y=Axis(label=sel_names[0]),
            layers=layers,
            legend=Legend() if len(sel) > 1 else None,
            meta=dict(self.meta),
        )

    def _phase_portrait_spec(
        self,
        spec_kind: Any,
        sel: list[int],
        ys: np.ndarray,
        sel_names: tuple[str, ...],
        discrete: bool,
        color_by: Any,
    ) -> PlotSpec:
        """Build a 2-D / 3-D phase portrait over the selected components."""
        from tsdynamics.errors import InvalidParameterError
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        # Shape (2-D vs 3-D) follows the resolved kind, not the raw selection —
        # forcing ``kind="phase_portrait_2d"`` yields a clean 2-D schema (no z).
        want_3d = spec_kind == PlotKind.PHASE_PORTRAIT_3D
        need = 3 if want_3d else 2
        if len(sel) < need:
            raise InvalidParameterError(
                f"kind={spec_kind.value!r} needs at least {need} components, but "
                f"{len(sel)} were selected; use 'time_series' or select more."
            )
        if color_by is not None:
            from tsdynamics.viz import producers

            return producers.phase_portrait(self, components=sel[:need], color_by=color_by)
        cols: dict[str, np.ndarray] = {"x": ys[:, 0], "y": ys[:, 1]}
        z = Axis(label=sel_names[2]) if want_3d else None
        if want_3d:
            cols["z"] = ys[:, 2]
        flow_mark = PlotKind.LINE3D if want_3d else PlotKind.LINE
        layer_kind = PlotKind.SCATTER if discrete else flow_mark
        return PlotSpec(
            kind=spec_kind,
            ndim=3 if want_3d else 2,
            aspect="equal",
            title=self._title(),
            x=Axis(label=sel_names[0]),
            y=Axis(label=sel_names[1]),
            z=z,
            layers=[Layer(layer_kind, cols)],
            meta=dict(self.meta),
        )

    def _is_discrete(self) -> bool:
        """Whether the producing system is a discrete map (default ``False``).

        Reads ``self.system.is_discrete`` defensively — a synthetic trajectory
        with no system, or one whose system does not expose the flag, is treated
        as a flow.
        """
        flag = getattr(self.system, "is_discrete", False)
        try:
            return bool(flag)
        except Exception:  # pragma: no cover - defensive
            return False

    def _spacetime_spec(
        self, ys: np.ndarray, names: tuple[str, ...], *, transpose: bool = False
    ) -> PlotSpec:
        """Build the component-index vs time ``IMAGE`` spec (``SPACETIME``).

        The spatiotemporal field view of a high-dimensional flow (a Lorenz-96
        lattice): the selected columns are drawn as a single color-mapped
        ``IMAGE`` with time along ``x`` and component index along ``y`` (or the
        axes swapped when ``transpose=True``); the colorbar / ``clim`` are
        inferred from the field.
        """
        from tsdynamics.viz.spec import Axis, Colorbar, Layer, PlotKind, PlotSpec

        comp_idx = np.arange(ys.shape[1], dtype=float)
        if transpose:
            img = ys
            x_axis, y_axis = Axis(label="component"), Axis(label="t")
            x_data, y_data = comp_idx, self.t
        else:
            img = ys.T
            x_axis, y_axis = Axis(label="t"), Axis(label="component")
            x_data, y_data = self.t, comp_idx
        layer = Layer(PlotKind.IMAGE, {"x": x_data, "y": y_data, "c": img.ravel(), "z": img})
        spec = PlotSpec(
            kind=PlotKind.SPACETIME,
            ndim=2,
            title=self._title(),
            x=x_axis,
            y=y_axis,
            layers=[layer],
            colorbar=Colorbar(label="state"),
            # ``time_axis`` records which image axis runs in time so an
            # animate={"mode":"frames"} movie grows along time under either
            # orientation (rows when transposed, columns otherwise).
            meta={
                **dict(self.meta),
                "component_names": list(names),
                "time_axis": "row" if transpose else "col",
            },
        )
        return spec.autocolor()

    def _spatial_field_spec(self, component: int | str | Sequence[int | str] | None) -> PlotSpec:
        """Build a :data:`SPATIAL_FIELD` spec via the ``spatial_field`` producer.

        The spatial grid is read from ``meta["field_shape"]`` (recorded by a system
        declaring ``_field_shape``); ``component`` selects a field block
        (Gray–Scott's ``"u"`` / ``"v"``).  A bare field with no grid metadata falls
        back to a 1-D profile.
        """
        from tsdynamics.errors import InvalidParameterError
        from tsdynamics.viz import producers

        block: int | str | None
        if component is None:
            block = None
        elif isinstance(component, (int, str, np.integer)):
            block = int(component) if isinstance(component, np.integer) else component
        else:  # a sequence — a field movie plots exactly one block
            items = list(component)
            if len(items) != 1:
                raise InvalidParameterError(
                    "kind='field' plots a single field block; select exactly one via "
                    f"components= (got {len(items)})."
                )
            block = items[0]
        return producers.spatial_field(self, component=block)

    def plot(self, backend: str | None = None, **kwargs: Any) -> Any:
        """Render this trajectory via a visualization backend.

        Sugar over :meth:`to_plot_spec`: the spec-shaping keywords (``kind``,
        ``components``, and the per-kind options ``tau`` / ``color_by`` /
        ``transpose``) are peeled off and passed to :meth:`to_plot_spec`; the
        remaining keywords are inline spec tweaks (``xlabel`` / ``yscale`` /
        ``title`` / …) or backend keyword arguments (see
        :meth:`tsdynamics.viz.spec.Plottable.plot`).

        The viz package is imported lazily here (not at module scope) so plain
        ``import tsdynamics`` never pulls it in — honouring the
        no-backend-on-import contract.  Raises
        :class:`~tsdynamics.viz.spec.VisualizationNotInstalled` until a backend
        is registered.
        """
        from tsdynamics.viz.spec import _apply_inline_tweaks

        spec_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in _PLOT_SPEC_KEYS}
        spec = self.to_plot_spec(**spec_kw)
        backend_kw = _apply_inline_tweaks(spec, kwargs)
        return spec.render(backend, **backend_kw)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """Notebook display hook — lazily delegated to ``Plottable`` (see :meth:`plot`).

        No-ops (returns ``None``) until a backend is installed, so a trajectory
        still reprs as text in a plain console and ``import`` stays backend-free.
        """
        from tsdynamics.viz.spec import Plottable

        return Plottable._repr_mimebundle_(cast("Plottable", self), include, exclude)

    def _poincare_section_spec(self, names: tuple[str, ...]) -> PlotSpec:
        """Build the 2-D in-plane scatter spec for a Poincaré-section trajectory.

        Projects the recorded crossing states onto the section plane (dropping
        the normal coordinate) and picks the two in-plane axes with the largest
        spread to display — so the section reads as a 2-D point cloud, not a 3-D
        flow.  Falls back to the first two components if the plane is unavailable.
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        i, j = self._section_axes()
        layers = [Layer(PlotKind.SCATTER, {"x": self.y[:, i], "y": self.y[:, j]})]
        return PlotSpec(
            kind=PlotKind.POINCARE_SECTION,
            ndim=2,
            aspect="equal",
            title=self._title("Poincaré section"),
            x=Axis(label=names[i]),
            y=Axis(label=names[j]),
            layers=layers,
            meta=dict(self.meta),
        )

    def _section_axes(self) -> tuple[int, int]:
        """Pick the two in-plane display axes for a Poincaré section.

        Drops the coordinate the section plane fixes (read from ``meta["plane"]``
        as ``(index, value)`` when present) and, of the remaining coordinates,
        keeps the two with the largest range so the scatter is maximally
        informative.  Defaults to ``(0, 1)`` when the plane / extra columns are
        unavailable.
        """
        plane = self.meta.get("plane")
        normal_idx: int | None = None
        if isinstance(plane, (tuple, list)) and len(plane) == 2 and np.isscalar(plane[0]):
            normal_idx = int(cast(Any, plane[0]))
        candidates = [c for c in range(self.dim) if c != normal_idx]
        if len(candidates) < 2:
            candidates = list(range(self.dim))[:2]
        if len(candidates) < 2:
            return 0, 0 if self.dim == 1 else 1
        if self.y.shape[0] == 0:
            # An empty section (the plane caught no crossings) has no spread to
            # rank — a reduction over the zero-size axis would raise. Keep the
            # first two in-plane candidates so the section still yields a valid
            # (empty) 2-D scatter spec.
            i, j = candidates[:2]
            return (i, j) if i < j else (j, i)
        spreads = self.y.max(axis=0) - self.y.min(axis=0)
        i, j = sorted(candidates, key=lambda c: spreads[c], reverse=True)[:2]
        return (i, j) if i < j else (j, i)

    def _title(self, prefix: str | None = None) -> str:
        """Compose a title from the originating system name and an optional prefix."""
        system = self.meta.get("system")
        name = str(system) if system else ""
        if prefix and name:
            return f"{prefix} — {name}"
        return prefix or name

    # --- point-set operations ---

    def minmax(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-component ``(minima, maxima)``, each of shape ``(dim,)``."""
        return self.y.min(axis=0), self.y.max(axis=0)

    def standardize(self) -> Trajectory:
        """
        Return a copy with zero mean and unit standard deviation per component.

        The applied transform is recorded in ``meta["standardized"]``.
        """
        mean = self.y.mean(axis=0)
        std = self.y.std(axis=0)
        std = np.where(std < np.finfo(float).tiny, 1.0, std)
        return Trajectory(
            self.t,
            (self.y - mean) / std,
            self.system,
            meta={**self.meta, "standardized": {"mean": mean, "std": std}},
        )

    def neighbors(self, q: Any, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Nearest trajectory points to query point(s) ``q``.

        Builds a KD-tree lazily on first call and caches it; subsequent
        queries are O(log T).

        Parameters
        ----------
        q : array-like, shape (dim,) or (m, dim)
            Query point(s).
        k : int
            Number of neighbours per query point.

        Returns
        -------
        (distances, indices)
            As returned by :meth:`scipy.spatial.cKDTree.query`.
        """
        from scipy.spatial import cKDTree

        if self._kdtree is None:
            self._kdtree = cKDTree(self.y)
        return cast(
            "tuple[np.ndarray, np.ndarray]",
            self._kdtree.query(np.asarray(q, dtype=float), k=k),
        )

    def set_distance(self, other: Any, *, method: str = "centroid") -> float:
        """
        Distance to another point set (Trajectory or array), as a set.

        ``method`` is ``"centroid"`` (default), ``"hausdorff"``, or
        ``"minimum"`` — see :func:`tsdynamics.data.set_distance`.  The
        matching primitive behind attractor deduplication and continuation.
        """
        from tsdynamics.data import set_distance

        return set_distance(
            self, other, method=cast('Literal["centroid", "hausdorff", "minimum"]', method)
        )

    def __repr__(self) -> str:
        return (
            f"Trajectory(n_steps={self.n_steps}, dim={self.dim}, "
            f"t=[{self.t[0]:.3g}, {self.t[-1]:.3g}])"
        )
