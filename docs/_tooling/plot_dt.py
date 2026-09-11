"""Pick a smooth, non-pixelated output ``dt`` for a system's documentation plot.

Both attractor renderers — the interactive viewer (:mod:`threejs_viewer`) and the
static figure renderer (:mod:`figures`) — need an output sampling step that keeps
the drawn curve **smooth** rather than faceted, while staying coarse enough that
the inlined WebGL payload / PNG stays light.  :func:`choose_plot_dt` is the *one*
canonical selector both call: it runs a short, deliberately **fine** *pilot*
integration of the system and hands its samples to
:func:`tsdynamics.analysis.sampling.estimate_dt_from_sagitta`, which returns the
largest stride whose per-triple *sagitta* (the bow of the curve off its local
chord) still stays inside a geometric tolerance ``epsilon`` — i.e. the coarsest
output step that does not visibly straighten the curve.

The universal sagitta rule
--------------------------
The maintainer's rule: *"per system calculate the sagitta with error 0.01, that
yields the dt, integrate with that — guarantees no pixelated attractor."*  So the
**default** ``epsilon`` here is ``0.01`` (tightened from the earlier ``0.1``,
which still left the fast attractors faceted) and every attractor integration
(three.js and static alike) should sample at the ``dt`` this returns.

Crucially the pilot is integrated at a **fine** step (``dt0``), *finer* than any
plausible output step, so the sagitta search has room to find the true output
``dt`` **from below**.  This is what fixes the pixelated group (Chen, DequanLi,
QiChen, YuWang, …): those are fast attractors whose tight curvature demands a
``dt`` *finer* than the naïve ``0.01`` — a pilot integrated at ``0.01`` can only
ever report "``0.01`` is already too coarse", never the finer step that actually
reads smooth.  A sagitta search can never report a step *finer than the pilot it
ran*, so at ``epsilon=0.01`` the fastest attractors (DequanLi, QiChen, HyperQi)
saturate the pilot floor — the ODE pilot is therefore integrated at ``0.001`` (was
``0.002``) so a ``0.01``-sagitta step is actually achievable and those systems get
their true fine output ``dt`` (``0.001`` for DequanLi/QiChen/HyperQi, ``0.004`` for
Chen, and, at the other end, ``0.064`` for the slow Rössler).

Robustness
----------
The helper never raises: an editorial ``plot_dt`` override short-circuits the
heuristic, maps and SDEs are no-ops (a map is iterated by *steps*; an SDE sample
path is not an attractor curve), and **any** failure (a system that will not
integrate from a random start, a too-short pilot, an import problem) falls back to
``dt0`` — a docs build must never break on the choice of a plotting step.  The
pilot + its chosen ``dt`` are memoised so the two renderers do not integrate the
same system twice.

Public API
----------
``choose_plot_dt(entry, *, final_time=None, dt0=None, epsilon=0.01)`` -> ``float``
    The canonical smooth-output-``dt`` selector.
``FINE_PILOT_DT`` / ``PILOT_FINAL_TIME``
    The per-family fine-pilot defaults (also the fallback ``dt``).
``clear_cache()``
    Drop the memoised pilots (tests / a fresh build).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Per-family fine-pilot defaults
# ---------------------------------------------------------------------------
#: Fine integration step for the pilot, per family.  Deliberately *finer* than any
#: plausible output ``dt`` so :func:`estimate_dt_from_sagitta` can find the true
#: output step from below (a coarse pilot cannot report a step finer than itself).
#: These values are also the robust *fallback* ``dt`` when the heuristic cannot run.
FINE_PILOT_DT: dict[str, float] = {
    "ode": 0.001,  # fine enough that an ε=0.01 sagitta step is achievable for a fast attractor
    "dde": 0.02,
    "sde": 0.01,
    "map": 1.0,  # a map is iterated by steps — dt is nominal (never used to smooth)
}

#: Pilot integration window per family — long enough to trace the attractor, short
#: enough that a fine-``dt`` pilot over the whole catalogue stays a couple of seconds.
PILOT_FINAL_TIME: dict[str, float] = {
    "ode": 40.0,
    "dde": 320.0,
    "sde": 100.0,
    "map": 0.0,  # unused for maps
}

#: Drop this leading fraction of the pilot as transient before measuring sagitta
#: (mirrors the renderers' own transient trim so the chosen dt matches what is drawn).
_TRANSIENT_FRAC = 0.15

#: IC-retry budget for an off-basin random start (mirrors the renderers).
_IC_RETRIES = 6

#: A non-equilibrium DDE history (mirrors :mod:`figures` / :mod:`threejs_viewer`).
_DDE_HISTORY_AMP = 0.2
_DDE_HISTORY_OFF = 0.8
_DDE_HISTORY_FREQ = 0.2

_EDITORIAL_PATH = Path(__file__).with_name("editorial.json")


# ---------------------------------------------------------------------------
# Editorial override lookup
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _editorial_plot_dt() -> dict[str, float]:
    """``{system name: plot_dt}`` from ``editorial.json`` (empty on any problem)."""
    try:
        with _EDITORIAL_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        systems = data.get("systems", {}) if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}
    out: dict[str, float] = {}
    for name, ed in systems.items():
        if not isinstance(ed, dict):
            continue
        val = ed.get("plot_dt")
        try:
            if val is not None and float(val) > 0.0:
                out[str(name)] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def _override_for(entry: Any) -> float | None:
    """Resolve an editorial ``plot_dt`` override for ``entry``, or ``None``.

    Accepts a rich catalogue ``SystemRecord`` (carries a ``.plot_dt`` attribute,
    already merged from editorial) *or* a bare registry ``SystemEntry`` (looked up
    by name in ``editorial.json``).  Only a positive float counts.
    """
    rec_override = getattr(entry, "plot_dt", None)
    try:
        if rec_override is not None and float(rec_override) > 0.0:
            return float(rec_override)
    except (TypeError, ValueError):
        pass
    name = getattr(entry, "name", None)
    if name is not None:
        return _editorial_plot_dt().get(str(name))
    return None


# ---------------------------------------------------------------------------
# Renderer-shared knobs (IC resolution + per-system figure overrides)
# ---------------------------------------------------------------------------
def _fig_overrides(name: str) -> dict[str, Any]:
    """Per-system :data:`figures.FIG_OVERRIDES` (``ic`` / ``final_time`` / ``dt``).

    Import is deferred and best-effort: :mod:`plot_dt` must stay import-safe even
    when the ``docs/_tooling`` siblings are not on the path.
    """
    try:
        import figures  # docs/_tooling sibling

        opts = figures.FIG_OVERRIDES.get(name, {})
        return opts if isinstance(opts, dict) else {}
    except Exception:  # noqa: BLE001 — no siblings on path / import error
        return {}


@lru_cache(maxsize=1)
def _editorial_viewers() -> dict[str, dict[str, Any]]:
    """``{system name: viewer directive}`` from ``editorial.json`` (empty on error)."""
    try:
        with _EDITORIAL_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        systems = data.get("systems", {}) if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, ed in systems.items():
        if isinstance(ed, dict) and isinstance(ed.get("viewer"), dict):
            out[str(name)] = ed["viewer"]
    return out


def _viewer_cfg(entry: Any) -> dict[str, Any]:
    """Resolve the editorial ``viewer`` directive for ``entry`` (or ``{}``).

    Accepts a rich catalogue ``SystemRecord`` (carries a ``.viewer`` dict) *or* a
    bare registry ``SystemEntry`` (looked up by name in ``editorial.json``).  The
    viewer block is where a Group-B "weird" system pins the on-attractor ``ic`` /
    ``final_time`` / ``method`` — the pilot must integrate that same trajectory or
    the sagitta ``dt`` describes a *different* (often degenerate) orbit than the
    one the renderer draws.
    """
    cfg = getattr(entry, "viewer", None)
    if isinstance(cfg, dict) and cfg:
        return cfg
    name = getattr(entry, "name", None)
    if name is not None:
        return _editorial_viewers().get(str(name), {})
    return {}


def _resolve_ic(sys_obj: Any, override: Any):
    """Resolve a pilot IC, honouring the same contract the renderers use.

    Prefers :func:`figures._resolve_ic` (``"0.1*ones"`` sentinel + class
    ``default_ic``) so a finite-basin system (RabinovichFabrikant, Sprott*, …)
    whose random start escapes still lands on its attractor.  Falls back to a
    local reimplementation if the sibling import is unavailable.
    """
    try:
        import figures  # docs/_tooling sibling

        return figures._resolve_ic(sys_obj, override)
    except Exception:  # noqa: BLE001 — local fallback
        if override == "0.1*ones":
            return 0.1 * np.ones(sys_obj.dim)
        if override is not None:
            return np.asarray(override, dtype=float)
        default_ic = getattr(type(sys_obj), "default_ic", None)
        if default_ic is not None:
            return np.asarray(default_ic, dtype=float).reshape(sys_obj.dim)
        return None


# ---------------------------------------------------------------------------
# Pilot integration (per family)
# ---------------------------------------------------------------------------
def _ode_pilot(entry: Any, final_time: float, dt0: float) -> np.ndarray | None:
    """Integrate a bounded ODE pilot at the fine step ``dt0`` (``(n, dim)`` or ``None``).

    Marches with the fixed-step ``rk4`` kernel (a divergent off-basin start raises
    promptly and is retried, rather than sending the adaptive controller into a
    step-shrinking spiral).  Honours a class ``default_ic`` / a
    :data:`figures.FIG_OVERRIDES` ``ic`` and drops the leading transient — exactly
    the trajectory the renderers draw, so the chosen ``dt`` matches what is shown.
    """
    opts = _fig_overrides(getattr(entry, "name", ""))
    vcfg = _viewer_cfg(entry)
    # A Group-B system pins its on-attractor start in the editorial ``viewer``
    # block; honour that (and its adaptive ``method``) so the pilot integrates the
    # *same* orbit the renderer draws.  Otherwise a random start can land on a slow
    # transient / near-fixed manifold and sagitta reports an absurdly coarse dt
    # (ItikBanksTumor: a random start decays to a near-equilibrium ⇒ dt≈114).
    ic_override = vcfg.get("ic", opts.get("ic"))
    method = str(vcfg.get("method") or opts.get("engine_method") or "rk4")
    rng = np.random.default_rng(42)
    sys_obj = entry.cls()
    ic = _resolve_ic(sys_obj, ic_override)
    for attempt in range(_IC_RETRIES):
        if ic is None or attempt > 0:
            ic = sys_obj.resolve_ic(rng.uniform(0.0, 1.0, sys_obj.dim))
        try:
            traj = sys_obj.run(
                final_time=final_time,
                dt=dt0,
                ic=np.asarray(ic, dtype=float),
                backend="interp",
                solver=method,
            )
        except (RuntimeError, ValueError):  # divergence / off-basin start
            ic = None
            continue
        y = np.asarray(traj.y, dtype=float)
        if y.ndim == 2 and len(y) > 50 and np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1e6:
            drop = int(_TRANSIENT_FRAC * len(y))
            return y[drop:]
        ic = None
    return None


def _dde_pilot(entry: Any, final_time: float, dt0: float) -> np.ndarray | None:
    """Build the 2-D delay embedding ``[x(t), x(t-τ)]`` pilot for a scalar DDE.

    The DDE viewer / figure draw the delay embedding, so the smoothing criterion
    lives there too.  ``τ`` comes from the live instance (``_delays()``).
    """
    sys_obj = entry.cls()

    def history(s: float) -> list[float]:
        return [_DDE_HISTORY_OFF + _DDE_HISTORY_AMP * np.sin(_DDE_HISTORY_FREQ * s)] * sys_obj.dim

    try:
        traj = sys_obj.run(final_time=final_time, dt=dt0, history=history)
    except (RuntimeError, ValueError):
        return None
    x = np.asarray(traj.y[:, 0], dtype=float)
    if x.size < 64 or not np.all(np.isfinite(x)):
        return None
    try:
        tau = float(sys_obj._delays()[0])
    except Exception:  # noqa: BLE001 — no resolvable delay
        return None
    lag = max(1, int(round(tau / dt0)))
    if lag >= x.size - 8:
        return None
    drop = int(_TRANSIENT_FRAC * x.size)
    return np.column_stack([x[lag:], x[:-lag]])[drop:]


# ---------------------------------------------------------------------------
# The canonical selector
# ---------------------------------------------------------------------------
def _family_defaults(family: str) -> tuple[float, float]:
    """``(final_time, dt0)`` fine-pilot defaults for a family (ODE fallback)."""
    return (
        PILOT_FINAL_TIME.get(family, PILOT_FINAL_TIME["ode"]),
        FINE_PILOT_DT.get(family, FINE_PILOT_DT["ode"]),
    )


@lru_cache(maxsize=512)
def _cached_choice(name: str, family: str, final_time: float, dt0: float, epsilon: float) -> float:
    """Memoised heavy path — integrate the pilot + run sagitta (keyed by name)."""
    from tsdynamics import registry

    entry = None
    for e in registry.all_systems():
        if e.name == name:
            entry = e
            break
    if entry is None:
        return dt0

    from tsdynamics.analysis.sampling import estimate_dt_from_sagitta

    if family == "dde":
        pilot = _dde_pilot(entry, final_time, dt0)
    else:
        pilot = _ode_pilot(entry, final_time, dt0)
    if pilot is None or pilot.ndim != 2 or len(pilot) < 50:
        return dt0

    # Feed at most the first three components to the geometric criterion (a 3-D
    # portrait is the most a comet ever draws); multivariate input is used directly.
    sample = pilot[:, : min(3, pilot.shape[1])]
    try:
        result = estimate_dt_from_sagitta(sample, float(dt0), epsilon=float(epsilon))
    except Exception:  # noqa: BLE001 — heuristic is best-effort
        return dt0
    dt = float(result.delta_t)
    if not (np.isfinite(dt) and dt > 0):
        return dt0
    # The sagitta dt is >= the fine pilot dt0 by construction (coarsen-only); the
    # fine pilot is precisely why a fast attractor can land on a dt *finer* than the
    # naive 0.01 default.  Never coarser than a whole pilot window.
    return min(dt, final_time)


def choose_plot_dt(
    entry: Any,
    *,
    final_time: float | None = None,
    dt0: float | None = None,
    epsilon: float = 0.01,
) -> float:
    """Return a smooth, non-pixelated output ``dt`` for ``entry``'s attractor plot.

    This is the **one** selector both :mod:`figures` and :mod:`threejs_viewer` call
    for every attractor integration.  Resolution order:

    1. an editorial ``plot_dt`` override (verbatim, if a positive float);
    2. a family no-op — **maps** (steps-based) and **SDEs** (sample paths, not
       attractor curves) return ``dt0``;
    3. otherwise a **fine** pilot integration + the sagitta heuristic
       (:func:`~tsdynamics.analysis.sampling.estimate_dt_from_sagitta` at
       ``epsilon``), returning the coarsest output ``dt`` that keeps the curve
       smooth — which for a fast attractor may be *finer* than the usual ``0.01``.

    Parameters
    ----------
    entry
        A registry ``SystemEntry`` *or* a catalogue ``SystemRecord`` (duck-typed on
        ``.cls`` / ``.family`` / ``.name``; a ``SystemRecord`` also supplies its
        merged ``.plot_dt`` override directly).
    final_time, dt0
        Pilot window and *fine* base step.  ``None`` (the default) resolves to the
        per-family :data:`PILOT_FINAL_TIME` / :data:`FINE_PILOT_DT`.  ``dt0`` is
        also the robust fallback returned on any failure.
    epsilon
        Geometric (sagitta) tolerance — larger ⇒ coarser ``dt``.  Default ``0.01``
        (the maintainer's universal "no pixelated attractor" rule); the ODE fine
        pilot (``0.001``) is deliberately finer than this so the tolerance is
        actually reachable for a fast, tightly-curved attractor.

    Returns
    -------
    float
        The chosen output ``dt`` (a positive float), or ``dt0`` on any failure.
        Never raises.
    """
    family = str(getattr(entry, "family", "ode"))
    default_ft, default_dt0 = _family_defaults(family)
    if final_time is None:
        final_time = default_ft
    if dt0 is None:
        dt0 = default_dt0

    # 1) Editorial override wins outright.
    try:
        override = _override_for(entry)
    except Exception:  # noqa: BLE001 — editorial is decoration, never load-bearing
        override = None
    if override is not None:
        return override

    # A figure ``dt`` override (a curated per-system step) is authoritative too — the
    # renderer draws that trajectory, so the smooth dt must not fight it.
    fig_dt = _fig_overrides(str(getattr(entry, "name", ""))).get("dt")
    try:
        if fig_dt is not None and float(fig_dt) > 0.0:
            return float(fig_dt)
    except (TypeError, ValueError):
        pass

    # 2) Family no-ops: maps are iterated by steps, SDE sample paths are not curves.
    if family in ("map", "sde"):
        return float(dt0)

    # 3) The sagitta heuristic (memoised).  Any failure falls back to ``dt0``.
    try:
        name = str(entry.name)
        return _cached_choice(name, family, float(final_time), float(dt0), float(epsilon))
    except Exception:  # noqa: BLE001 — a plotting-step choice must never break the build
        return float(dt0)


def clear_cache() -> None:
    """Drop the memoised pilots / choices (tests, or a fresh build)."""
    _cached_choice.cache_clear()
    _editorial_plot_dt.cache_clear()
