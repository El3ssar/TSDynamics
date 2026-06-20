"""``method=`` resolution + auto-stiffness selection (stream C-SOLV).

This is the selection layer that sits on top of the F2 registry mechanism.  It
turns a user-supplied ``method=`` string into a concrete, validated kernel and
decides, when asked, *which* kernel to use:

* :func:`resolve` — normalise a ``method=`` name (case + punctuation + common
  aliases), validate it against the registry and the problem family, and return
  a :class:`Resolution` carrying the spec and the ``with_jacobian`` decision.
* :func:`select` / :func:`default_method` — the *policy* half: given a family
  and a stiffness verdict, name the kernel to use.
* :func:`is_stiff` / :func:`recommend` — a cheap a-priori **auto-stiffness**
  detector from the Jacobian spectrum, so ``recommend(system)`` picks an
  implicit kernel on a stiff RHS and an explicit one otherwise.
* :func:`build_kwargs` / :func:`needs_jacobian` — the hook the family
  engine-dispatch seam (stream C-FAM) calls so an implicit method automatically
  builds a Jacobian-carrying tape (``with_jacobian=True``), satisfying the
  engine's Jacobian guard (PR #74) instead of raising.

The registry itself (``register`` / ``get`` / ``available``) and the kernel
*specs* live in the package ``__init__`` and the ``explicit`` / ``implicit`` /
``stochastic`` spec modules; this module only *reads* the registry, so it stays
correct as plugins add solvers — a plugin-registered name resolves here exactly
like a built-in one (ROADMAP §4d / D4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from . import SolverSpec, all_specs, get

if TYPE_CHECKING:  # avoid importing the families package at solver-import time
    from ..families import SystemBase

__all__ = [
    "DEFAULT_METHOD",
    "STIFF_METHOD",
    "Resolution",
    "available_for",
    "build_kwargs",
    "default_method",
    "is_implicit",
    "is_stiff",
    "needs_jacobian",
    "normalize",
    "recommend",
    "resolve",
    "select",
]


# ---------------------------------------------------------------------------
# Name normalisation + aliases
# ---------------------------------------------------------------------------

#: Common alternate spellings → canonical kernel name.  Keys are already
#: *normalised* (see :func:`normalize`): lower-cased with runs of whitespace and
#: ``-`` collapsed to a single ``_``.  The canonical names themselves
#: (``"rk45"`` …) need no entry — they pass straight through.
_ALIASES: dict[str, str] = {
    # explicit RK
    "dopri5": "rk45",
    "dopri": "rk45",
    "dp45": "rk45",
    "dormand_prince": "rk45",
    "rkf45": "rk45",
    "tsitouras5": "tsit5",
    "tsit": "tsit5",
    "dopri8": "dop853",
    "dop8": "dop853",
    "dop": "dop853",
    # implicit / stiff
    "ros": "rosenbrock",
    "rosw": "rosenbrock",
    "rosenbrock_w": "rosenbrock",
    "tr_bdf2": "trbdf2",
    "trbdf": "trbdf2",
    "sdirk": "trbdf2",
    "gear": "bdf",
    # SDE
    "em": "euler_maruyama",
    "eulermaruyama": "euler_maruyama",
    "maruyama": "euler_maruyama",
}

#: SciPy / v2 stiff method names that have **no** engine kernel.  Resolving one
#: fails with a hint pointing at the engine's stiff family rather than a bare
#: "unknown method", because a user reaching for these clearly wants a stiff
#: solver.  (``"bdf"`` is **not** here — it now names the engine's variable-order
#: BDF kernel, stream E-BDF — so it resolves like any built-in.)
_STIFF_SCIPY_NAMES: frozenset[str] = frozenset({"lsoda", "radau", "vode"})

#: The default kernel per family (the zero-config choice).  DDE reuses the ODE
#: stage integrators (method-of-steps), so it shares the ODE default.
DEFAULT_METHOD: dict[str, str] = {"ode": "rk45", "dde": "rk45", "sde": "euler_maruyama"}

#: The default *stiff* kernel per family — what :func:`select` returns when the
#: RHS is stiff.  Only families with an implicit kernel appear.  The ODE default
#: is the variable-order ``bdf`` (stream E-BDF): it takes far larger steps through
#: a smooth stiff phase than the fixed-order ``rosenbrock``/``trbdf2`` (which stay
#: selectable by name), closing the warm-throughput gap to a variable-order BDF
#: reference (``benches/REPORT.md``).
STIFF_METHOD: dict[str, str] = {"ode": "bdf", "dde": "rosenbrock"}


def normalize(method: str) -> str:
    """Return *method* lower-cased with whitespace/``-`` runs collapsed to ``_``.

    The single normal form both :data:`_ALIASES` keys and registry lookups use,
    so ``"RK45"``, ``"rk-45"`` would-be variants and ``"Tr BDF2"`` all map to a
    comparable token before alias resolution.

    Parameters
    ----------
    method : str
        A raw ``method=`` string.

    Returns
    -------
    str
        The normalised token.
    """
    return re.sub(r"[\s\-]+", "_", str(method).strip().lower())


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolution:
    """The outcome of resolving a ``method=`` string to a concrete kernel.

    Attributes
    ----------
    name : str
        The canonical kernel name (the resolved ``method=`` the engine receives).
    spec : SolverSpec
        The registered spec for :attr:`name`.
    family : str or None
        The problem family resolution was validated against, or ``None`` if the
        family was not constrained.
    """

    name: str
    spec: SolverSpec
    family: str | None = None

    @property
    def needs_jacobian(self) -> bool:
        """Whether the kernel requires the analytic Jacobian (drives ``with_jacobian``)."""
        return self.spec.caps.needs_jacobian

    @property
    def is_implicit(self) -> bool:
        """Whether the kernel is implicit."""
        return self.spec.caps.kind == "implicit"

    @property
    def adaptive(self) -> bool:
        """Whether the kernel controls its own step size."""
        return self.spec.caps.adaptive

    @property
    def kernel(self) -> str:
        """The Rust kernel name the engine dispatches to."""
        return self.spec.kernel

    @property
    def build_kwargs(self) -> dict[str, bool]:
        """``{"with_jacobian": True}`` iff the kernel needs the Jacobian, else ``{}``.

        The exact kwargs to merge into
        :func:`tsdynamics.engine.problem.build_problem` so an implicit method
        gets a Jacobian-carrying tape and the engine guard is satisfied.
        """
        return {"with_jacobian": True} if self.needs_jacobian else {}


def available_for(family: str | None = None) -> list[str]:
    """Return the registered solver names, optionally filtered to *family*.

    Parameters
    ----------
    family : str, optional
        If given, keep only solvers whose caps support that problem family
        (``"ode"``, ``"sde"``, …).  ``"dde"`` reuses the explicit ODE stage
        integrators (method-of-steps), so it returns the explicit ODE kernels.

    Returns
    -------
    list[str]
        Sorted solver names.
    """
    specs = all_specs()
    if family is None:
        return sorted(specs)
    return sorted(name for name, spec in specs.items() if _spec_supports(spec, family))


def _spec_supports(spec: SolverSpec, family: str) -> bool:
    """Whether *spec* can integrate *family* (with the DDE-reuse rule)."""
    if spec.caps.supports_family(family):
        return True
    # The DDE method-of-steps drives an explicit ODE stage integrator (ROADMAP
    # E-DDE), so any explicit ODE kernel is usable for a DDE even though the Rust
    # caps tag it ``ode`` only.
    return family == "dde" and spec.caps.kind == "explicit" and spec.caps.supports_family("ode")


def resolve(method: str, *, family: str | None = None) -> Resolution:
    """Resolve a ``method=`` string to a concrete, validated kernel.

    Normalises the name (:func:`normalize`), applies the alias table, looks the
    canonical name up in the registry, and — if *family* is given — checks the
    kernel supports it.  Built-in and plugin-registered solvers resolve
    identically.

    Parameters
    ----------
    method : str
        The requested method (e.g. ``"RK45"``, ``"dopri5"``, ``"rosenbrock"``).
    family : str, optional
        The problem family the kernel must support (``"ode"``, ``"dde"``,
        ``"sde"``).  When omitted, only existence is checked.

    Returns
    -------
    Resolution
        The canonical name, its spec, and the ``with_jacobian`` decision.

    Raises
    ------
    ValueError
        If *method* names no registered kernel (the message lists the available
        methods, with a stiff-family hint for SciPy/v2 stiff names), or if the
        kernel does not support *family*.
    """
    norm = normalize(method)
    canonical = _ALIASES.get(norm, norm)

    specs = all_specs()
    if canonical not in specs:
        raise ValueError(_unknown_method_message(method, norm, family))

    spec = get(canonical)
    if family is not None and not _spec_supports(spec, family):
        raise ValueError(
            f"solver {canonical!r} does not support the {family!r} family "
            f"(it supports {sorted(spec.caps.supports)}); "
            f"available for {family!r}: {available_for(family)}"
        )
    return Resolution(name=canonical, spec=spec, family=family)


def _unknown_method_message(method: str, norm: str, family: str | None) -> str:
    """Build the 'unknown method' error, with a stiff hint where it helps."""
    scope = f" for family {family!r}" if family is not None else ""
    msg = f"unknown solver method {method!r}; available{scope}: {available_for(family)}"
    if norm in _STIFF_SCIPY_NAMES:
        msg += (
            f". ({method!r} is a SciPy/v2 stiff method with no engine kernel; "
            f"use {STIFF_METHOD.get('ode', 'rosenbrock')!r} or 'trbdf2' for stiff problems.)"
        )
    return msg


# ---------------------------------------------------------------------------
# Convenience predicates / kwargs (the C-FAM seam hooks)
# ---------------------------------------------------------------------------


def needs_jacobian(method: str, *, family: str | None = None) -> bool:
    """Whether *method* resolves to a kernel that requires the analytic Jacobian."""
    return resolve(method, family=family).needs_jacobian


def is_implicit(method: str, *, family: str | None = None) -> bool:
    """Whether *method* resolves to an implicit kernel."""
    return resolve(method, family=family).is_implicit


def build_kwargs(method: str, *, family: str | None = None) -> dict[str, bool]:
    """Return the ``build_problem`` kwargs implied by *method*.

    ``{"with_jacobian": True}`` when the resolved kernel needs the Jacobian
    (every implicit method, plus Milstein), else ``{}``.  The family
    engine-dispatch seam merges this into its
    :func:`~tsdynamics.engine.problem.build_problem` call so the stiff path
    "just works" — an implicit ``method=`` produces a Jacobian-carrying tape and
    the engine's Jacobian guard (PR #74) is satisfied rather than raising.

    Parameters
    ----------
    method : str
        The requested ``method=`` (resolved through :func:`resolve`).
    family : str, optional
        The problem family, forwarded to :func:`resolve` for validation.

    Returns
    -------
    dict[str, bool]
    """
    return resolve(method, family=family).build_kwargs


# ---------------------------------------------------------------------------
# Selection policy
# ---------------------------------------------------------------------------


def default_method(family: str = "ode") -> str:
    """Return the zero-config default kernel name for *family*.

    Parameters
    ----------
    family : str, default "ode"
        ``"ode"``, ``"dde"``, or ``"sde"``.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If *family* has no default kernel (e.g. ``"map"`` — maps iterate on the
        engine's own loop and take no ``method=``).
    """
    try:
        return DEFAULT_METHOD[family]
    except KeyError:
        raise ValueError(
            f"no default solver for family {family!r}; "
            f"families with a method= are {sorted(DEFAULT_METHOD)} "
            f"(maps iterate without a solver kernel)"
        ) from None


def select(family: str = "ode", *, stiff: bool = False) -> str:
    """Name the kernel to use for *family*, given a stiffness verdict (policy).

    This is the *policy* half of auto-stiffness: it consumes a boolean verdict
    (from :func:`is_stiff`, the engine's runtime detector, or the caller) and
    returns the appropriate kernel — the family's stiff kernel when ``stiff`` and
    one exists, else the family default.

    Parameters
    ----------
    family : str, default "ode"
        ``"ode"``, ``"dde"``, or ``"sde"``.
    stiff : bool, default False
        Whether the RHS is stiff.  Ignored for families with no implicit kernel
        (e.g. ``"sde"``), which always return their explicit default.

    Returns
    -------
    str
        The selected canonical kernel name.
    """
    if stiff and family in STIFF_METHOD:
        return STIFF_METHOD[family]
    return default_method(family)


# ---------------------------------------------------------------------------
# Auto-stiffness detection (a-priori, from the Jacobian spectrum)
# ---------------------------------------------------------------------------

#: Default stiffness-ratio threshold for :func:`is_stiff`.  The ratio of the
#: fastest to the slowest decaying mode; a few orders of magnitude is the
#: classic boundary above which an explicit method is step-bounded by stability
#: rather than accuracy (Lambert, *Numerical Methods for Ordinary Differential
#: Systems*, 1991).
_DEFAULT_RATIO_THRESHOLD: float = 1.0e3

#: Below this magnitude a mode's real part is treated as ~0 (a non-decaying /
#: marginal mode) and excluded from the ratio's denominator.
_NEGLIGIBLE_RE: float = 1.0e-8


def is_stiff(
    system: SystemBase,
    *,
    ic: Any = None,
    t: float = 0.0,
    ratio_threshold: float = _DEFAULT_RATIO_THRESHOLD,
) -> bool:
    """Cheap a-priori stiffness test from the Jacobian spectrum at one point.

    Evaluates the analytic Jacobian ``∂f/∂u`` at ``(ic, t)`` (via the pure-Python
    reference evaluator — no compiled engine needed), takes its eigenvalues, and
    reports stiff when the **stiffness ratio** — the largest decay rate over the
    smallest non-negligible decay rate — exceeds *ratio_threshold* while at least
    one genuinely fast decaying mode is present.

    This is a one-point heuristic meant to seed solver choice, not a guarantee:
    the engine's *runtime* detector (rejected-step ratio over a window) is the
    authoritative signal once integration is underway.  Returns ``False``
    conservatively when the Jacobian cannot be formed or has no decaying mode.

    Parameters
    ----------
    system : SystemBase
        An ODE system (anything :func:`tsdynamics.engine.run.eval_jac` can lower
        with a Jacobian).  Non-ODE families return ``False``.
    ic : array-like, optional
        Point to evaluate the Jacobian at.  Defaults to the system's resolved IC.
    t : float, default 0.0
        Time to evaluate at (for non-autonomous systems).
    ratio_threshold : float, default 1e3
        Stiffness-ratio boundary above which the RHS is called stiff.

    Returns
    -------
    bool
    """
    eigs = _jacobian_eigenvalues(system, ic=ic, t=t)
    if eigs is None:
        return False

    # Decay rates: negative real parts (stable/decaying modes drive stiffness).
    decay = -eigs.real
    fast = float(decay.max()) if decay.size else 0.0
    if fast <= _NEGLIGIBLE_RE:
        # No decaying mode at all → an explicit method is never stability-bound.
        return False

    slow_modes = decay[decay > _NEGLIGIBLE_RE]
    slowest = float(slow_modes.min())
    ratio = fast / slowest
    return ratio >= ratio_threshold


def _jacobian_eigenvalues(system: Any, *, ic: Any, t: float) -> np.ndarray | None:
    """Return the eigenvalues of ``∂f/∂u`` at ``(ic, t)``, or ``None`` on failure.

    Lazily imports the engine run seam so importing :mod:`tsdynamics.solvers`
    never pulls in the engine.  Any lowering / evaluation failure (non-ODE
    family, no Jacobian, non-finite values) is swallowed into ``None`` so callers
    fall back to the explicit default rather than erroring on solver *selection*.
    """
    try:
        from ..engine.run import eval_jac

        u = system.resolve_ic() if ic is None else np.asarray(ic, dtype=float)
        _deriv, jac = eval_jac(system, u, float(t), backend="reference")
        jac = np.asarray(jac, dtype=float)
        if jac.ndim != 2 or jac.shape[0] != jac.shape[1] or jac.size == 0:
            return None
        if not np.all(np.isfinite(jac)):
            return None
        return np.linalg.eigvals(jac)
    except Exception:  # noqa: BLE001 — selection must never raise on a bad probe
        return None


def recommend(
    system: SystemBase,
    *,
    family: str = "ode",
    ic: Any = None,
    t: float = 0.0,
    ratio_threshold: float = _DEFAULT_RATIO_THRESHOLD,
) -> Resolution:
    """Recommend a solver for *system*, auto-selecting implicit on a stiff RHS.

    Combines the detector (:func:`is_stiff`) with the policy (:func:`select`):
    probes the Jacobian spectrum and resolves to the family's stiff kernel when
    stiff, the explicit default otherwise.

    Parameters
    ----------
    system : SystemBase
        The system to integrate.
    family : str, default "ode"
        The problem family (only ``"ode"``/``"dde"`` have a stiff alternative).
    ic, t, ratio_threshold
        Forwarded to :func:`is_stiff`.

    Returns
    -------
    Resolution
        The recommended kernel, ready to feed to the engine (and its
        :attr:`~Resolution.build_kwargs`).
    """
    stiff = family in STIFF_METHOD and is_stiff(system, ic=ic, t=t, ratio_threshold=ratio_threshold)
    return resolve(select(family, stiff=stiff), family=family)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
