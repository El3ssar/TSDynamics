r"""
Expansion entropy (Hunt & Ott 2015) — a definition of chaos via volume growth.

For a region :math:`S` of state space, expansion entropy measures the
exponential growth rate of the *expansion* the linearised dynamics produce on
trajectories that remain in :math:`S`.  Writing :math:`G(A)` for the product of
the singular values of a matrix :math:`A` that exceed 1 (the volume growth of
the unit ball under :math:`A`, restricted to expanding directions), and
:math:`DF^{t}` for the fundamental/tangent matrix accumulated over :math:`t`,

.. math::

    E(t) = \frac{1}{N}\!\!\sum_{\substack{i:\ \text{orbit stays in } S}}\!\! G\big(DF^{t}_i\big),
    \qquad H = \lim_{t\to\infty} \frac{\ln E(t)}{t},

estimated by sampling :math:`N` initial conditions uniformly in :math:`S` and
reading :math:`H` as the slope of :math:`\ln E(t)` against :math:`t`.  Positive
:math:`H` is chaos.  For a uniformly expanding map :math:`H` is exact — the tent
map with unit height has :math:`|f'| \equiv 2`, so :math:`E(t) = 2^t` and
:math:`H = \ln 2`.

Supported systems are discrete maps (exact tangent map) and flows (RK4
variational fundamental matrix).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from tsdynamics.data import Box, sampler
from tsdynamics.errors import InvalidInputError, InvalidParameterError, remedy
from tsdynamics.families import ContinuousSystem, DiscreteMap

from .._common import reject_data
from .._result import AnalysisResult, ScalingResult
from .._tangent import flow_fns, map_fns, rk4_variational
from . import _common as _c

__all__ = ["ExpansionEntropyResult", "expansion_entropy"]


@dataclass(frozen=True, eq=False)
class ExpansionEntropyResult(ScalingResult):
    r"""An expansion-entropy estimate with the growth curve it was read from.

    A :class:`~tsdynamics.analysis._result.ScalingResult` — the entropy is the
    slope of :math:`\ln E(t)` against :math:`t` — so it inherits the canonical
    ``estimate`` / ``abscissa`` / ``ordinate`` / ``fit_region`` schema, the result
    surface (``.meta`` / the readout ``repr`` / ``.to_dict()`` / the ``.plot`` seam) and
    ``float(result)`` (the entropy :math:`H`).  Domain-named ``@property`` aliases
    (:attr:`entropy`, :attr:`times`, :attr:`log_growth`, :attr:`fit_slice`)
    preserve the original field names.

    Attributes
    ----------
    estimate : float
        The estimated expansion entropy :math:`H`.  Aliased :attr:`entropy`.
    abscissa : ndarray
        The :math:`t` grid (iterations for maps, time for flows).  Aliased
        :attr:`times`.
    ordinate : ndarray
        :math:`\ln E(t)` at each :math:`t`.  Aliased :attr:`log_growth`.
    fit_region : tuple[int, int]
        Inclusive ``(lo, hi)`` indices of the fitted range.  Aliased
        :attr:`fit_slice`.
    n_samples : int
        Number of initial conditions sampled in the region.
    n_survivors : int
        How many of them stayed in the region for the whole run.
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ("entropy", "stderr", "n_survivors", "n_samples")

    n_samples: int = 0
    n_survivors: int = 0

    @property
    def entropy(self) -> float:
        """The estimated expansion entropy (alias of :attr:`estimate`)."""
        return float(self.estimate)

    @property
    def times(self) -> np.ndarray:
        """The :math:`t` grid (alias of :attr:`abscissa`)."""
        return self.abscissa

    @property
    def log_growth(self) -> np.ndarray:
        r"""The :math:`\ln E(t)` curve (alias of :attr:`ordinate`)."""
        return self.ordinate

    @property
    def fit_slice(self) -> tuple[int, int]:
        """The fitted index range (alias of :attr:`fit_region`)."""
        return self.fit_region

    def _quantity(self) -> str:
        r"""Return ``H0`` — the symbol expansion entropy is known by."""
        return "H0"

    @property
    def applicable(self) -> bool:
        r"""Whether enough trajectories survived the region for the fit to mean anything.

        ``False`` on a degenerate fit — fewer than two survivors, or a
        ``stderr`` of exactly zero, which is what an empty survivor set produces.
        Before v6 the "indistinguishable from zero" guard was written
        ``se > 0.0 and abs(h) < 2*se``, so ``0 ± 0`` fell straight through to the
        **most** confident branch and a run where nothing survived printed
        ``H0 = 0 ± 0   non-chaotic (H0 ≤ 0)``.

        Returns
        -------
        bool
        """
        se = float(self.stderr)
        return bool(self.n_survivors >= 2 and np.isfinite(se) and se > 0.0)

    @property
    def chaotic(self) -> bool | None:
        r"""Whether :math:`H_0 > 0` — Hunt & Ott's definition of chaos.

        ``None`` — never ``False`` — when the fit is not :attr:`applicable` or
        the slope is within two standard errors of zero.

        Returns
        -------
        bool or None
        """
        h, se = float(self.estimate), float(self.stderr)
        if not self.applicable or not np.isfinite(h):
            return None
        if abs(h) < 2.0 * se:
            return None
        return bool(h > 0.0)

    def _interpretation(self) -> str | None:
        r"""Name the dynamics: a **positive** expansion entropy is chaos.

        Hunt & Ott (2015) define chaos as :math:`H_0 > 0`, so this is the one
        place the sign genuinely is the verdict.  It is read against the fit's
        own standard error rather than against zero, so a slope indistinguishable
        from flat is not sold as a positive one — and a fit with **no** surviving
        trajectories (``0 ± 0``) says so rather than taking the confident branch.
        """
        h = float(self.estimate)
        if not np.isfinite(h):
            return None
        if not self.applicable:
            return (
                f"not applicable — {self.n_survivors} of {self.n_samples} samples "
                "stayed in the region"
            )
        if not self.trusted:
            return self._fit_quality_clause()
        verdict = self.chaotic
        if verdict is None:
            return "indistinguishable from zero (H0 within 2 s.e. of 0)"
        return "chaotic (H0 > 0)" if verdict else "non-chaotic (H0 ≤ 0)"

    def _derived(self) -> dict[str, Any]:
        r"""Export the entropy, the applicability flag and the verdict."""
        data = super()._derived()
        data.update(entropy=self.entropy, applicable=self.applicable, chaotic=self.chaotic)
        return data

    def _context(self) -> str | None:
        """Return how many sampled initial conditions survived the whole run."""
        bits = [b for b in (self._system_label(),) if b]
        bits.append(f"{self.n_survivors}/{self.n_samples} survivors")
        return ", ".join(bits)


def expansion_entropy(
    system: Any,
    region: Any = None,
    *,
    n_samples: int = 1000,
    n: int | None = None,
    final_time: float | None = None,
    dt: float | None = None,
    fit_range: tuple[int, int] | None = None,
    seed: int | None = 0,
    n_internal: int = 5,
) -> ExpansionEntropyResult:
    r"""Estimate the expansion entropy :math:`H` of a map or flow on a region.

    Parameters
    ----------
    system : DiscreteMap or ContinuousSystem
        The system whose expansion to measure.
    region : Box, (lo, hi), or None
        The restricting region :math:`S`.  ``None`` uses the (10%-expanded)
        bounding box of a burn-in orbit.
    n_samples : int, default 1000
        Number of **initial conditions** sampled uniformly in the region — the
        same quantity ``attractors`` / ``fixed_points`` spell ``n_seeds`` and
        ``basin_fractions`` / ``continuation`` spell ``n``.
    n : int, optional
        Number of iterations (maps).  Default 15.  (Kept modest: the raw tangent
        product is not renormalised, so very long horizons overflow.)
    final_time : float, optional
        Integration time (flows).  Default 5.0.
    dt : float, optional
        Recording step for flows.  Default 0.1.  Not valid for maps.
    fit_range : (int, int), optional
        Inclusive index range in the :math:`t` grid to fit the slope over.
        Default skips :math:`t = 0` and uses the rest.
    seed : int, optional
        Seed for the initial-condition sampling.
    n_internal : int, default 5
        RK4 sub-steps per ``dt`` for flows.

    Returns
    -------
    ExpansionEntropyResult

    Raises
    ------
    InvalidInputError
        If ``system`` is measured data rather than a model, or is neither a
        discrete map nor a continuous flow.  A ``TypeError`` subclass, like every
        other wrong-subject refusal in the analysis layer — it used to be a
        ``NotImplementedError``, so ``except TypeError`` missed it here alone.
    InvalidParameterError
        If ``n_samples < 1``; if the step count is degenerate (``n < 1`` for a
        map, ``final_time <= 0`` or ``dt <= 0`` for a flow); if ``dt`` is passed
        for a map (or ``n`` for a flow); if the region dimension does not match
        the system; or if fewer than two finite :math:`\ln E(t)` points remain to
        fit (too few survivors stayed in the region — enlarge it or shorten the
        horizon).

    Examples
    --------
    >>> float(expansion_entropy(Tent(params={"mu": 1.0}), Box([0.0], [1.0])))
    0.69...                                              # ln 2, exact

    References
    ----------
    Hunt & Ott, "Defining chaos", *Chaos* **25** (2015) 097618.
    """
    # Measured data first, and through the shared guard (see ``gali``): expansion
    # entropy re-integrates the equations' tangent dynamics from a lattice of
    # starts, so a point set cannot stand in for the model.
    reject_data(system, analysis="expansion_entropy")
    if isinstance(system, DiscreteMap):
        mode = "map"
    elif isinstance(system, ContinuousSystem):
        mode = "flow"
    else:
        raise InvalidInputError(
            f"expansion_entropy evolves the tangent dynamics of a map or a continuous "
            f"flow, and {type(system).__name__} is neither (a delay system's tangent "
            f"space is the infinite-dimensional history, and a derived wrapper has no "
            f"equations of its own)."
            + remedy(
                "ts.analysis.expansion_entropy(system, region)",
                lead="Measure it on the underlying system:",
            )
        )

    n_samples = int(n_samples)
    if n_samples < 1:
        raise InvalidParameterError(f"n_samples must be >= 1; got {n_samples}.")

    box = _c._resolve_region(system, region)
    if box.dim != system.dim:
        raise InvalidParameterError(
            f"region dimension ({box.dim}) does not match system dimension ({system.dim})."
        )
    draw = sampler(box, seed=seed)
    ics = np.array([draw() for _ in range(n_samples)])

    if mode == "map":
        if dt is not None:
            raise InvalidParameterError("dt has no meaning for a discrete map — omit it.")
        n_steps = 15 if n is None else int(n)
        if n_steps < 1:
            raise InvalidParameterError(f"n (number of iterations) must be >= 1; got {n_steps}.")
        times, log_growth, survivors = _expansion_map(system, ics, box, n_steps)
    else:
        if n is not None:
            raise InvalidParameterError("n applies to maps; use final_time/dt for a flow.")
        t_end = 5.0 if final_time is None else float(final_time)
        step_dt = 0.1 if dt is None else float(dt)
        if step_dt <= 0.0:
            raise InvalidParameterError(f"dt must be positive; got {step_dt}.")
        if t_end <= 0.0 or int(round(t_end / step_dt)) < 1:
            raise InvalidParameterError(
                f"final_time must be positive and span at least one dt step; "
                f"got final_time={t_end}, dt={step_dt}."
            )
        times, log_growth, survivors = _expansion_flow(
            system, ics, box, t_end, step_dt, int(n_internal)
        )

    lo, hi = _resolve_fit_range(times, log_growth, fit_range)
    # Fit only over finite ln E(t): once every sample has left S, E = 0 and ln E
    # = -inf (a contiguous tail in practice, but mask defensively rather than
    # rely on that invariant).
    t_fit, y_fit = times[lo : hi + 1], log_growth[lo : hi + 1]
    finite = np.isfinite(y_fit)
    if int(np.count_nonzero(finite)) < 2:
        raise InvalidParameterError(
            "expansion entropy: fewer than two finite ln E(t) points in the fit range "
            "(too few survivors stayed in the region — enlarge it or shorten the horizon)."
        )
    slope, intercept, stderr = _c._linfit(t_fit[finite], y_fit[finite])
    return ExpansionEntropyResult(
        estimate=slope,
        stderr=stderr,
        abscissa=times,
        ordinate=log_growth,
        fit_region=(lo, hi),
        intercept=intercept,
        n_samples=int(n_samples),
        n_survivors=int(survivors),
        meta=AnalysisResult.build_meta(
            system, analysis="expansion_entropy", n_samples=int(n_samples)
        ),
    )


def _resolve_fit_range(
    times: np.ndarray, log_growth: np.ndarray, fit_range: tuple[int, int] | None
) -> tuple[int, int]:
    """Choose the fit window: explicit, else skip ``t=0`` and any non-finite tail."""
    n = times.size
    if fit_range is not None:
        lo, hi = int(fit_range[0]), int(fit_range[1])
        if not (0 <= lo < hi < n):
            raise InvalidParameterError(f"fit_range {fit_range} out of bounds for {n} points.")
        return lo, hi
    finite = np.nonzero(np.isfinite(log_growth))[0]
    if finite.size < 2:
        raise InvalidParameterError(
            "expansion entropy: fewer than two finite ln E(t) points to fit."
        )
    lo = int(finite[0])
    if lo == 0 and finite.size > 2:
        lo = int(finite[1])  # drop t=0 (ln E = 0 trivially)
    hi = int(finite[-1])
    return lo, hi


def _expansion_volumes(mats: np.ndarray) -> np.ndarray:
    r"""Batched Hunt--Ott ``G(M)`` over a stack of matrices ``mats`` (shape ``(n, d, d)``).

    Returns one volume per matrix, identical to calling
    :func:`~tsdynamics.analysis.chaos._common.expansion_volume` on each: a
    non-finite matrix reports ``+inf``; otherwise the product of its singular
    values exceeding 1 (``1.0`` when none exceed 1).  One ``np.linalg.svd`` over
    the whole stack replaces the per-matrix Python calls; LAPACK computes each
    matrix's singular values independently, so the values match per matrix.

    If the batched SVD fails to converge on some (finite) matrix it raises for
    the whole stack, so we fall back to the per-matrix scalar path, which reports
    ``+inf`` for a non-convergent matrix — preserving the robustness of the
    original per-call :func:`~tsdynamics.analysis.chaos._common.expansion_volume`.
    """
    n = mats.shape[0]
    out = np.empty(n)
    finite = np.all(np.isfinite(mats), axis=(1, 2))
    out[~finite] = np.inf
    if np.any(finite):
        try:
            # Singular values of every finite matrix at once (descending per row).
            sv = np.linalg.svd(mats[finite], compute_uv=False)  # (n_finite, d)
            # Product of singular values > 1 per matrix: mask sub-unit values to 1
            # so they drop out of the product (matches ``s[s > 1.0]`` then
            # ``prod``; the all-<=1 row then yields the empty-product 1.0).
            out[finite] = np.prod(np.where(sv > 1.0, sv, 1.0), axis=1)
        except np.linalg.LinAlgError:
            # A non-convergent (but finite) matrix poisons the whole batch; redo
            # the finite ones one at a time so the offender reports +inf and the
            # rest still get their exact volume.
            out[finite] = [_c.expansion_volume(m) for m in mats[finite]]
    return out


def _expansion_map(
    system: Any, ics: np.ndarray, box: Box, n_steps: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Accumulate ``G(DF^t)`` for a map; ``DF^t = J(x_{t-1})...J(x_0)`` per sample."""
    step, jac = map_fns(system)
    dim = int(system.dim)
    n = ics.shape[0]
    eye = np.eye(dim)
    mats = [eye.copy() for _ in range(n)]
    states = [np.asarray(ic, dtype=float).ravel() for ic in ics]
    alive = np.ones(n, dtype=bool)
    lo, hi = box.lo, box.hi

    times = np.arange(0, n_steps + 1, dtype=float)
    ln_e = np.empty(n_steps + 1)
    ln_e[0] = 0.0  # E(0) = mean of G(I) = 1
    for t in range(1, n_steps + 1):
        # Advance each still-alive sample's fundamental matrix and state. The map
        # ``_step``/``_jacobian`` are not batchable over samples (a user kernel may
        # not broadcast), so the advance stays a per-sample loop; the kill test and
        # the volume SVD that follow are batched.
        live_idx = np.flatnonzero(alive)
        for i in live_idx:
            x = states[i]
            mats[i] = jac(x) @ mats[i]
            x = step(x)
            states[i] = x
        # Box-membership of every freshly-stepped state in one pass (same
        # ``lo <= x <= hi`` predicate as ``box.contains``).
        stepped = np.array([states[i] for i in live_idx]) if live_idx.size else np.empty((0, dim))
        inside = (
            np.all((stepped >= lo) & (stepped <= hi), axis=1)
            if live_idx.size
            else np.empty(0, dtype=bool)
        )
        survivors = live_idx[inside]
        alive[live_idx[~inside]] = False
        # G(DF^t) summed over the survivors via one batched SVD; division by the
        # full sample count ``n`` (escaped samples contribute 0) is unchanged.
        if survivors.size:
            vol_mats = np.stack([mats[i] for i in survivors])
            total = float(np.sum(_expansion_volumes(vol_mats)))
        else:
            total = 0.0
        e = total / n
        ln_e[t] = np.log(e) if e > 0.0 else -np.inf
    return times, ln_e, int(np.count_nonzero(alive))


def _expansion_flow(
    system: Any, ics: np.ndarray, box: Box, final_time: float, dt: float, n_internal: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Accumulate ``G(Phi(t))`` for a flow; ``Phi`` is the RK4 fundamental matrix per sample."""
    rhs, jac = flow_fns(system)
    dim = int(system.dim)
    n = ics.shape[0]
    n_steps = int(round(final_time / dt))
    h = dt / max(1, n_internal)

    states = [np.asarray(ic, dtype=float).ravel() for ic in ics]
    mats = [np.eye(dim) for _ in range(n)]
    alive = np.ones(n, dtype=bool)
    t_local = np.zeros(n)
    lo, hi = box.lo, box.hi

    times = np.empty(n_steps + 1)
    times[0] = 0.0
    ln_e = np.empty(n_steps + 1)
    ln_e[0] = 0.0
    for s in range(1, n_steps + 1):
        # The RK4 variational sub-stepping calls the SymEngine-lambdified
        # ``rhs``/``jac``, which are not batchable over samples, so the advance
        # stays a per-sample loop (the variational core is untouched). The kill
        # test and the volume SVD are batched across the survivors.
        live_idx = np.flatnonzero(alive)
        for i in live_idx:
            x, m, t = states[i], mats[i], t_local[i]
            for _ in range(n_internal):
                x, m = rk4_variational(rhs, jac, x, m, t, h)
                t += h
            states[i], mats[i], t_local[i] = x, m, t
        stepped = np.array([states[i] for i in live_idx]) if live_idx.size else np.empty((0, dim))
        inside = (
            np.all((stepped >= lo) & (stepped <= hi), axis=1)
            if live_idx.size
            else np.empty(0, dtype=bool)
        )
        survivors = live_idx[inside]
        alive[live_idx[~inside]] = False
        if survivors.size:
            vol_mats = np.stack([mats[i] for i in survivors])
            total = float(np.sum(_expansion_volumes(vol_mats)))
        else:
            total = 0.0
        e = total / n
        times[s] = s * dt
        ln_e[s] = np.log(e) if e > 0.0 else -np.inf
    return times, ln_e, int(np.count_nonzero(alive))


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
