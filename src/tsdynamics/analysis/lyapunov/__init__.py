"""Lyapunov-based quantifiers: spectra, maximal exponent, Kaplan–Yorke dimension."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np

from tsdynamics.errors import ConvergenceError, InvalidParameterError
from tsdynamics.families import DelaySystem, DiscreteMap

from ... import registry as _registry
from .._result import AnalysisResult, ArrayResult, ScalarResult
from .from_data import LyapunovFromData, lyapunov_from_data

__all__ = [
    "LyapunovFromData",
    "LyapunovSpectrum",
    "kaplan_yorke_dimension",
    "lyapunov_from_data",
    "lyapunov_spectrum",
    "max_lyapunov",
]


@dataclass(frozen=True, eq=False)
class LyapunovSpectrum(ArrayResult):
    """The Lyapunov spectrum of a system — the exponents and the result surface.

    An :class:`~tsdynamics.analysis._result.ArrayResult`, so it is a drop-in for
    the bare exponent array: ``np.asarray(result)``, indexing, iteration and
    ``result.shape`` all defer to the wrapped exponents, while it also carries
    ``.meta`` / ``.summary()`` / ``.to_dict()`` / the ``.plot`` seam.

    Attributes
    ----------
    exponents : numpy.ndarray
        The Lyapunov exponents, largest first.  Alias of the wrapped
        :attr:`~tsdynamics.analysis._result.ArrayResult.values`;
        ``np.asarray(result)`` returns them.
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ("exponents",)

    @property
    def exponents(self) -> np.ndarray:
        """The Lyapunov exponents (alias of the wrapped array)."""
        return np.asarray(self.values)

    @property
    def kaplan_yorke(self) -> float:
        """The Kaplan--Yorke (Lyapunov) dimension implied by this spectrum."""
        return float(kaplan_yorke_dimension(self.values))

    def _interpretation(self) -> str | None:
        """Name the dynamics from the count of *positive* exponents.

        A flow's zero exponent is only numerically near zero, so "positive" is
        thresholded relative to the spectrum's scale (``1e-3`` of the largest
        magnitude) rather than at an absolute floor.
        """
        exps = np.asarray(self.values, dtype=float)
        if exps.size == 0:
            return None
        tol = max(1e-6, 1e-3 * float(np.max(np.abs(exps))))
        n_pos = int((exps > tol).sum())
        if n_pos >= 2:
            return f"hyperchaotic: {n_pos} positive exponents"
        if n_pos == 1:
            return "chaotic: 1 positive exponent"
        return "regular: no positive exponent"

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe the Lyapunov spectrum as a backend-agnostic :class:`PlotSpec`.

        Builds a ``LYAPUNOV_SPECTRUM`` spec — one ``BAR`` per exponent
        :math:`\lambda_i` against its index :math:`i` (largest first) — with the
        :math:`\lambda = 0` line drawn as a horizontal reference.  The sign of each
        exponent (a bar above or below the zero line) is what separates the
        expanding from the contracting directions, so chaos reads off as a bar
        rising above the zero line.  The :mod:`tsdynamics.viz.spec` import is lazy,
        so building a spec never pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"lyapunov_spectrum"``).  ``None``
            uses ``LYAPUNOV_SPECTRUM``.

        Returns
        -------
        PlotSpec
        """
        from .. import _plotbuilder as pb

        exps = np.asarray(self.values, dtype=float)
        index = np.arange(exps.size, dtype=float)
        return pb.spec(
            kind,
            "lyapunov_spectrum",
            layers=[pb.bar(exps, x=index, label=r"$\lambda_i$")],
            xlabel="index $i$",
            ylabel=r"$\lambda_i$",
            title=f"Lyapunov spectrum ($D_{{KY}}$ = {self.kaplan_yorke:.3g})"
            if exps.size
            else "Lyapunov spectrum",
            annotations=[pb.hline(0.0, text=r"$\lambda = 0$")],
        )


def kaplan_yorke_dimension(spectrum: Any) -> ScalarResult:
    r"""Kaplan--Yorke (Lyapunov) dimension from a Lyapunov spectrum.

    .. math::

        D_{KY} = j + \frac{\lambda_1 + \cdots + \lambda_j}{|\lambda_{j+1}|},

    where :math:`j` is the largest index whose cumulative exponent sum is
    non-negative (Kaplan & Yorke 1979).  Interpolating between the :math:`j`-th
    and :math:`(j+1)`-th exponents gives a fractional estimate of the attractor's
    information dimension.

    Parameters
    ----------
    spectrum : array-like
        Lyapunov exponents (any order; sorted descending internally).

    Returns
    -------
    ScalarResult
        The dimension, a drop-in for its ``float`` value (``float(result)`` /
        comparisons work): ``0.0`` when every exponent is negative;
        ``len(spectrum)`` when the cumulative sum never turns negative (spectrum
        incomplete).

    Examples
    --------
    >>> float(kaplan_yorke_dimension([0.906, 0.0, -14.57]))   # Lorenz
    2.062...

    References
    ----------
    J. L. Kaplan & J. A. Yorke, "Chaotic behavior of multidimensional difference
    equations", in *Functional Differential Equations and Approximation of Fixed
    Points*, Lecture Notes in Mathematics **730**, Springer (1979) 204--227.
    """
    s = np.sort(np.asarray(spectrum, dtype=float))[::-1]
    if s.size == 0 or s[0] < 0.0:
        dky = 0.0
    else:
        cum = np.cumsum(s)
        j = int(np.nonzero(cum >= 0.0)[0][-1])
        # spectrum doesn't close (j is the last index) -> dimension saturates at len
        dky = float(s.size) if j == s.size - 1 else float(j + 1 + cum[j] / abs(s[j + 1]))
    return ScalarResult(value=dky, meta={"analysis": "kaplan_yorke_dimension", "k": int(s.size)})


def lyapunov_spectrum(
    system: Any,
    *,
    k: int | None = None,
    final_time: float | None = None,
    n: int | None = None,
    transient: float | None = None,
    dt: float | None = None,
    ic: Any | None = None,
    method: str | None = None,
) -> LyapunovSpectrum:
    """Lyapunov spectrum of any system — the uniform, documented entry point.

    Dispatches to the family implementation (QR tangent dynamics for maps, the
    extended variational system on the engine for ODEs, the engine
    function-space estimator for DDEs), translating this one signature to each
    family's native keywords.  The exponents are obtained by Benettin
    renormalisation of an evolving orthonormal frame (Benettin et al. 1980).

    Parameters
    ----------
    system : System
        A flow (ODE/DDE) or a discrete map.
    k : int, optional
        Number of exponents to compute (was ``n_exp``).  Defaults to
        ``system.dim`` for flows/maps; a DDE may request more than ``dim`` (its
        tangent space is the infinite-dimensional history).
    final_time : float, optional
        Averaging-window length for a **flow** (after the transient).  Mutually
        exclusive with ``n``; a flow uses ``final_time``.
    n : int, optional
        Number of iterations for a **map**.  Mutually exclusive with
        ``final_time``; a map uses ``n``.
    transient : float, optional
        Amount discarded before averaging (a flow burn-in **time**).  Maps
        reorthonormalise from the initial condition and take no transient here.
    dt : float, optional
        Sampling / integration step (flows only).
    ic : array-like, optional
        Initial condition.  Falls back to ``system.ic``, then random.
    method : str, optional
        Solver kernel (continuous flows only).

    Returns
    -------
    LyapunovSpectrum
        The exponents (largest first), a drop-in for the bare ``(k,)`` array —
        ``np.asarray(result)``, indexing and iteration work — that also carries
        ``.meta``, ``.summary()`` and the ``.kaplan_yorke`` dimension.

    Raises
    ------
    TypeError
        If ``system`` has no ``lyapunov_spectrum`` implementation (e.g. a derived
        wrapper — compute the spectrum on the underlying system).
    ValueError
        If ``k <= 0``, or a keyword is passed to the wrong family (``final_time``
        / ``dt`` / ``transient`` / ``method`` for a map; ``n`` for a flow; or a
        ``method`` for a DDE, which selects its engine via ``backend``).

    Examples
    --------
    >>> lyapunov_spectrum(Lorenz(), final_time=300.0)   # [0.91, ~0, -14.57]
    >>> lyapunov_spectrum(Henon(), k=2, n=5000)         # [0.42, -1.62]

    References
    ----------
    G. Benettin, L. Galgani, A. Giorgilli & J.-M. Strelcyn, "Lyapunov
    characteristic exponents for smooth dynamical systems and for Hamiltonian
    systems; a method for computing all of them", *Meccanica* **15** (1980)
    9--20 (Part 1) and 21--30 (Part 2).
    """
    method_fn = getattr(system, "lyapunov_spectrum", None)
    if method_fn is None:
        raise TypeError(
            f"{type(system).__name__} has no lyapunov_spectrum implementation. "
            f"For derived wrappers, compute the spectrum on the underlying system."
        )
    if k is not None and k <= 0:
        raise ValueError(f"k (number of exponents) must be a positive integer, got {k!r}.")

    fwd: dict[str, Any] = {}
    if k is not None:
        fwd["n_exp"] = k
    if ic is not None:
        fwd["ic"] = ic

    if getattr(system, "is_discrete", False):
        # Maps: horizon is `n` (iterations); no time, solver or burn-in concept.
        if final_time is not None:
            raise ValueError("lyapunov_spectrum: final_time is for flows; a map uses n.")
        if dt is not None:
            raise ValueError("lyapunov_spectrum: dt has no meaning for a discrete map.")
        if transient is not None:
            raise ValueError(
                "lyapunov_spectrum: transient is not supported for a map spectrum "
                "(the QR iteration reorthonormalises from the initial condition)."
            )
        if method is not None:
            raise ValueError("lyapunov_spectrum: a map spectrum has no solver method.")
        if n is not None:
            fwd["steps"] = n
    else:
        # Flows (ODE/DDE): horizon is `final_time`; transient is a burn-in time.
        if n is not None:
            raise ValueError("lyapunov_spectrum: n is for maps; a flow/DDE uses final_time.")
        if final_time is not None:
            fwd["final_time"] = final_time
        if transient is not None:
            fwd["burn_in"] = transient
        if dt is not None:
            fwd["dt"] = dt
        if method is not None:
            if isinstance(system, DelaySystem):
                raise ValueError(
                    "lyapunov_spectrum: a DDE selects its engine via backend, not method."
                )
            fwd["method"] = method
    exponents = np.asarray(method_fn(**fwd), dtype=float)
    meta = AnalysisResult.build_meta(
        system,
        analysis="lyapunov_spectrum",
        k=int(exponents.size),
        final_time=final_time,
        n=n,
        transient=transient,
    )
    return LyapunovSpectrum(values=exponents, meta=meta)


def _max_lyapunov_map(
    system: Any,
    *,
    n: int,
    steps_per: int,
    transient: int,
    ic: Any | None,
    seed: int | None,
) -> float | None:
    """Maximal exponent of a map as the top of the Rust QR tangent-map spectrum.

    Burns in ``transient`` map iterations to land on the attractor, then runs the
    engine kernel (:func:`tsdynamics.engine.run.map_lyapunov`) for ``n * steps_per``
    iterations and returns the leading exponent (``k=1``).  Returns ``None`` to
    decline — so :func:`max_lyapunov` falls back to the two-trajectory loop — when
    the map will not lower to the engine IR
    (:class:`~tsdynamics.engine.compile.TapeCompileError`) or the compiled wheel is
    absent (:class:`~tsdynamics.engine.run.EngineNotAvailableError`).

    ``seed`` makes the off-basin random-IC retry reproducible: each retry draws
    its initial condition from a seeded ``numpy`` generator (``U[0, 1)^dim``, the
    same distribution :meth:`~tsdynamics.families.base.SystemBase.resolve_ic`
    uses for a random fallback) rather than the unseeded global RNG, so a result
    that triggers the retry is deterministic when ``seed`` is given — matching the
    two-trajectory path's seed contract.
    """
    from tsdynamics.engine import run
    from tsdynamics.engine.compile import (
        TapeCompileError,
        lower_map_cached,
        tape_jacobian_is_smooth,
    )

    # Only a genuine DiscreteMap has an analytic _step/_jacobian to lower; a
    # WrappedSystem (adapted external stepper — is_discrete=True but no _step) and
    # anything else must take the two-trajectory loop.
    if not isinstance(system, DiscreteMap):
        return None

    name = type(system).__name__
    try:
        tape = lower_map_cached(system, with_jacobian=True)
        tape_arrays = tape.to_arrays()
    except TapeCompileError:
        return None  # a non-lowering _step (piecewise/ufunc) → two-trajectory loop
    except run.EngineNotAvailableError:  # pragma: no cover - wheel-free env
        return None
    # A piecewise map's lowered Jacobian collapses at kinks (the full-height Tent's
    # dyadic orbit hits x=0.5, poisoning the QR growth), so decline → two-trajectory.
    if not tape_jacobian_is_smooth(tape):
        return None

    steps = max(1, int(n) * int(steps_per))
    burn = max(0, int(transient))

    # Burn-in + the kernel run, with random-IC retry on divergence (a random draw
    # can land off-basin); an explicit ``ic`` that diverges re-raises. The retry
    # draws a *reproducible* random IC from a seeded generator when ``seed`` is
    # given, so a retried result is deterministic (the two-trajectory path's
    # contract); with ``seed=None`` the draw is unseeded, matching the prior
    # behaviour.
    ic_explicit = ic is not None
    rng = np.random.default_rng(seed)
    dim = cast("int", system.dim)
    max_retries = 10
    for attempt in range(max_retries):
        ref = system.copy()
        try:
            if attempt == 0:
                ref.reinit(ic)
            elif seed is None:
                ref.reinit(None)  # unseeded random fallback (resolve_ic)
            else:
                ref.reinit(rng.random(dim))  # reproducible random retry IC
            for _ in range(burn):
                ref.step()
            start = np.asarray(ref.state(), dtype=float)
            exponents, _intervals = run.map_lyapunov(
                tape_arrays,
                start,
                steps=steps,
                k=1,
                reortho_interval=1,
                name=name,
            )
        except run.EngineNotAvailableError:  # pragma: no cover - wheel-free env
            return None
        except (ConvergenceError, ArithmeticError):
            if ic_explicit or attempt == max_retries - 1:
                raise
            continue  # off-basin random draw diverged; retry from a fresh IC
        return float(exponents[0])

    # Unreachable: the loop returns on success or re-raises on the last attempt.
    raise ConvergenceError(  # pragma: no cover
        f"{name}.max_lyapunov: map iterates diverge from every tried IC."
    )


def max_lyapunov(
    system: Any,
    *,
    d0: float = 1e-9,
    n: int = 400,
    steps_per: int = 5,
    dt: float | None = None,
    transient: int = 500,
    ic: Any | None = None,
    seed: int | None = None,
) -> ScalarResult:
    r"""Maximal Lyapunov exponent by two-trajectory rescaling (Benettin et al. 1976).

    Runs a reference and a perturbed copy of the system in lockstep through
    the :class:`~tsdynamics.families.System` protocol — no Jacobian needed, so it
    works for any ODE or map (including ones with non-smooth right-hand
    sides).  The separation is rescaled back to ``d0`` after every cycle and the
    accumulated :math:`\ln(d / d_0)` is averaged over elapsed time.  Not
    available for DDEs (their state cannot be ``set_state``-ed); use
    ``DelaySystem.lyapunov_spectrum`` instead.

    Parameters
    ----------
    system : System
        ODE or map.
    d0 : float
        Perturbation size restored at every rescaling.
    n : int
        Number of rescaling cycles (more → better averaging).
    steps_per : int
        Protocol steps between rescalings.
    dt : float, optional
        Step size for continuous systems (default: the system's step default).
    transient : int
        Protocol steps discarded before measuring.
    ic : array-like, optional
        Initial condition for the reference trajectory.
    seed : int, optional
        Seed for the random perturbation direction (two-trajectory path) and for
        the off-basin random-IC retry on the map engine-kernel path, so a result
        that triggers the retry is reproducible.

    Returns
    -------
    ScalarResult
        Estimated maximal exponent (per unit time / per iteration), a drop-in for
        its ``float`` value that also carries ``.meta`` / ``.summary()``.

    Raises
    ------
    NotImplementedError
        If ``system`` is a delay system (it has no ``set_state``).
    ValueError
        If ``dt`` is passed for a discrete map.
    ConvergenceError
        If the two trajectories collapse or diverge (zero / non-finite
        separation), or a continuous system's clock does not advance.

    Examples
    --------
    >>> max_lyapunov(Lorenz(ic=[1.0, 1.0, 1.0]), dt=0.05)   # ≈ 0.91

    References
    ----------
    G. Benettin, L. Galgani & J.-M. Strelcyn, "Kolmogorov entropy and numerical
    experiments", *Physical Review A* **14** (1976) 2338--2345.
    """
    if isinstance(system, DelaySystem):
        raise NotImplementedError(
            "max_lyapunov needs set_state, which delay systems cannot support — "
            "use DelaySystem.lyapunov_spectrum (the engine estimator) instead."
        )
    if system.is_discrete and dt is not None:
        raise InvalidParameterError(
            "dt has no meaning for discrete maps — omit it (every step is one iteration)."
        )

    # Maps: the maximal exponent is the leading entry of the QR tangent-map
    # spectrum, run in one Rust engine call (stream perf/map-lyapunov-kernel) —
    # thousands of times faster than the per-iteration two-trajectory rescaling, and
    # more robust (no perturbation-size / collapse tuning).  The continuous-system
    # path below is unchanged.  Only the map path moves to the kernel; a map whose
    # ``_step`` will not lower, or a wheel-free environment, falls back to the
    # two-trajectory loop transparently.
    if system.is_discrete:
        mle = _max_lyapunov_map(
            system, n=n, steps_per=steps_per, transient=transient, ic=ic, seed=seed
        )
        if mle is not None:
            meta = AnalysisResult.build_meta(
                system, analysis="max_lyapunov", n=n, transient=transient
            )
            return ScalarResult(value=mle, meta=meta)

    rng = np.random.default_rng(seed)
    ref = system.copy()
    ref.reinit(ic)
    for _ in range(transient):
        ref.step(dt)

    pert = system.copy()
    direction = rng.normal(size=system.dim)
    direction *= d0 / np.linalg.norm(direction)
    pert.reinit(ref.state() + direction)

    t_start = ref.time()
    log_sum = 0.0
    for _ in range(n):
        for _ in range(steps_per):
            ref.step(dt)
            pert.step(dt)
        delta = pert.state() - ref.state()
        d = float(np.linalg.norm(delta))
        if d == 0.0 or not np.isfinite(d):
            raise ConvergenceError(
                "max_lyapunov: trajectories collapsed or diverged — "
                "try a larger d0 or smaller steps_per."
            )
        log_sum += np.log(d / d0)
        pert.set_state(ref.state() + (d0 / d) * delta)

    if system.is_discrete:
        elapsed = float(n * steps_per)
    else:
        # Normalize by the *actual* elapsed integration time, read from the
        # reference trajectory's clock — robust to whatever per-step advance the
        # system makes when ``dt`` is ``None`` (built-in flows step by their own
        # ``_default_step_dt``; a continuous ``WrappedSystem`` steps by its
        # ``default_dt``). Guessing a step-size attribute name silently rescales
        # the exponent whenever the guess misses the real per-step advance.
        elapsed = float(ref.time() - t_start)
        if elapsed <= 0.0 or not np.isfinite(elapsed):
            raise ConvergenceError(
                "max_lyapunov: the reference clock did not advance — a continuous "
                "system must report elapsed time through time(); pass an explicit dt."
            )
    mle = float(log_sum / elapsed)
    meta = AnalysisResult.build_meta(system, analysis="max_lyapunov", n=n, transient=transient)
    return ScalarResult(value=mle, meta=meta)


# Self-register the headline Lyapunov quantifiers (D4 / §4e: in-tree analyses
# register from their own subpackage).  Idempotent across re-imports — `register`
# keeps the same object under the same name.
_registrations: tuple[tuple[str, Callable[..., Any], dict[str, Any]], ...] = (
    ("lyapunov_spectrum", lyapunov_spectrum, {"needs": "system", "family": "lyapunov"}),
    ("max_lyapunov", max_lyapunov, {"needs": "system", "family": "lyapunov"}),
    ("lyapunov_from_data", lyapunov_from_data, {"needs": "series", "family": "lyapunov"}),
    ("kaplan_yorke_dimension", kaplan_yorke_dimension, {"needs": "spectrum", "family": "lyapunov"}),
)
for _name, _fn, _meta in _registrations:
    _registry.analyses.register(_name, _fn, **_meta)
del _name, _fn, _meta, _registrations


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
