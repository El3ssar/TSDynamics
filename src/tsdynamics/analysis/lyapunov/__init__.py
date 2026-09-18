"""Lyapunov-based quantifiers: spectra, maximal exponent, Kaplan–Yorke dimension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from tsdynamics.errors import (
    ConvergenceError,
    InvalidInputError,
    InvalidParameterError,
    remedy,
)
from tsdynamics.families import DelaySystem
from tsdynamics.utils.escape import escaped

from .._common import reject_data, reject_system
from .._discovery import register as _register
from .._result import ArrayResult, ScalarResult, _build_meta
from .._result_json import _vector
from .from_data import LyapunovFromData, ScalingRegionWarning, lyapunov_from_data

#: The data-first sibling every system-first Lyapunov entry point points at when
#: it is handed a measured series: the same question (how fast do nearby states
#: separate?) answered without a model, from the series alone.
_FROM_DATA_LINE = "ts.analysis.lyapunov_from_data({data})"

__all__ = [
    "LyapunovFromData",
    "LyapunovSpectrum",
    "ScalingRegionWarning",
    "kaplan_yorke_dimension",
    "lyapunov_from_data",
    "lyapunov_spectrum",
]

#: Fraction of the two-trajectory cycles run as an *unaveraged* alignment
#: warm-up, so the random initial perturbation has settled onto the leading
#: Lyapunov direction before the accumulation begins.
_ALIGN_FRACTION = 0.05

#: Default averaging window for the two-trajectory fallback on a **flow**, in
#: time units.  The window — not a cycle count — is what sets the accuracy of a
#: Benettin average, so this is what the default holds fixed across ``dt``.
#: Measured on Lorenz against an independent variational integration (true
#: 0.90763): window 20 -> 0.659, 40 -> 0.747, 100 -> 0.890, 200 -> 0.889,
#: 400 -> 0.907.
_DEFAULT_WINDOW = 200.0

#: Default horizon for the two-trajectory fallback on a **map**, in ITERATIONS.
_DEFAULT_MAP_ITERATIONS = 20_000

#: Protocol steps between rescalings of the two-trajectory loop.
_DEFAULT_STEPS_PER = 10

#: Separation the two trajectories are reset to after every rescaling.
_DEFAULT_D0 = 1e-9

#: How many off-basin random initial conditions a map's burn-in will try before
#: giving up and re-raising the divergence.
_MAX_IC_RETRIES = 10

#: Floor on the automatically-sized cycle count, so a coarsely-stepped flow still
#: averages over enough independent rescalings.
_MIN_CYCLES = 200

#: Burn-in a **map**'s QR iteration discards before it starts accumulating, in
#: iterations.  The Lyapunov exponent is a property of the *attractor*, so the
#: iterates spent falling onto it are not part of the measurement; the QR
#: iteration starting cold measured them anyway.  Folded in from the retired
#: ``max_lyapunov``, which always did this and was the more accurate of the two
#: doors because of it — measured on Hénon at ``n = 20 000`` over 12 random
#: starts (literature 0.41922): cold ``MAE = 0.00156, sd = 0.00182``, burnt-in
#: ``MAE = 0.00087, sd = 0.00111``.
_DEFAULT_BURN_ITERATIONS = 500


@dataclass(frozen=True, eq=False)
class LyapunovSpectrum(ArrayResult):
    """The Lyapunov spectrum of a system — the exponents and the result surface.

    An :class:`~tsdynamics.analysis._result.ArrayResult`, so it is a drop-in for
    the bare exponent array: ``np.asarray(result)``, indexing, iteration and
    ``result.shape`` all defer to the wrapped exponents, while it also carries
    ``.meta`` / the readout ``repr`` / ``.to_dict()`` / the ``.plot`` seam.

    The repr names the dynamics — see :meth:`_interpretation` for the rule and
    why it refuses to name one when the horizon does not support it.

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

    @property
    def _is_flow(self) -> bool:
        """Whether the spectrum came from a continuous-time system.

        A flow has at least one *structural* zero exponent (translation along
        the orbit); a map has none.  That single fact is what calibrates the
        classification below, so it has to be right — it is read off the
        registry entry for ``meta["system"]``, falling back to the horizon
        keyword the estimator recorded.  Absent evidence the answer is "map",
        the conservative choice: the map floor is a fixed relative one, so a
        misread flow is *hedged* by the band test rather than mislabelled.
        """
        return self._subject_family() in ("ode", "dde", "sde")

    @property
    def _zero_tolerance(self) -> float:
        """The floor below which an exponent counts as zero, in this spectrum.

        Calibrated on the estimator's **own realised zero**: a flow has a
        structural zero exponent, so ``min|λ|`` measures how close to zero this
        estimator got at this horizon — which is exactly the tolerance the
        classification needs.  A map has no structural zero and keeps the
        relative floor.
        """
        e = np.abs(np.asarray(self.values, dtype=float))
        if e.size == 0:
            return 0.0
        base = 1e-3 * float(e.max())
        return max(base, float(e.min())) if self._is_flow else max(1e-6, base)

    def _n_positive(self) -> tuple[int, int]:
        """Return the positive-exponent count at ``lo`` and at ``10 × lo``."""
        e = np.asarray(self.values, dtype=float)
        lo = self._zero_tolerance
        return int((e > lo).sum()), int((e > 10.0 * lo).sum())

    def _interpretation(self) -> str | None:
        """Name the dynamics, or refuse to.

        The rule (v6, contract §4.4).  The old one thresholded at ``1e-3`` of
        the largest magnitude, which is blind to how well the estimator actually
        resolved zero at the horizon it was given, and so read a flow's
        imperfectly-converged zero exponent as a second positive one:
        ``lyapunov_spectrum(Lorenz, final_time=20)`` was reported
        **hyperchaotic**, and ``LotkaVolterra`` — a shipped conservative system
        — is reported **chaotic** at every horizon.

        Two changes fix both.  The floor is the estimator's own realised zero
        (:attr:`_zero_tolerance`), and the verdict is printed **only when the
        count is stable across a 10× tolerance band**; when it is not, the
        honest answer is that the horizon is too short, and the repr says so
        instead of naming a regime.

        Scored over every catalogue system carrying a literature
        ``known_lyapunov`` (20 systems): the old rule is wrong once, this one is
        wrong zero times with one honest hedge (``HyperBao``, whose ``k``
        defaults to 2 so the spectrum is truncated and the classification
        genuinely cannot be made).

        A σ-carrying estimator and a properly-tested classifier are a v6.1
        ticket; this is a **repr-only** rule with no estimator change.
        """
        e = np.asarray(self.values, dtype=float)
        if e.size == 0:
            return None
        if self.unbounded:
            # Exponents of an orbit that left the building are not attractor
            # exponents: measured, an escaped Chua run reported a confident
            # λ = [0.3057, 0.3054, -6.068] and hedged only about the horizon.
            return "⚠ the orbit is UNBOUNDED — these are not attractor exponents"
        n_lo, n_hi = self._n_positive()
        if n_lo != n_hi:
            return (
                f"indeterminate at this horizon (n_pos = {n_hi}..{n_lo} across a 10× "
                "tolerance band; raise final_time)"
            )
        word = "hyperchaotic" if n_lo >= 2 else "chaotic" if n_lo == 1 else "regular"
        if n_lo == 0:
            # No expanding direction, so the Kaplan--Yorke dimension is the whole
            # (integer) state space and reporting it would look like a measurement.
            return word
        dky = self.kaplan_yorke
        return f"{word} · D_KY = {dky:.4g}" if np.isfinite(dky) else word

    @property
    def unbounded(self) -> bool:
        """Whether the orbit these exponents were measured on **escaped**.

        Read off ``meta["orbit_peak"]`` — the magnitude of the state the
        estimator landed on, recorded by
        :func:`tsdynamics.families.base.orbit_peak`.  A run that blows up
        without reaching the engine's hard ``1e150`` guard still returns a full,
        finite spectrum; this is what tells that apart from an answer.
        """
        return escaped(self.meta.get("orbit_peak") if self.meta else None)

    @property
    def n_positive(self) -> int:
        """How many exponents are positive, at :attr:`_zero_tolerance`."""
        return self._n_positive()[0]

    @property
    def chaotic(self) -> bool | None:
        """Is this chaotic?  ``None`` when the horizon cannot support a verdict.

        The verdict the repr prints, as a value you can branch on — ``if
        spec.chaotic:`` is the line a user writes next, and before v6.0 the
        library did the whole classification (including the significance floor)
        and then threw it away inside a formatted string.

        Three-valued on purpose, and the third value is the point: it is
        ``None`` exactly when :meth:`_interpretation` says *indeterminate at this
        horizon*, so a short run cannot be read as a confident "no".

        Examples
        --------
        >>> import tsdynamics as ts
        >>> ts.analysis.lyapunov_spectrum(ts.systems.Lorenz(), final_time=200.0).chaotic
        True
        """
        n_lo, n_hi = self._n_positive()
        if np.asarray(self.values).size == 0 or n_lo != n_hi or self.unbounded:
            return None
        return n_lo >= 1

    @property
    def regime(self) -> str:
        """``"chaotic"`` / ``"hyperchaotic"`` / ``"regular"`` / ``"indeterminate"``."""
        n_lo, n_hi = self._n_positive()
        if self.unbounded:
            return "unbounded"
        if np.asarray(self.values).size == 0 or n_lo != n_hi:
            return "indeterminate"
        return "hyperchaotic" if n_lo >= 2 else "chaotic" if n_lo == 1 else "regular"

    def _answer(self) -> str:
        r"""Return ``λ = [...]`` — the exponents, each at its own scale."""
        return f"λ = {_vector(self.values)}"

    def _details(self) -> tuple[str, ...]:
        """Return the audit line: the floor the verdict was decided at."""
        e = np.asarray(self.values, dtype=float)
        if e.size == 0:
            return ()
        kind = "flow (realised zero)" if self._is_flow else "map (relative floor)"
        word = "exponent" if e.size == 1 else "exponents"
        return (f"({e.size} {word} · {kind} · λ > {self._zero_tolerance:.3g})",)

    def _derived(self) -> dict[str, Any]:
        """Export the derived answers the repr reports."""
        n_lo, _ = self._n_positive()
        return {
            "kaplan_yorke": self.kaplan_yorke,
            "n_positive": n_lo,
            "zero_tolerance": self._zero_tolerance,
            "chaotic": self.chaotic,
            "regime": self.regime,
        }

    def __plot_spec__(self, kind: str | None = None) -> Any:
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

    When you already hold a :class:`LyapunovSpectrum`, ``spectrum.kaplan_yorke``
    is the same number and the shorter spelling — it calls this.  Reach for the
    function when the exponents came from somewhere else: a literature spectrum,
    a hand-written array, another library's output.  That is what it takes, and
    why both exist::

        ts.analysis.kaplan_yorke_dimension([0.906, 0.0, -14.572])   # Lorenz, from the paper

    Parameters
    ----------
    spectrum : array-like
        Lyapunov exponents (any order; sorted descending internally).  Must be
        non-empty and finite — an empty spectrum has no dimension to report and a
        ``nan``/``inf`` exponent means the estimator did not converge, so both
        raise rather than return a plausible number.

    Returns
    -------
    ScalarResult
        The dimension, a drop-in for its ``float`` value (``float(result)`` /
        comparisons work).  The documented conventions at the edges of the
        definition:

        * every exponent negative (:math:`\lambda_1 < 0`) — a stable fixed point:
          ``0.0``;
        * the cumulative sum never turns negative — the spectrum is *incomplete*
          (a truncated ``k < dim`` spectrum, or a conservative system whose
          exponents sum to zero, e.g. ``[1, -1]``): saturates at
          ``len(spectrum)``, since :math:`D_{KY}` cannot exceed the number of
          directions supplied.

        :math:`\lambda_{j+1}` cannot be zero at the interpolation step: if it
        were, the cumulative sum at ``j + 1`` would equal the one at ``j`` and
        still be non-negative, so ``j`` would not have been the last such index.

    Raises
    ------
    InvalidParameterError
        If ``spectrum`` is empty, is not one-dimensional, or holds a non-finite
        exponent.

    Examples
    --------
    >>> float(kaplan_yorke_dimension([0.906, 0.0, -14.57]))   # Lorenz
    2.062...
    >>> float(kaplan_yorke_dimension([-0.5, -1.0]))           # stable fixed point
    0.0
    >>> float(kaplan_yorke_dimension([1.0, -1.0]))            # conservative: saturates
    2.0

    References
    ----------
    J. L. Kaplan & J. A. Yorke, "Chaotic behavior of multidimensional difference
    equations", in *Functional Differential Equations and Approximation of Fixed
    Points*, Lecture Notes in Mathematics **730**, Springer (1979) 204--227.
    """
    # No ``hint=``: the shared builder already knows this is a *result*-first
    # analysis and which analysis produces its subject.  The bespoke hint opened
    # with "expects measured data" — which this function does not — and that
    # wrong clause is the one CONTRACT §5.6 names.
    reject_system(spectrum, analysis="kaplan_yorke_dimension")
    s = np.asarray(spectrum, dtype=float)
    if s.ndim == 0:
        # A single number is never a spectrum: the formula needs a negative
        # exponent to interpolate against, so a scalar always saturates and
        # returns a confident 1.0.  ``kaplan_yorke_dimension(spec[0])`` is the
        # natural way to make this mistake -- both are "Lyapunov things" -- so
        # refuse it rather than answer it.
        raise InvalidParameterError(
            f"kaplan_yorke_dimension needs the whole Lyapunov spectrum, not a single "
            f"exponent (got {float(s):.6g}). The dimension is read off where the "
            f"running sum of the exponents changes sign, so one number carries no "
            f"dimension — it would saturate at 1.0 whatever its value."
            + remedy(
                "exps = ts.analysis.lyapunov_spectrum(system)",
                "ts.analysis.kaplan_yorke_dimension(exps)",
                lead="Compute the full spectrum first:",
            )
        )
    s = np.atleast_1d(s)
    if s.ndim != 1:
        raise InvalidParameterError(
            f"kaplan_yorke_dimension: spectrum must be one-dimensional, got shape {s.shape}."
        )
    if s.size == 0:
        raise InvalidParameterError(
            "kaplan_yorke_dimension: the spectrum is empty — there is no dimension to "
            "report. Pass at least one Lyapunov exponent."
        )
    if not np.all(np.isfinite(s)):
        raise InvalidParameterError(
            f"kaplan_yorke_dimension: the spectrum holds a non-finite exponent "
            f"({np.array2string(s, precision=4)}); the estimator did not converge, so "
            "any dimension read off it would be meaningless."
        )
    s = np.sort(s)[::-1]
    if s[0] < 0.0:
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
    solver: str | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    backend: str | None = None,
    reortho_interval: int | None = None,
    d0: float | None = None,
    steps_per: int | None = None,
    **aliases: Any,
) -> LyapunovSpectrum:
    """Lyapunov spectrum — lambda_1 > 0 is the chaos test.

    The **one** entry point for every family, and since v6 the one entry point
    for the maximal exponent too: ask for ``k=1``.

    Dispatches to the family implementation (QR tangent dynamics for maps, the
    extended variational system on the engine for ODEs, the engine
    function-space estimator for DDEs), translating this one signature to each
    family's native keywords.  The exponents are obtained by Benettin
    renormalisation of an evolving orthonormal frame (Benettin et al. 1980).
    A system with **no right-hand side to differentiate** — a
    :class:`~tsdynamics.families.WrappedSystem` around an opaque external
    stepper — is answered instead by Benettin *two-trajectory* rescaling, which
    forms no Jacobian; it resolves one exponent, so ``k > 1`` is refused there.

    .. versionchanged:: 6.0
        Absorbed ``max_lyapunov``, which was a second public door onto the same
        question and answered it with a different number: on Hénon at one
        nominal horizon it returned 0.4233 where this function returned 0.4160.
        The two things it had that this one lacked came with it — the **map
        burn-in** (see ``transient``, now on by default) and the
        **two-trajectory** machine for a system with no Jacobian.

    Parameters
    ----------
    system : System
        A flow (ODE/DDE) or a discrete map.
    k : int, optional
        Number of exponents to compute.  Defaults to
        ``system.dim`` for flows/maps; a DDE may request more than ``dim`` (its
        tangent space is the infinite-dimensional history).
    final_time : float, optional
        Averaging-window length for a **flow** (after the transient).  Mutually
        exclusive with ``n``; a flow uses ``final_time``.
    n : int, optional
        Number of iterations for a **map**.  Mutually exclusive with
        ``final_time``; a map uses ``n``.
    **aliases
        ``steps=`` only — the alias of ``n`` that a map's
        :meth:`~tsdynamics.families.discrete.DiscreteMap.iterate` already
        accepts, taken here too so the flat function speaks the same horizon
        vocabulary as the method it wraps.  Passing both ``n`` and ``steps``
        raises, as does any other keyword.
    rtol, atol : float, optional
        Solver tolerances for the flow / DDE variational integration (they have
        no meaning for a map, whose tangent iteration is exact arithmetic).
    transient : float, optional
        Dynamics discarded before the exponents are accumulated — **time units**
        for a flow (the same unit as ``final_time``, and the unit
        ``run(transient=)`` uses), a count of **ITERATIONS** for a map.

        A Lyapunov exponent is a property of the *attractor*, so the orbit has
        to be on it first; a map's default is
        :data:`_DEFAULT_BURN_ITERATIONS` iterations, and ``transient=0``
        disables it for a caller who has already landed their own ``ic``.

        .. versionchanged:: 6.0
            A map used to *refuse* this word — "its QR iteration
            reorthonormalises from the initial condition, so there is nothing to
            discard", which confuses the tangent frame with the base orbit.  The
            burn-in is folded in from the retired ``max_lyapunov``, which always
            did it and was measurably the more accurate door because of it.
    dt : float, optional
        Integration step for the variational system, in **time units** (flows
        only; a map iterates).  Defaults to the family's own step.
    ic : array-like, optional
        Initial condition.  Falls back to ``system.ic``, then random.
    solver : str, optional
        Numerical kernel for the variational integration (continuous flows
        only), e.g. ``"rk45"`` / ``"bdf"``.  ``solver=`` picks a *kernel*;
        ``method=`` picks an *estimator*, and this analysis has one — so
        ``method=`` raises, naming this word.

        .. versionchanged:: 6.0
            Was ``method=``.  v6 renamed the kernel word to ``solver=`` at
            ``run()`` for exactly this reason, and this door had not followed —
            so obeying the library's own rule here used to be a dead end.
    backend : {"jit", "interp", "reference"}, optional
        Which engine runs the tangent dynamics.  Named here because this free
        function is the ONLY door in v6 (ruling A2): a keyword the removed method
        accepted and this one refuses is a signature bug, not a curation (C1).
    reortho_interval : int, optional
        Iterations between Gram--Schmidt reorthonormalisations (maps only).
    d0 : float, optional
        Separation the two trajectories are reset to after every rescaling —
        **only** for a system with no right-hand side to differentiate, which is
        the one case answered that way.  Default ``1e-9``.  A system whose
        variational machine exists refuses it by name: nothing is being
        perturbed there.

        It is a real knob, not a formality: on an exactly-linear expanding flow
        the growth rate is independent of the perturbation size, and a ``d0``
        that small against a growing reference loses the separation to
        floating-point cancellation — so that measurement needs ``d0=1e-4``.

        .. versionadded:: 6.0
            Carried over from the retired ``max_lyapunov``, whose two-trajectory
            machine this is.
    steps_per : int, optional
        Protocol steps between rescalings of that same two-trajectory loop
        (default 10).  Refused, by name, for a system that has a tangent frame —
        a frame is reorthonormalised on its own cadence
        (``reortho_interval``), not rescaled.

        .. versionadded:: 6.0
            Carried over from the retired ``max_lyapunov``.

    Returns
    -------
    LyapunovSpectrum
        The exponents (largest first), a drop-in for the bare ``(k,)`` array —
        ``np.asarray(result)``, indexing and iteration work — that also carries
        ``.meta``, ``.summary()`` and the ``.kaplan_yorke`` dimension.

    Raises
    ------
    InvalidInputError
        If ``system`` is measured data (a ``Trajectory`` / array — estimate the
        exponent from the series with :func:`lyapunov_from_data` instead), or
        offers neither a right-hand side to differentiate nor a ``set_state``
        for two trajectories to be rescaled through.  A ``TypeError`` subclass,
        so an existing ``except TypeError`` keeps catching it.
    ValueError
        If ``k <= 0``, or a keyword is passed to the wrong family (``final_time``
        / ``dt`` / ``solver`` for a map; ``n`` for a flow; or a ``solver`` for a
        DDE, which selects its engine via ``backend``), or ``k > 1`` is asked of
        a system with no Jacobian (two trajectories resolve one exponent).

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
    reject_data(system, analysis="lyapunov_spectrum", sibling=_FROM_DATA_LINE)
    method_fn = getattr(system, "_lyapunov_spectrum", None)
    if k is not None:
        # Typed, and checked BEFORE the tape is built — ``k`` is an option value,
        # so the library's ``InvalidParameterError`` (a ``ValueError``) is the
        # class, and a non-integer must not reach a bare ``'<=' not supported``.
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise InvalidParameterError(
                f"k (number of exponents) must be a positive integer, got {k!r}."
                + remedy("ts.analysis.lyapunov_spectrum(system, k=2)")
            )
        if k <= 0:
            raise InvalidParameterError(
                f"k (number of exponents) must be a positive integer, got {k!r}."
                + remedy("ts.analysis.lyapunov_spectrum(system, k=2)")
            )

    # ``steps`` is the alias of ``n`` a map's ``iterate`` / ``run`` accept, so it
    # reaches this door too: the flat function must take the same horizon words
    # as the method it wraps, or ``hen.lyapunov_spectrum(steps=2000)`` working
    # while ``ts.analysis.lyapunov_spectrum(hen, steps=2000)`` raises is a signature bug
    # a user has no way to predict.  It is accepted through ``**aliases`` rather
    # than named in the signature on purpose: ``n`` is the declared vocabulary
    # (the naming gate in ``test_polish_standards.py`` bans ``steps`` as a
    # *parameter spelling*), and an alias is a courtesy, not a second word.
    steps = aliases.pop("steps", None)
    if "method" in aliases:
        # v6 split the two words everywhere else — ``solver=`` selects a numerical
        # kernel, ``method=`` selects an *estimator* — and ``run()`` already says
        # so in its own refusal.  This door kept the old spelling, so a user who
        # obeyed the rule they had just been taught hit "unexpected keyword
        # 'solver'" and had nowhere to go.  Same sentence, same fix, both doors.
        raise InvalidParameterError(
            "method= selects an *estimator*, and lyapunov_spectrum has only one "
            "(Benettin QR); the numerical kernel is solver=."
            + remedy('ts.analysis.lyapunov_spectrum(system, solver="bdf")')
        )
    if aliases:
        # An unknown keyword is a call-shape error, so it must stay catchable as
        # the ``TypeError`` the interpreter would have raised (``InvalidInputError``
        # subclasses it); ``InvalidParameterError`` is a ``ValueError`` and is for
        # a keyword that exists with a bad value.
        raise InvalidInputError(
            f"lyapunov_spectrum got an unexpected keyword argument {sorted(aliases)[0]!r}. "
            f"It takes k, final_time (a flow) / n (a map), transient, dt, ic, solver, "
            f"rtol, atol, backend and reortho_interval."
            + remedy("ts.analysis.lyapunov_spectrum(system, k=2, final_time=200.0)")
        )
    if steps is not None:
        if n is not None:
            raise InvalidParameterError(
                f"n and steps are the same argument (the iteration count), so pass "
                f"only one; got n={n!r} and steps={steps!r}."
                + remedy("ts.analysis.lyapunov_spectrum(system, n=5000)")
            )
        n = steps

    if method_fn is None:
        # No variational machine — an adapted external stepper (a
        # ``WrappedSystem``) has no right-hand side to differentiate and no tape
        # to lower.  The Benettin TWO-TRAJECTORY loop needs neither, so it is not
        # a second implementation of anything here: it is the only one, and it
        # lives behind this door rather than behind a second public name.  That
        # is the whole of the v6 fold — ``max_lyapunov`` was a second door onto
        # the same question that answered it with a different number (Hénon:
        # 0.4233 against this function's 0.4160 at one nominal horizon).
        return _two_trajectory_spectrum(
            system,
            k=k,
            final_time=final_time,
            n=n,
            dt=dt,
            transient=transient,
            ic=ic,
            solver=solver,
            reortho_interval=reortho_interval,
            d0=d0,
            steps_per=steps_per,
        )

    for word, value, why in (
        ("d0", d0, "the separation two rescaled trajectories are reset to"),
        ("steps_per", steps_per, "the number of protocol steps between those rescalings"),
    ):
        if value is not None:
            raise InvalidParameterError(
                f"{word} is {why}, and {type(system).__name__} has a right-hand side to "
                f"differentiate — so its exponents come from a tangent FRAME carried "
                f"alongside the orbit, with nothing perturbed and nothing rescaled."
                + remedy("ts.analysis.lyapunov_spectrum(system, k=2)")
            )

    fwd: dict[str, Any] = {}
    if k is not None:
        fwd["k"] = k
    if ic is not None:
        fwd["ic"] = ic
    if backend is not None:
        fwd["backend"] = backend

    if getattr(system, "family", None) == "map":
        # Maps: horizon is `n` (iterations); no time, solver or burn-in concept.
        # Every refusal below names the unit, because the mistake is never a typo
        # — it is the right word for the other kind of dynamics.
        if final_time is not None:
            raise InvalidParameterError(
                "final_time is a horizon in TIME UNITS and a map has no continuous time; "
                "a map's horizon is n, a count of ITERATIONS."
                + remedy("ts.analysis.lyapunov_spectrum(system, n=5000)")
            )
        if dt is not None:
            raise InvalidParameterError(
                "dt is an integration step in TIME UNITS and a map has no continuous time "
                "— every map step is exactly one iteration."
                + remedy("ts.analysis.lyapunov_spectrum(system, n=5000)")
            )
        # ``transient`` is a burn-in in ITERATIONS here, and it defaults to
        # :data:`_DEFAULT_BURN_ITERATIONS` rather than to nothing.  This door used
        # to REFUSE the word ("a map spectrum has nothing to discard: the QR
        # iteration reorthonormalises from the initial condition"), which
        # confuses reorthonormalisation — a property of the tangent frame — with
        # landing on the attractor, a property of the base orbit: iterates spent
        # falling onto the attractor are stretched by the *transient* dynamics,
        # and averaging them in is exactly why the two Lyapunov doors returned
        # two numbers.  The retired ``max_lyapunov`` burnt in and was the more
        # accurate of the pair; that burn-in now lives here.
        transient = transient_n = _resolve_map_transient(transient)
        if transient_n:
            fwd["ic"] = _burn_in_map(system, ic=ic, iterations=transient_n)
        if solver is not None:
            raise InvalidParameterError(
                "solver names a numerical kernel for a differential equation, and a map is "
                "iterated exactly — there is no kernel to choose."
                + remedy("ts.analysis.lyapunov_spectrum(system, n=5000)")
            )
        if n is not None:
            fwd["n"] = n
        if reortho_interval is not None:
            fwd["reortho_interval"] = reortho_interval
    else:
        if reortho_interval is not None:
            raise InvalidParameterError(
                "reortho_interval counts ITERATIONS between reorthonormalisations, so it "
                "is a map keyword; a flow reorthonormalises once per dt chunk."
                + remedy("ts.analysis.lyapunov_spectrum(system, dt=0.01)")
            )
        # Flows (ODE/DDE): horizon is `final_time`; transient is a burn-in time.
        if n is not None:
            raise InvalidParameterError(
                "n is a count of ITERATIONS and a flow advances in continuous time; its "
                "horizon is final_time, a length in TIME UNITS."
                + remedy("ts.analysis.lyapunov_spectrum(system, final_time=200.0)")
            )
        if final_time is not None:
            fwd["final_time"] = final_time
        if transient is not None:
            fwd["transient"] = transient
        if dt is not None:
            fwd["dt"] = dt
        if solver is not None:
            if isinstance(system, DelaySystem):
                raise InvalidParameterError(
                    "a DDE's variational system is integrated by the engine's method of "
                    "steps, which has no selectable kernel; choose the engine with backend=."
                    + remedy('ts.analysis.lyapunov_spectrum(system, backend="jit")')
                )
            fwd["method"] = solver
        # The solver tolerances the family estimator has always taken.  They are
        # named here because the free function is the ONLY door in v6 (ruling
        # A2), so a keyword the removed method accepted and this one refuses is a
        # signature bug, not a curation (C1).
        if rtol is not None:
            fwd["rtol"] = rtol
        if atol is not None:
            fwd["atol"] = atol
    estimate = method_fn(**fwd)
    exponents = np.asarray(estimate, dtype=float)
    meta = _build_meta(
        system,
        analysis="lyapunov_spectrum",
        k=int(exponents.size),
        final_time=final_time,
        n=n,
        transient=transient,
        # Carry the family estimator's record of where the orbit LANDED, so a
        # spectrum measured on an escaping orbit can refuse a verdict instead of
        # hedging about the horizon.  This function rebuilds ``meta`` from
        # scratch, so anything the family recorded has to be re-read here.
        orbit_peak=getattr(estimate, "meta", {}).get("orbit_peak"),
    )
    return LyapunovSpectrum(values=exponents, meta=meta)


def _resolve_map_transient(transient: float | None) -> int:
    """Read a **map**'s ``transient`` as a whole number of iterations.

    ``None`` is :data:`_DEFAULT_BURN_ITERATIONS` — the burn-in is on by default,
    because a Lyapunov exponent is a property of the attractor and the iterates
    spent falling onto it are not part of it.  ``0`` disables it, for a caller
    who has already landed their own initial condition.
    """
    if transient is None:
        return _DEFAULT_BURN_ITERATIONS
    value = float(transient)
    if not np.isfinite(value) or value < 0.0:
        raise InvalidParameterError(
            f"transient is the number of ITERATIONS a map discards before the "
            f"exponents are accumulated, so it must be finite and >= 0; got "
            f"{transient!r}." + remedy("ts.analysis.lyapunov_spectrum(system, transient=500)")
        )
    if value != int(value):
        raise InvalidParameterError(
            f"transient counts ITERATIONS for a map, so it must be a whole number; "
            f"got {transient!r}." + remedy("ts.analysis.lyapunov_spectrum(system, transient=500)")
        )
    return int(value)


def _burn_in_map(system: Any, *, ic: Any | None, iterations: int) -> np.ndarray:
    """Iterate a map from ``ic`` past the burn-in; return the landed state.

    **The off-basin retry lives here, not only downstream.**  A map with no
    ``_default_ic`` draws a random start, and for some of the catalogue
    (``FoldedTowel``) a good fraction of the box is outside the basin: the
    family estimator already retried such a draw, but the burn-in runs *before*
    it, so an unguarded loop turned a retryable draw into a hard
    ``ConvergenceError`` — measured, ``FoldedTowel`` failed roughly half its
    calls with "map diverged at iteration 12".  Retries come from a **fixed**
    ``default_rng(0)``, so a repeated call gives a repeated answer, and an
    explicit ``ic`` is never silently replaced by a different starting point.
    """
    rng = np.random.default_rng(0)
    for attempt in range(_MAX_IC_RETRIES):
        start = ic if attempt == 0 else rng.random(int(system.dim))
        ref = system.copy()
        try:
            ref.reinit(start)
            for _ in range(int(iterations)):
                ref.step()
            landed = np.asarray(ref.state(), dtype=float)
        except (ConvergenceError, ArithmeticError):
            if ic is not None or attempt == _MAX_IC_RETRIES - 1:
                raise
            continue
        if np.all(np.isfinite(landed)):
            return landed
        if ic is not None:
            raise ConvergenceError(
                f"{type(system).__name__}: the orbit left the attractor during the "
                f"{iterations}-iteration burn-in, so there is nothing to measure "
                f"exponents on." + remedy("ts.analysis.lyapunov_spectrum(system, transient=0)")
            )
    raise ConvergenceError(  # pragma: no cover - the loop returns or re-raises
        f"{type(system).__name__}: iterates diverge from every tried initial condition."
    )


def _burn_in_flow(ref: Any, *, dt: float | None, transient: float | None) -> int:
    """Advance ``ref`` past the burn-in; return the number of steps it took.

    ``transient`` is a **time**, so the loop watches the clock rather than
    counting a number of steps chosen for it — which is what makes the burn-in
    the same amount of *dynamics* at every ``dt``.  ``None`` is the historical
    default and is still a step count, so the shipped number does not move.  The
    step count comes back because the caller sizes its averaging window from the
    measured per-step advance.
    """
    if transient is None:
        for _ in range(_DEFAULT_BURN_ITERATIONS):
            ref.step(dt)
        return _DEFAULT_BURN_ITERATIONS
    if transient <= 0.0:
        return 0
    t0 = float(ref.time())
    steps = 0
    # A step count still bounds the loop: a system whose clock does not advance
    # (a WrappedSystem driven externally) must not spin forever.
    budget = _DEFAULT_BURN_ITERATIONS * 1000
    while float(ref.time()) - t0 < transient and steps < budget:
        ref.step(dt)
        steps += 1
    return steps


def _two_trajectory_spectrum(
    system: Any,
    *,
    k: int | None,
    final_time: float | None,
    n: int | None,
    dt: float | None,
    transient: float | None,
    ic: Any | None,
    solver: str | None,
    reortho_interval: int | None,
    d0: float | None,
    steps_per: int | None,
) -> LyapunovSpectrum:
    r"""Measure the maximal exponent of a system with no variational machine.

    Benettin two-trajectory rescaling (Benettin, Galgani & Strelcyn 1976): a
    reference and a perturbed copy stepped in lockstep through the
    :class:`~tsdynamics.families.System` protocol, the separation rescaled back
    to a small :math:`d_0` every cycle and :math:`\ln(d/d_0)` averaged over the
    elapsed time.  It forms no Jacobian, so it reaches what the variational path
    cannot: a :class:`~tsdynamics.families.WrappedSystem` around an opaque
    external stepper.

    It returns **one** exponent — two trajectories measure one separation rate —
    so ``k > 1`` is refused by name rather than silently truncated.

    The perturbation starts in a *random* direction, so its first cycles measure
    a mixture of every exponent (for a dissipative flow it initially *shrinks*)
    and only then aligns with the leading Lyapunov direction.  Those alignment
    cycles are run and rescaled but **not** averaged
    (:data:`_ALIGN_FRACTION` of the cycle count, the standard Benettin warm-up).

    References
    ----------
    G. Benettin, L. Galgani & J.-M. Strelcyn, "Kolmogorov entropy and numerical
    experiments", *Physical Review A* **14** (1976) 2338--2345.
    """
    name = type(system).__name__
    if k is not None and int(k) != 1:
        raise InvalidParameterError(
            f"{name} has no right-hand side to differentiate, so its exponents are "
            f"measured by rescaling TWO trajectories — which resolves the single "
            f"fastest separation rate, not a k={int(k)} frame. Ask for one exponent, "
            f"or give the dynamics to a family that owns its equations."
            + remedy("ts.analysis.lyapunov_spectrum(system, k=1)")
        )
    if solver is not None:
        raise InvalidParameterError(
            f"solver names a numerical kernel for a differential equation, and "
            f"{name} steps itself — there is no kernel to choose."
            + remedy("ts.analysis.lyapunov_spectrum(system, k=1)")
        )
    if reortho_interval is not None:
        raise InvalidParameterError(
            "reortho_interval spaces the Gram-Schmidt reorthonormalisations of a "
            "tangent FRAME, and two trajectories carry no frame — the separation is "
            "rescaled every cycle." + remedy("ts.analysis.lyapunov_spectrum(system, k=1)")
        )
    if not hasattr(system, "set_state"):
        raise InvalidInputError(
            f"lyapunov_spectrum needs either a right-hand side to differentiate or a "
            f"state it can set, and {name} offers neither."
            + remedy("ts.analysis.lyapunov_from_data(traj)")
        )
    is_map = getattr(system, "family", None) == "map"
    if is_map and dt is not None:
        raise InvalidParameterError(
            "dt is an integration step in TIME UNITS and a map has no continuous time "
            "— every map step is exactly one iteration."
            + remedy("ts.analysis.lyapunov_spectrum(system, n=20000)")
        )
    if is_map and final_time is not None:
        raise InvalidParameterError(
            "final_time is a horizon in TIME UNITS and a map has no continuous time; "
            "a map's horizon is n, a count of ITERATIONS."
            + remedy("ts.analysis.lyapunov_spectrum(system, n=20000)")
        )
    if not is_map and n is not None:
        raise InvalidParameterError(
            "n is a count of ITERATIONS and a flow advances in continuous time; its "
            "horizon is final_time, a length in TIME UNITS."
            + remedy("ts.analysis.lyapunov_spectrum(system, final_time=200.0)")
        )

    if steps_per is not None and (int(steps_per) < 1 or steps_per != int(steps_per)):
        raise InvalidParameterError(
            f"steps_per counts the protocol steps between rescalings, so it must be a "
            f"whole number >= 1; got {steps_per!r}."
            + remedy("ts.analysis.lyapunov_spectrum(system, k=1, steps_per=10)")
        )
    steps_per = _DEFAULT_STEPS_PER if steps_per is None else int(steps_per)
    if d0 is None:
        d0 = _DEFAULT_D0
    elif not np.isfinite(d0) or d0 <= 0.0:
        # ln(d / d0) with a zero d0 divides by zero and with a negative one takes
        # the log of a negative number; both used to surface as a bare
        # ZeroDivisionError / nan from deep inside the cycle loop.
        raise InvalidParameterError(
            f"d0 is the separation the two trajectories are reset to after every "
            f"rescaling, so it must be a small positive number; got {d0!r}."
            + remedy("ts.analysis.lyapunov_spectrum(system, k=1, d0=1e-9)")
        )
    rng = np.random.default_rng(0)  # the answer must not move between two calls
    ref = system.copy()
    ref.reinit(ic)
    t_burn = float(ref.time())
    transient_steps = _burn_in_flow(ref, dt=dt, transient=transient)

    if is_map:
        # A map has no ``dt`` to shorten the window, so a cycle count IS a window.
        horizon = _DEFAULT_MAP_ITERATIONS if n is None else int(n)
        n_cycles = max(1, horizon // max(1, steps_per))
    else:
        # Size the averaging window in TIME, read off the reference clock — the
        # per-step advance is whatever the system actually makes, and guessing it
        # is what made the estimate dt-dependent.
        per_step = (float(ref.time()) - t_burn) / transient_steps if transient_steps > 0 else 0.0
        if not np.isfinite(per_step) or per_step <= 0.0:
            t_probe = float(ref.time())  # transient=0: one probe step
            ref.step(dt)
            per_step = float(ref.time()) - t_probe
        if not np.isfinite(per_step) or per_step <= 0.0:
            raise ConvergenceError(
                "lyapunov_spectrum: the reference clock did not advance, so the "
                "averaging window cannot be sized — pass an explicit dt."
            )
        window = _DEFAULT_WINDOW if final_time is None else float(final_time)
        if not np.isfinite(window) or window <= 0.0:
            raise InvalidParameterError(
                f"final_time is the averaging-window length, so it must be finite and "
                f"> 0; got {final_time!r}."
                + remedy("ts.analysis.lyapunov_spectrum(system, final_time=200.0)")
            )
        n_cycles = max(_MIN_CYCLES, int(np.ceil(window / (steps_per * per_step))))

    pert = system.copy()
    direction = rng.normal(size=system.dim)
    direction *= d0 / np.linalg.norm(direction)
    pert.reinit(ref.state() + direction)

    def cycle() -> float:
        """Advance both trajectories one cycle, rescale, return ``ln(d / d0)``."""
        for _ in range(steps_per):
            ref.step(dt)
            pert.step(dt)
        delta = pert.state() - ref.state()
        d = float(np.linalg.norm(delta))
        if d == 0.0 or not np.isfinite(d):
            raise ConvergenceError(
                "lyapunov_spectrum: the two rescaled trajectories collapsed or "
                "diverged, so no separation rate can be read off them."
            )
        pert.set_state(ref.state() + (d0 / d) * delta)
        return float(np.log(d / d0))

    for _ in range(max(1, int(n_cycles * _ALIGN_FRACTION))):
        cycle()
    t_start = ref.time()
    log_sum = 0.0
    for _ in range(n_cycles):
        log_sum += cycle()

    if is_map:
        elapsed = float(n_cycles * steps_per)  # per ITERATION
    else:
        # Normalize by the *actual* elapsed integration time, read off the
        # reference clock: guessing a step-size attribute name silently rescales
        # the exponent whenever the guess misses the real per-step advance.
        elapsed = float(ref.time() - t_start)
        if elapsed <= 0.0 or not np.isfinite(elapsed):
            raise ConvergenceError(
                "lyapunov_spectrum: the reference clock did not advance — a continuous "
                "system must report elapsed time through time(); pass an explicit dt."
            )
    meta = _build_meta(
        system,
        analysis="lyapunov_spectrum",
        k=1,
        final_time=final_time,
        n=n,
        transient=transient,
        estimator="two-trajectory",
        window=elapsed,
    )
    return LyapunovSpectrum(values=np.array([log_sum / elapsed], dtype=float), meta=meta)


# Self-register the quantifiers: the definition site is the registration site
# (CONTRACT §7.7), through the public ``ts.analysis.register`` door.
_register(
    lyapunov_spectrum,
    subjects=("system",),
    area="lyapunov",
    returns=LyapunovSpectrum,
    keywords=(
        "chaotic chaos exponents spectrum predictability benettin largest maximal "
        "mle divergence rate sensitivity butterfly horizon"
    ),
    cite="Benettin, Galgani, Giorgilli & Strelcyn (1980), Meccanica 15, 9",
    doi="10.1007/BF02128236",
)
_register(
    lyapunov_from_data,
    subjects=("trajectory", "array"),
    area="lyapunov",
    returns=LyapunovFromData,
    keywords="chaotic chaos measured series kantz rosenstein",
    cite="Kantz (1994), Phys. Lett. A 185, 77",
    doi="10.1016/0375-9601(94)90991-1",
)
_register(
    kaplan_yorke_dimension,
    subjects=("LyapunovSpectrum",),
    area="lyapunov",
    returns=ScalarResult,
    keywords="fractal dimension attractor lyapunov information",
    cite="Kaplan & Yorke (1979), Lecture Notes in Mathematics 730, 204",
    doi="10.1007/BFb0064319",
)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
