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

from .._common import reject_data, reject_system
from .._discovery import register as _register
from .._result import AnalysisResult, ArrayResult, ScalarResult
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
    "max_lyapunov",
]

#: Fraction of ``max_lyapunov``'s cycles run as an *unaveraged* alignment
#: warm-up, so the random initial perturbation has settled onto the leading
#: Lyapunov direction before the accumulation begins.
_ALIGN_FRACTION = 0.05

#: Default averaging window for ``max_lyapunov`` on a **flow**, in time units.
#: The window — not a cycle count — is what sets the accuracy of a Benettin
#: average, so this is what the default holds fixed across ``dt``.  Measured on
#: Lorenz against an independent variational integration (true 0.90763): window
#: 20 -> 0.659, 40 -> 0.747, 100 -> 0.890, 200 -> 0.889, 400 -> 0.907.
_DEFAULT_WINDOW = 200.0

#: Default horizon for ``max_lyapunov`` on a **map**, in ITERATIONS — the same
#: unit ``lyapunov_spectrum(map, n=)`` counts in, which is what lets one door
#: delegate to the other.  It is the pre-v6 default expressed honestly: 2000
#: rescaling cycles of ``steps_per = 10`` iterations was always 20 000 iterates.
_DEFAULT_MAP_ITERATIONS = 20_000

#: Floor on the automatically-sized cycle count, so a coarsely-stepped flow still
#: averages over enough independent rescalings.
_MIN_CYCLES = 200

#: Historical burn-in for ``max_lyapunov``, in protocol steps (a flow) or
#: iterations (a map).  It is what ``transient=None`` resolves to, so the shipped
#: default is bit-identical to the pre-v6 ``transient: int = 500``.
_DEFAULT_BURN_STEPS = 500

#: How many off-basin initial conditions ``max_lyapunov`` will try on a map
#: before giving up and re-raising the divergence.
_MAX_IC_RETRIES = 10

#: An ``int`` ``transient`` at or above this on a **flow** is refused rather than
#: read as a time.  ``transient`` means time for a flow now; the pre-v6 spelling
#: was a step count, and a step count large enough to be one (500 steps ≈ 5 time
#: units at the default ``dt``) would silently become a ~100x longer burn-in.
_TRANSIENT_AMBIGUITY_FLOOR = 100


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
        if np.asarray(self.values).size == 0 or n_lo != n_hi:
            return None
        return n_lo >= 1

    @property
    def regime(self) -> str:
        """``"chaotic"`` / ``"hyperchaotic"`` / ``"regular"`` / ``"indeterminate"``."""
        n_lo, n_hi = self._n_positive()
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
        return (f"({e.size} exponents · {kind} · λ > {self._zero_tolerance:.3g})",)

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
        # returns a confident 1.0.  ``kaplan_yorke_dimension(max_lyapunov(sys))``
        # is the natural way to make this mistake -- both are "Lyapunov things"
        # -- so refuse it rather than answer it.
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
    **aliases: Any,
) -> LyapunovSpectrum:
    """Lyapunov spectrum — lambda_1 > 0 is the chaos test.

    The uniform, documented entry point for every family.

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
        Dynamics discarded before averaging, in **time units** — the same unit
        as ``final_time``, and the same unit ``run(transient=)`` uses for a flow.
        A **map** takes none here: its QR iteration reorthonormalises from the
        initial condition, so there is nothing to discard.
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
        exponent from the series with :func:`lyapunov_from_data` instead), or has
        no ``lyapunov_spectrum`` implementation (e.g. a derived wrapper — compute
        the spectrum on the underlying system).  A ``TypeError`` subclass, so an
        existing ``except TypeError`` keeps catching it.
    ValueError
        If ``k <= 0``, or a keyword is passed to the wrong family (``final_time``
        / ``dt`` / ``transient`` / ``solver`` for a map; ``n`` for a flow; or a
        ``solver`` for a DDE, which selects its engine via ``backend``).

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
    if method_fn is None:
        raise InvalidInputError(
            f"lyapunov_spectrum() needs a system that implements it, and "
            f"{type(system).__name__} does not — a derived wrapper measures the "
            f"exponents of the system it wraps."
            + remedy(
                "ts.analysis.lyapunov_spectrum(wrapper.system)",
                lead="Compute the spectrum on the underlying system:",
            )
        )
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
            "method= selects an *estimator* in v6, and lyapunov_spectrum has only one "
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
        if transient is not None:
            raise InvalidParameterError(
                "transient is a burn-in and a map spectrum has nothing to discard: the QR "
                "iteration reorthonormalises from the initial condition. Burn in yourself "
                "and pass the landed state."
                + remedy("ts.analysis.lyapunov_spectrum(system, n=5000, ic=traj.y[-1])")
            )
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


def _resolve_transient(transient: float | None, *, is_map: bool) -> float | None:
    """Read ``transient`` in the unit its family measures dynamics in.

    Time for a flow, iterations for a map — the rule ``run(transient=)`` already
    follows, and the rule this function's own docstring states.  ``None`` comes
    back as ``None``, meaning "the historical burn-in of
    :data:`_DEFAULT_BURN_STEPS` protocol steps", so no shipped default moves.

    The ``int``-on-a-flow refusal is the whole point of the guard: before v6
    this keyword counted *protocol steps* for both families, so the same
    ``transient=500`` that meant "≈5 time units" would now mean 500 time units
    — a 100x longer burn-in, with no error and no visible symptom beyond the
    wait.  A value that large cannot be disambiguated, so it is named, not
    guessed.
    """
    if transient is None:
        return None
    value = float(transient)
    if not np.isfinite(value) or value < 0.0:
        unit = "iterations" if is_map else "time units"
        raise InvalidParameterError(
            f"transient is the dynamics discarded before measuring, in {unit}, so it "
            f"must be finite and >= 0; got {transient!r}."
            + remedy("ts.analysis.max_lyapunov(system)")
        )
    if is_map:
        if value != int(value):
            raise InvalidParameterError(
                f"transient counts ITERATIONS for a map, so it must be a whole number; "
                f"got {transient!r}." + remedy("ts.analysis.max_lyapunov(system, transient=500)")
            )
        return value
    if (
        isinstance(transient, (int, np.integer))
        and not isinstance(transient, bool)
        and int(transient) >= _TRANSIENT_AMBIGUITY_FLOOR
    ):
        raise InvalidParameterError(
            f"transient={transient} is ambiguous: in v6 it is the dynamics discarded "
            f"before measuring in TIME UNITS for a flow (it counted protocol steps "
            f"before), and {transient} time units is about {transient} / "
            f"dt steps — roughly 100x the old burn-in. Say which you meant:\n"
            f"    ts.analysis.max_lyapunov(system, transient={float(transient)})"
            f"   # {transient} time units\n"
            f"    ts.analysis.max_lyapunov(system)"
            f"   # the default burn-in ({_DEFAULT_BURN_STEPS} steps)"
        )
    return value


def _burn_in_map(system: Any, *, ic: Any | None, iterations: float | None) -> np.ndarray:
    """Iterate a map from ``ic`` past the burn-in; return the landed state."""
    ref = system.copy()
    ref.reinit(ic)
    count = _DEFAULT_BURN_STEPS if iterations is None else int(iterations)
    for _ in range(count):
        ref.step()
    return np.asarray(ref.state(), dtype=float)


def _map_max_exponent(
    system: Any,
    *,
    iterations: int,
    transient: float | None,
    ic: Any | None,
    seed: int | None,
    backend: str | None,
) -> float:
    """Return a map's maximal exponent: burn in, then ask :func:`lyapunov_spectrum`.

    The estimator is not re-implemented here — that is the whole point of the
    v6 fold.  What *is* here is the **initial-condition policy**, because it is
    ``max_lyapunov``'s own contract: a random draw can land off-basin and
    diverge, and a retry that re-drew from the unseeded global RNG made a seeded
    call non-reproducible.  So the retry IC comes from ``default_rng(seed)``, and
    an explicit ``ic`` that diverges re-raises instead of being quietly replaced
    by a different starting point.  Passing the landed state on as an explicit
    ``ic`` also keeps the delegate's own (unseeded) retry out of the picture.
    """
    rng = np.random.default_rng(seed)
    dim = int(system.dim)
    ic_explicit = ic is not None
    for attempt in range(_MAX_IC_RETRIES):
        if attempt == 0:
            start_ic = ic
        elif seed is None:
            start_ic = None  # unseeded random fallback (the family's resolve_ic)
        else:
            start_ic = rng.random(dim)  # reproducible random retry IC
        try:
            landed = _burn_in_map(system, ic=start_ic, iterations=transient)
            spectrum = lyapunov_spectrum(system, k=1, n=iterations, ic=landed, backend=backend)
        except (ConvergenceError, ArithmeticError):
            if ic_explicit or attempt == _MAX_IC_RETRIES - 1:
                raise
            continue
        return float(np.asarray(spectrum)[0])
    raise ConvergenceError(  # pragma: no cover - the loop returns or re-raises
        f"{type(system).__name__}.max_lyapunov: iterates diverge from every tried IC."
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
        for _ in range(_DEFAULT_BURN_STEPS):
            ref.step(dt)
        return _DEFAULT_BURN_STEPS
    if transient <= 0.0:
        return 0
    t0 = float(ref.time())
    steps = 0
    # A step count still bounds the loop: a system whose clock does not advance
    # (a WrappedSystem driven externally) must not spin forever.
    budget = _DEFAULT_BURN_STEPS * 1000
    while float(ref.time()) - t0 < transient and steps < budget:
        ref.step(dt)
        steps += 1
    return steps


def max_lyapunov(
    system: Any,
    *,
    d0: float = 1e-9,
    n: int | None = None,
    final_time: float | None = None,
    steps_per: int = 10,
    dt: float | None = None,
    transient: float | None = None,
    ic: Any | None = None,
    seed: int | None = 0,
    backend: str | None = None,
) -> ScalarResult:
    r"""Maximal Lyapunov exponent — the one number that says "chaotic".

    Two machines behind one name, chosen by family, and they are chosen for you:

    **A map** is answered by :func:`lyapunov_spectrum` with ``k=1`` — literally,
    by calling it — so the two doors cannot disagree.  The QR tangent-map
    iteration runs in one Rust kernel call and is both faster and more robust
    than rescaling two orbits (no ``d0`` to tune, nothing to collapse).

    **A flow** is answered by Benettin two-trajectory rescaling (Benettin et al.
    1976): a reference and a perturbed copy stepped in lockstep through the
    :class:`~tsdynamics.families.System` protocol, the separation rescaled back
    to ``d0`` every cycle and :math:`\ln(d / d_0)` averaged over elapsed time.
    **Choose it over ``lyapunov_spectrum`` when there is no usable Jacobian** —
    it never forms one, so it works on a non-smooth right-hand side and on a
    :class:`~tsdynamics.families.WrappedSystem` wrapping an external stepper,
    where the variational path cannot go.  When a Jacobian *is* available,
    ``lyapunov_spectrum(system, k=1)[0]`` is the more accurate answer and gives
    you the rest of the spectrum for the same integration.

    .. versionchanged:: 6.0
        The map path is now a delegation rather than a second implementation,
        and ``n`` counts **iterations** on a map, as it does at
        :func:`lyapunov_spectrum`.  It used to count *rescaling cycles* of
        ``steps_per`` iterations each, so ``max_lyapunov(m, n=20000)`` did ten
        times the work of ``lyapunov_spectrum(m, k=1, n=20000)`` and the two
        returned different numbers for one nominal horizon.

    Neither path is available for DDEs (their state cannot be ``set_state``-ed);
    use ``ts.analysis.lyapunov_spectrum(dde, k=1)`` instead.

    The perturbation starts in a *random* direction, so its first cycles measure
    a mixture of every exponent — for a dissipative flow it initially **shrinks**
    (Lorenz's first ten cycles contribute a negative log-ratio) and only then
    aligns with the leading Lyapunov direction.  Those alignment cycles are run
    and rescaled but **not** averaged (:data:`_ALIGN_FRACTION` of the cycle count,
    the standard Benettin warm-up); counting them, as this function used to,
    biased a short flow run low by ~25 %.

    The accuracy of the average is set by the **length of the averaging window in
    time**, so for a flow that is what the default holds fixed — see ``n``.

    Parameters
    ----------
    system : System
        A flow (ODE) or a discrete map.
    d0 : float
        Perturbation size restored at every rescaling (flows only).
    n : int, optional
        The horizon, counted in the unit the family iterates in:

        * a **map** — a count of **ITERATIONS**, exactly as at
          :func:`lyapunov_spectrum`.  Default :data:`_DEFAULT_MAP_ITERATIONS`.
        * a **flow** — a count of **RESCALING CYCLES**, each ``steps_per``
          protocol steps.  ``None`` (the default) sizes the averaging *window*
          instead of the count: enough cycles to cover :data:`_DEFAULT_WINDOW`
          time units, measured from the reference clock, so the estimate does
          not move with ``dt``.  (A fixed count *is* a window of
          ``n * steps_per * dt`` time, so it silently shortens as ``dt``
          shrinks: with the pre-v6 ``n = 2000`` default, ``dt = 0.001``
          averaged over 20 time units and returned 0.659 for Lorenz against a
          true 0.906.)  Prefer ``final_time`` — it says what it means.
    final_time : float, optional
        Averaging-window length for a **flow**, in **time units** — the quantity
        that actually sets the accuracy of the estimate, expressed directly
        instead of via a cycle count.  It is the same word, in the same unit,
        that :func:`lyapunov_spectrum` and every family's ``run`` already use
        for a horizon.  Overrides the :data:`_DEFAULT_WINDOW` default; mutually
        exclusive with ``n`` (which fixes the cycle count instead), and rejected
        for a **map**, whose horizon is a count of iterations (``n``).

        .. versionadded:: 6.0
            Previously the window could only be reached indirectly, by solving
            ``n * steps_per * dt`` for ``n`` — so the one keyword every other
            Lyapunov entry point spelled ``final_time`` was simply missing here.
    steps_per : int, default 10
        Protocol steps between rescalings (**flows only** — a map's QR iteration
        reorthonormalises every iterate, so there is no cycle length to set).
    dt : float, optional
        Integration step for a flow, in **time units** (default: the system's
        own step).  Rejected for a map, which iterates.
    transient : float, optional
        Dynamics discarded before measuring — **time units** for a flow
        (the unit ``final_time`` and ``run(transient=)`` use), **iterations**
        for a map.  ``None`` (the default) keeps the historical burn-in:
        :data:`_DEFAULT_BURN_STEPS` protocol steps for a flow,
        :data:`_DEFAULT_BURN_STEPS` iterations for a map.

        .. versionchanged:: 6.0
            Was a count of protocol steps *for both families*, so the one word
            the rest of the library measures in time measured in steps here.
            An integer ``>= 100`` on a flow is refused rather than silently
            read as a 100x longer burn-in — see :exc:`InvalidParameterError`.
    ic : array-like, optional
        Initial condition for the reference trajectory.
    seed : int, default 0
        Seed for the random perturbation direction (flow) and for the off-basin
        random-IC retry (map), so a repeated call returns the same number.
        Pass ``seed=None`` for an explicitly unseeded draw.

        .. versionchanged:: 6.0
            Was ``None``, so the default answer was not reproducible.
    backend : {"jit", "interp", "reference"}, optional
        Which engine runs a **map**'s QR tangent iteration; forwarded to
        :func:`lyapunov_spectrum`.  A flow's two-trajectory loop drives the
        system through the stepping protocol and takes the system's own backend.

    Returns
    -------
    ScalarResult
        Estimated maximal exponent — **per unit time** for a flow, **per
        iteration** for a map — a drop-in for its ``float`` value that also
        carries ``.meta`` and the ``.plot`` seam.

    See Also
    --------
    lyapunov_spectrum : all ``k`` exponents; the more accurate answer whenever
        the system has a usable Jacobian, and what this delegates to for a map.
    lyapunov_from_data : the same question from a measured series, with no model.

    Raises
    ------
    InvalidInputError
        If ``system`` is measured data rather than a model — use
        :func:`lyapunov_from_data` on the series.
    NotImplementedError
        If ``system`` is a delay system (it has no ``set_state``).
    InvalidParameterError
        If a keyword carries the other family's unit: ``dt`` / ``final_time``
        for a map (both are times, and a map has none), ``steps_per`` for a map,
        or an ``int`` ``transient >= 100`` for a flow — which is the pre-v6
        step-count spelling of a keyword that now means time, and is refused
        rather than silently read as a 100x longer burn-in.
    ConvergenceError
        If the two trajectories collapse or diverge (zero / non-finite
        separation), or a continuous system's clock does not advance (so neither
        the averaging window nor the elapsed time can be read from it).

    Examples
    --------
    >>> max_lyapunov(Lorenz(ic=[1.0, 1.0, 1.0]))   # ≈ 0.89, literature 0.906

    References
    ----------
    G. Benettin, L. Galgani & J.-M. Strelcyn, "Kolmogorov entropy and numerical
    experiments", *Physical Review A* **14** (1976) 2338--2345.
    """
    reject_data(system, analysis="max_lyapunov", sibling=_FROM_DATA_LINE)
    if isinstance(system, DelaySystem):
        raise NotImplementedError(
            "max_lyapunov needs set_state, which delay systems cannot support."
            + remedy(
                "ts.analysis.lyapunov_spectrum(system, k=1, dt=0.5)",
                lead="Use the delay system's own engine estimator:",
            )
        )
    is_map = system.family == "map"
    # The fold applies exactly where ``lyapunov_spectrum`` can answer.  An adapted
    # external stepper (``WrappedSystem(family="map")``) has no ``_lyapunov_spectrum``
    # — no Jacobian, no tape — so for it the two-trajectory loop is not a second
    # implementation of anything, it is the only machine, and its cycle vocabulary
    # (``n`` cycles of ``steps_per`` iterates) stands.
    delegates = is_map and callable(getattr(system, "_lyapunov_spectrum", None))
    if is_map and dt is not None:
        raise InvalidParameterError(
            "dt is an integration step in TIME UNITS and a map has no continuous time "
            "— every map step is exactly one iteration."
            + remedy("ts.analysis.max_lyapunov(system, n=20000)")
        )
    if delegates and steps_per != 10:
        # ``steps_per`` was the cycle length of the two-trajectory loop.  The map
        # path is now ``lyapunov_spectrum(k=1)``, whose QR iteration
        # reorthonormalises every iterate — there is no cycle to lengthen, and
        # silently ignoring the word would leave ``n`` meaning ``n * steps_per``
        # to the caller and ``n`` to the library.
        raise InvalidParameterError(
            f"steps_per is the number of protocol steps between rescalings of the "
            f"two-trajectory loop, and a map does not use it: its maximal exponent is "
            f"the top of the QR tangent-map spectrum, which reorthonormalises every "
            f"iterate. Count iterations with n instead; got steps_per={steps_per!r}."
            + remedy(f"ts.analysis.max_lyapunov(system, n={20000 if n is None else int(n)})")
        )
    transient_n = _resolve_transient(transient, is_map=is_map)
    if not (d0 > 0.0) or not np.isfinite(d0):
        # The separation is rescaled back to d0 every cycle and the log-ratio is
        # ln(d / d0): a zero d0 divides by zero, a negative one takes the log of a
        # negative number.  Both used to surface as a bare ZeroDivisionError /
        # nan from deep inside the cycle loop.
        raise InvalidParameterError(
            f"d0 is the separation the two trajectories are reset to after every "
            f"rescaling, so it must be a small positive number; got {d0!r}."
            + remedy("ts.analysis.max_lyapunov(system, d0=1e-9)")
        )
    if n is not None and int(n) < 1:
        unit = "iterations" if is_map else "measured rescaling cycles"
        raise InvalidParameterError(
            f"n is a count of {unit}, so it must be >= 1; got {n!r}. "
            f"Omit it to size the horizon automatically."
            + remedy("ts.analysis.max_lyapunov(system)")
        )
    if final_time is not None:
        # ``n`` (a cycle count) and ``final_time`` (a window in time) set the same
        # quantity two different ways, so accepting both would leave one silently
        # ignored.  A map has no clock for a time window to mean anything on.
        if is_map:
            raise InvalidParameterError(
                "final_time is a window in TIME UNITS, and a map has no continuous "
                "time — its horizon is n, a count of ITERATIONS."
                + remedy("ts.analysis.max_lyapunov(system, n=20000)")
            )
        if n is not None:
            raise InvalidParameterError(
                f"n and final_time both size the averaging window (a cycle count vs a "
                f"length in time), so pass only one; got n={n!r} and "
                f"final_time={final_time!r}."
                + remedy("ts.analysis.max_lyapunov(system, final_time=200.0)")
            )
        if not np.isfinite(final_time) or final_time <= 0.0:
            raise InvalidParameterError(
                f"final_time is the averaging-window length, so it must be finite and "
                f"> 0; got {final_time!r}."
                + remedy("ts.analysis.max_lyapunov(system, final_time=200.0)")
            )

    # A map's maximal exponent IS the top of its Lyapunov spectrum, so this asks
    # for it — by calling the public function, not by running a second copy of the
    # same QR tangent-map iteration.  Two implementations of one quantity is how
    # ``max_lyapunov(m, n=20000)`` and ``lyapunov_spectrum(m, k=1, n=20000)`` came
    # to return different numbers for one nominal horizon (they counted different
    # things), and a delegation cannot drift.
    if delegates:
        iterations = _DEFAULT_MAP_ITERATIONS if n is None else int(n)
        mle = _map_max_exponent(
            system,
            iterations=iterations,
            transient=transient_n,
            ic=ic,
            seed=seed,
            backend=backend,
        )
        meta = AnalysisResult.build_meta(
            system, analysis="max_lyapunov", n=iterations, transient=transient_n
        )
        return ScalarResult(value=mle, meta=meta)

    n_cycles = 0 if n is None else int(n)  # resolved from the clock below
    rng = np.random.default_rng(seed)
    ref = system.copy()
    ref.reinit(ic)
    t_burn = float(ref.time())
    transient_steps = _burn_in_flow(ref, dt=dt, transient=transient_n)

    if n is None and is_map:
        # A stepper-only map (no ``_lyapunov_spectrum`` to delegate to): the cycle
        # count already IS a fixed window, because a map has no ``dt`` to shorten it.
        n_cycles = _DEFAULT_MAP_ITERATIONS // max(1, steps_per)
    elif n is None:
        # Size the averaging window in TIME, read off the reference clock — the
        # per-step advance is whatever the system actually makes (an explicit
        # ``dt``, a family default, a WrappedSystem's own step), and guessing it
        # is what made the estimate dt-dependent.
        per_step = (float(ref.time()) - t_burn) / transient_steps if transient_steps > 0 else 0.0
        if not np.isfinite(per_step) or per_step <= 0.0:
            t_probe = float(ref.time())  # transient=0: one probe step
            ref.step(dt)
            per_step = float(ref.time()) - t_probe
        if not np.isfinite(per_step) or per_step <= 0.0:
            raise ConvergenceError(
                "max_lyapunov: the reference clock did not advance, so the "
                "averaging window cannot be sized — pass an explicit n (and dt)."
            )
        window = _DEFAULT_WINDOW if final_time is None else float(final_time)
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
                "max_lyapunov: trajectories collapsed or diverged — "
                "try a larger d0 or smaller steps_per."
            )
        pert.set_state(ref.state() + (d0 / d) * delta)
        return float(np.log(d / d0))

    # Alignment warm-up: rescale but do not average, so the random initial
    # direction has collapsed onto the leading Lyapunov direction before the
    # accumulation starts (see the docstring).  The clock starts after it.
    for _ in range(max(1, int(n_cycles * _ALIGN_FRACTION))):
        cycle()
    t_start = ref.time()
    log_sum = 0.0
    for _ in range(n_cycles):
        log_sum += cycle()

    if is_map:
        # Per ITERATION, and a map's clock counts them, so the window is exact.
        elapsed = float(n_cycles * steps_per)
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
    meta = AnalysisResult.build_meta(
        system,
        analysis="max_lyapunov",
        n=n_cycles,
        transient=transient_n,
        final_time=final_time,
        window=elapsed,
    )
    return ScalarResult(value=mle, meta=meta)


# Self-register the quantifiers: the definition site is the registration site
# (CONTRACT §7.7), through the public ``ts.analysis.register`` door.
_register(
    lyapunov_spectrum,
    subjects=("system",),
    area="lyapunov",
    returns=LyapunovSpectrum,
    keywords="chaotic chaos exponents spectrum predictability benettin",
    cite="Benettin, Galgani, Giorgilli & Strelcyn (1980), Meccanica 15, 9",
    doi="10.1007/BF02128236",
)
_register(
    max_lyapunov,
    subjects=("system",),
    area="lyapunov",
    returns=ScalarResult,
    keywords="chaotic chaos largest exponent predictability benettin",
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
