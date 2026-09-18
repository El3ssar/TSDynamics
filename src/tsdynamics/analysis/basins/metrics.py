r"""
Quantifiers of a basin diagram.

These read a basin *image* — a labelled grid from
:func:`~tsdynamics.analysis.basins.basins.basins` (or a raw integer
array) — and need no further integration, so they are cheap and exact on a
synthetic label grid:

- :func:`basin_entropy` — the basin entropy :math:`S_b` and boundary basin
  entropy :math:`S_{bb}` of Daza et al. (2016); :math:`S_{bb} > \log 2` is a
  sufficient condition for a fractal boundary.
- :func:`uncertainty_exponent` — the final-state-sensitivity exponent
  :math:`\alpha` of Grebogi, McDonald, Ott & Yorke (1983):
  :math:`f(\varepsilon)\sim\varepsilon^{\alpha}`, with boundary dimension
  :math:`D_0 = D - \alpha`.
- :func:`wada_property` — a grid test for the Wada property (Daza et al., 2015):
  every boundary cell sees all basins.
- :func:`resilience` — the minimal-fatal-shock distance (Halekotte & Feudel,
  2020): how far an attractor sits from its basin boundary.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ...errors import InvalidInputError, InvalidParameterError, invalid_value, remedy
from .._common import reject_system as _reject_system
from .._result import AnalysisResult, ScalarResult
from .._result_json import _sig
from ._common import _as_label_array
from .basins import BasinsResult

#: Minimum number of basins for the Wada test to mean anything: the Wada property
#: is DEFINED as every boundary point touching three or more basins (Kennedy &
#: Yorke 1991), so with two basins there is nothing to test — see
#: :attr:`WadaResult.applicable`.
_WADA_MIN_BASINS = 3

#: Below this R^2 the uncertainty-exponent power-law fit is not read as one, so
#: the repr withholds the predictability verdict rather than naming a regime off
#: a line that does not describe the data.
_FIT_ACCEPTABLE_R2 = 0.9

#: alpha at or below which the boundary is called final-state sensitive.  Grebogi
#: et al. (1983): alpha = D - D_0, so alpha = 1 is a smooth boundary and a small
#: alpha means halving the initial uncertainty barely improves predictability.
_FINAL_STATE_SENSITIVE_ALPHA = 0.8

#: Fewest uncertainty radii a power-law verdict may be read from.  Below four,
#: :data:`_FIT_ACCEPTABLE_R2` is passed by construction rather than by
#: measurement — a straight line through two points scores ``R² = 1``.
_MIN_UNCERTAINTY_RADII = 4

#: How much the LOCAL slope of ``log f`` vs ``log eps`` may decay across the
#: probed window and still be read as one power law: ``last / first`` must be at
#: least this.
#:
#: This is the gate that stops the verdict being a property of the grid rather
#: than of the system.  Measured on an unforced double-well oscillator — whose
#: boundary is a saddle's stable manifold, provably **smooth**, D0 = 1 — over
#: grids of 30/40/60/80/100 cells per side::
#:
#:     n     alpha   R^2      last/first local slope   verdict before   after
#:     30    0.676   0.9726   0.32                     FRACTAL (wrong)  withheld
#:     40    0.789   0.9926   0.48                     FRACTAL (wrong)  withheld
#:     60    0.883   0.9980   0.84                     smooth           smooth
#:     80    0.909   0.9975   0.76                     smooth           smooth
#:     100   0.941   0.9989   0.85                     smooth           smooth
#:
#: Note ``R^2 = 0.9926`` on the 40-cell run that answered "fractal boundary":
#: :math:`R^2` measures how straight the fitted line is, **not** whether the
#: underlying relation is a power law, so it cannot see the systematic decay.
#: The local slope can, and 0.6 separates the two regimes with margin on both
#: sides (0.48 | 0.76).  An engineer reading "fractal boundary" concludes the
#: safety margin is meaningless and stops, so the cost of a false positive here
#: is a project, not a re-run.
_SLOPE_STABLE_RATIO = 0.6


def _resolve_attractor_id(result: BasinsResult, attractor_id: int | None) -> int:
    """Pick which attractor to measure, or say which ones there are.

    ``resilience`` is the one basin metric that is *about* a particular
    attractor, so it cannot be defaulted in general — but a single-attractor
    image has only one answer, and when there is a real choice the caller needs
    the ids that exist, not the word "attractor_id".
    """
    present = sorted(int(i) for i in np.unique(result.labels) if int(i) >= 1)
    if attractor_id is not None:
        if int(attractor_id) not in present:
            raise invalid_value(
                "attractor_id",
                attractor_id,
                options=present,
                hint="those are the basin labels present in this image.",
            )
        return int(attractor_id)
    if len(present) == 1:
        return present[0]
    if not present:
        raise InvalidInputError(
            "this basin image has no attractor to measure — every cell diverged or "
            "went unlabelled, so there is no basin and no boundary."
            + remedy(
                "res = ts.analysis.basins(system, [(-2.0, 2.0, 200), (-2.0, 2.0, 200)])",
                lead="Widen the region (or raise max_steps) so trajectories settle:",
            )
        )
    raise InvalidInputError(
        f"resilience measures one attractor's distance to its basin boundary, so it "
        f"needs to know which: this image holds {len(present)} attractors, "
        f"labelled {present}." + remedy(f"ts.analysis.resilience(res, attractor_id={present[0]})")
    )


__all__ = [
    "BasinEntropy",
    "UncertaintyExponent",
    "WadaResult",
    "basin_entropy",
    "resilience",
    "uncertainty_exponent",
    "wada_property",
]


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BasinEntropy(AnalysisResult):
    r"""
    Basin entropy and boundary basin entropy of a basin diagram.

    Attributes
    ----------
    sb : float
        Basin entropy :math:`S_b` — the mean Gibbs entropy over all boxes.
    sbb : float
        Boundary basin entropy :math:`S_{bb}` — the mean over boundary boxes only
        (those holding more than one basin).  ``nan`` if there is no boundary box.
    n_boxes : int
        Number of boxes the grid was partitioned into.
    n_boundary_boxes : int
        Number of boxes containing more than one basin.
    fractal_boundary : bool
        ``True`` when :math:`S_{bb} > \log 2`, the sufficient fractal-boundary
        criterion of Daza et al. (2016).

    Notes
    -----
    ``box_size`` (the box side length in cells) and ``log_base`` (the base of the
    logarithm, ``e`` by default) are the caller's own settings echoed back, so
    they are carried and exported but kept off ``dir()`` — R1, contract §11.3.
    Both still resolve and both appear in :meth:`to_dict`.
    """

    #: R1 — inputs echoed back.  The measurements here are the two entropies and
    #: the two box counts.
    _HIDDEN_ATTRIBUTES: ClassVar[frozenset[str]] = frozenset({"box_size", "log_base"})

    #: The repr and ``to_dict`` both write Daza's own casing — ``Sb`` / ``Sbb`` —
    #: so both must resolve and both must be listed.  They did neither: the repr
    #: printed ``Sb = 0.4057 · Sbb = 0.5617`` while only the lower-case fields
    #: existed, which is a name the library teaches and then does not have.
    _extra_attribute_names: ClassVar[tuple[str, ...]] = ("Sb", "Sbb")

    sb: float = 0.0
    sbb: float = 0.0
    n_boxes: int = 0
    n_boundary_boxes: int = 0
    box_size: int = 0
    log_base: float = 0.0
    fractal_boundary: bool = False

    @property
    def Sb(self) -> float:  # noqa: N802
        """Basin entropy, in Daza's casing — the same number as :attr:`sb`."""
        return float(self.sb)

    @property
    def Sbb(self) -> float:  # noqa: N802
        """Boundary basin entropy, in Daza's casing — the same number as :attr:`sbb`."""
        return float(self.sbb)

    def _answer(self) -> str:
        """Return the two entropies."""
        return f"Sb = {_sig(self.sb, 4)} · Sbb = {_sig(self.sbb, 4)}"

    def _interpretation(self) -> str | None:
        r"""Report the sufficient fractal-boundary criterion, and only that.

        Daza et al. (2016) prove :math:`S_{bb} > \ln 2` is *sufficient* for a
        fractal boundary; it is not necessary, so failing it means "not
        established by this test", never "smooth".
        """
        return "fractal boundary (Sbb > ln 2)" if self.fractal_boundary else None

    def _printed_names(self) -> dict[str, str]:
        """Map ``Sb`` / ``Sbb`` — what the repr typesets — to ``sb`` / ``sbb``."""
        return {"Sb": "sb", "Sbb": "sbb"}

    def _details(self) -> tuple[str, ...]:
        """Return the box partition the entropies were computed over."""
        return (
            f"({self.n_boundary_boxes}/{self.n_boxes} boxes on the boundary, "
            f"box size {self.box_size}, log base {_sig(self.log_base, 4)})",
        )

    def _derived(self) -> dict[str, Any]:
        """Export the answers under the NAMES THE REPR PRINTS, as well as the fields.

        The repr says ``Sb`` and ``Sbb``, the fields are ``sb`` and ``sbb``, and
        ``to_dict()["Sbb"]`` was a ``KeyError`` for a word the library had just
        shown the reader.  Everywhere else in this library a wrong guess is
        translated; a name printed at the user is not even a guess.
        """
        return {"Sb": float(self.sb), "Sbb": float(self.sbb)}


@dataclass(frozen=True)
class UncertaintyExponent(AnalysisResult):
    r"""
    The uncertainty exponent of a basin boundary.

    Attributes
    ----------
    alpha : float
        Uncertainty exponent :math:`\alpha` (slope of :math:`\log f` vs
        :math:`\log\varepsilon`); ``0`` = boundary fills the space (maximally
        unpredictable), ``1`` = smooth boundary.
    boundary_dimension : float
        Box-counting dimension of the boundary, :math:`D_0 = D - \alpha`.
    state_dimension : int
        State-space (grid) dimension :math:`D`.
    epsilons : ndarray
        Perturbation radii used (in state-space units).
    uncertain_fractions : ndarray
        Fraction of :math:`\varepsilon`-uncertain cells at each radius — the
        measured curve :math:`f(\varepsilon)` whose slope is :attr:`alpha`.
        Pairs with :attr:`epsilons`, same length, same order.
    r_squared : float
        Coefficient of determination of the log-log fit.

    .. versionchanged:: 6.0
        The curve was called ``f``.  A one-letter field on a public result reads
        as a throwaway local, not as the measurement the exponent is read from,
        and it sat next to ``epsilons`` — its own abscissa — spelled in full.
        ``result.f`` now raises, naming :attr:`uncertain_fractions`.
        ``slope_drift`` is still readable but off ``dir()``: it is the
        intermediate behind the public :attr:`resolved` verdict.
    """

    #: The readable form of ``slope_drift`` is the :attr:`resolved` verdict,
    #: which is what the repr prints and what a caller acts on.
    _HIDDEN_ATTRIBUTES: ClassVar[frozenset[str]] = frozenset({"slope_drift"})

    #: The repr and ``to_dict`` both write Grebogi's ``D0`` for the boundary
    #: dimension, so it must resolve and be listed — it did neither.
    _extra_attribute_names: ClassVar[tuple[str, ...]] = ("D0",)

    alpha: float = 0.0
    boundary_dimension: float = 0.0
    state_dimension: int = 0
    epsilons: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    uncertain_fractions: np.ndarray = field(
        default_factory=lambda: np.empty(0), repr=False, compare=False
    )
    r_squared: float = 0.0

    @property
    def D0(self) -> float:  # noqa: N802
        """Boundary dimension in Grebogi's casing — the same number as :attr:`boundary_dimension`."""
        return float(self.boundary_dimension)

    def __plot_spec__(self, kind: str | None = None) -> Any:
        r"""Describe the uncertainty exponent as its log--log scaling fit.

        The uncertainty exponent *is* a scaling estimate:
        :math:`f(\varepsilon)\sim\varepsilon^{\alpha}` (Grebogi et al., 1983), so
        the natural figure is the ``SCALING_FIT`` of :math:`\log f` against
        :math:`\log\varepsilon` with the fitted slope :math:`\alpha`.  This builds
        that spec directly — a ``SCATTER`` of the curve, the fit region marked,
        and the fit line drawn from the slope :math:`\alpha` and an intercept
        recovered from the curve mean — so a single ``result.plot()``
        renders it like every other dimension / Lyapunov-from-data scaling result.
        The :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never
        pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.  ``None`` uses ``SCALING_FIT``; the
            ``scaling_fit`` transform passes ``"scaling_fit"`` explicitly, which
            resolves to the same kind.

        Returns
        -------
        PlotSpec
        """
        from .. import _plotbuilder as pb

        eps = np.asarray(self.epsilons, dtype=float)
        f = np.asarray(self.uncertain_fractions, dtype=float)
        positive = (eps > 0.0) & (f > 0.0)
        log_eps = np.log(eps[positive])
        log_f = np.log(f[positive])

        layers = [pb.scatter(log_eps, log_f, label="curve")]
        if log_eps.size:
            # The fitted line: slope alpha, intercept recovered so it passes
            # through the curve's centroid (log_f ≈ intercept + alpha * log_eps).
            intercept = float(np.mean(log_f) - self.alpha * np.mean(log_eps))
            fit_x = np.array([log_eps.min(), log_eps.max()], dtype=float)
            layers.append(
                pb.line(fit_x, intercept + self.alpha * fit_x, label=f"slope = {self.alpha:.3g}")
            )
        return pb.spec(
            kind,
            "scaling_fit",
            layers=layers,
            xlabel=r"$\log\varepsilon$",
            ylabel=r"$\log f$",
            title=type(self).__name__,
            meta=self.meta,
        )

    def _answer(self) -> str:
        """Return the exponent and the boundary dimension it implies."""
        return (
            f"α = {_sig(self.alpha, 4)} · D0 = {_sig(self.boundary_dimension, 4)} "
            f"in a {int(self.state_dimension)}-D state space"
        )

    @property
    def applicable(self) -> bool:
        r"""Whether enough radii were measured for a power law to be read off.

        ``False`` below :data:`_MIN_UNCERTAINTY_RADII` radii: :math:`R^2` clears
        any acceptance level trivially on two or three points, so the verdict was
        gated on a diagnostic that could not fail — measured, a two-radius run
        printed ``final-state sensitive (fractal boundary)``.

        Returns
        -------
        bool
        """
        return int(np.asarray(self.epsilons).size) >= _MIN_UNCERTAINTY_RADII

    @property
    def slope_drift(self) -> float:
        r"""Ratio of the **last** local slope to the **first**, across the window.

        A power law has one slope everywhere, so this is ``1`` for a converged
        measurement and falls towards ``0`` as the uncertain fraction saturates
        against its ceiling of 1.  It is the diagnostic :math:`R^2` cannot be:
        :math:`R^2` says how straight the fitted line is, not whether the
        relation is a power law at all (measured, ``R² = 0.9926`` on a fit whose
        local slope had already halved).

        Returns
        -------
        float
            ``nan`` when there are fewer than three usable radii, or when a local
            slope is non-positive (the fraction did not grow with the radius).
        """
        eps = np.asarray(self.epsilons, dtype=float)
        frac = np.asarray(self.uncertain_fractions, dtype=float)
        keep = (eps > 0.0) & (frac > 0.0)
        eps, frac = eps[keep], frac[keep]
        if eps.size < 3:
            return float("nan")
        slopes = np.diff(np.log(frac)) / np.diff(np.log(eps))
        if slopes.size < 2 or slopes[0] <= 0.0 or slopes[-1] <= 0.0:
            return float("nan")
        return float(slopes[-1] / slopes[0])

    @property
    def resolved(self) -> bool:
        r"""Whether the power law is converged enough for :math:`\alpha` to be read.

        ``False`` when the local slope decays by more than
        :data:`_SLOPE_STABLE_RATIO` across the probed radii — the uncertain
        fraction has saturated, so the fitted :math:`\alpha` is a **lower bound**
        set by the grid rather than a measurement of the boundary.
        """
        drift = self.slope_drift
        return bool(np.isfinite(drift) and drift >= _SLOPE_STABLE_RATIO)

    @property
    def final_state_sensitive(self) -> bool | None:
        r"""Whether :math:`\alpha \ll 1` — a fractal basin boundary.

        The one adjective-named spelling of the verdict (contract §4.2 rule 10).
        ``None`` — never ``False`` — when the test does not apply or the power
        law itself did not fit.

        Returns
        -------
        bool or None
        """
        if not self.applicable or not self.resolved:
            return None
        if not np.isfinite(self.r_squared) or self.r_squared < _FIT_ACCEPTABLE_R2:
            return None
        return bool(self.alpha <= _FINAL_STATE_SENSITIVE_ALPHA)

    def _interpretation(self) -> str | None:
        r"""Say what the exponent means for predictability.

        Grebogi et al. (1983): the uncertain fraction scales as
        :math:`f \sim \varepsilon^{\alpha}`, so :math:`\alpha \ll 1` means halving
        the uncertainty in the initial condition barely reduces the chance of
        predicting the wrong attractor — *final-state sensitivity*.  The verdict
        is withheld when the power law itself did not fit, and when there were
        too few radii for the fit to be a measurement at all.
        """
        n = int(np.asarray(self.epsilons).size)
        if not self.applicable:
            return f"not applicable — {n} radii is too few to read a power law"
        if not self.resolved:
            # The verdict used to be a property of the GRID, not of the system:
            # the same smooth boundary read "fractal" at 40 cells a side and
            # "smooth" at 60, with nothing saying it had changed its mind.
            return (
                f"inconclusive at this resolution — α ≥ {_sig(self.alpha, 3)} is a "
                "LOWER bound (the uncertain fraction has saturated); refine the grid"
            )
        verdict = self.final_state_sensitive
        if verdict is None:
            return "no clean power law (R² below the acceptance level)"
        if verdict:
            return "final-state sensitive (fractal boundary)"
        if self.contradicted_by_basin_entropy:
            # One picture, two tests, opposite answers.  Daza's is sufficient,
            # this one is a fit, so the fit is the one that has to give ground —
            # and either way a reader must not be handed "smooth" alone.
            sbb = self.meta.get("basin_entropy_sbb") if self.meta else None
            number = f"Sbb = {_sig(float(sbb), 4)}" if sbb is not None else "Sbb"
            return (
                f"DISPUTED — α ≈ 1 reads smooth, but basin_entropy on this same image "
                f"has {number} > ln 2, which is SUFFICIENT for a fractal boundary; "
                f"believe the sufficient test and refine the grid"
            )
        return "smooth boundary (α ≈ 1)"

    @property
    def contradicted_by_basin_entropy(self) -> bool:
        r"""Whether :func:`basin_entropy` proves fractal where this fit reads smooth.

        Only one direction is a contradiction.  :math:`S_{bb} > \ln 2` is
        *sufficient* for a fractal boundary and not necessary (Daza et al.
        2016), so failing it says "not established", never "smooth" — but
        *passing* it while :math:`\alpha \approx 1` is two answers to one
        question about one image.
        """
        if self.final_state_sensitive is not False or not self.meta:
            return False
        return bool(self.meta.get("basin_entropy_fractal"))

    def _derived(self) -> dict[str, Any]:
        """Export the applicability flags and the verdict the repr reports.

        The repr prints ``D0``; ``boundary_dimension`` is the field it comes
        from, so both spellings are exported — a reader who saw ``D0`` in the
        repr should not get a ``KeyError`` for typing it back.
        """
        return {
            "applicable": self.applicable,
            "final_state_sensitive": self.final_state_sensitive,
            "resolved": self.resolved,
            "slope_drift": self.slope_drift,
            "contradicted_by_basin_entropy": self.contradicted_by_basin_entropy,
        }

    def _printed_names(self) -> dict[str, str]:
        """Map ``D0`` — what the repr typesets — to ``boundary_dimension``."""
        return {"D0": "boundary_dimension"}

    def _details(self) -> tuple[str, ...]:
        """Return the fit quality, the radius range, and the refinement line."""
        eps = np.asarray(self.epsilons, dtype=float)
        span = f", ε ∈ [{_sig(eps.min(), 3)}, {_sig(eps.max(), 3)}]" if eps.size else ""
        drift = self.slope_drift
        wobble = f", local slope × {_sig(drift, 2)} across the window" if np.isfinite(drift) else ""
        lines = [f"(R² = {_sig(self.r_squared, 5)}{span}{wobble})"]
        if self.applicable and not self.resolved:
            grid = self.meta.get("grid_shape") if self.meta else None
            finer = (
                " × ".join(str(2 * int(n)) for n in grid)
                if isinstance(grid, (tuple, list)) and grid
                else "a finer grid"
            )
            lines.append(
                f"re-image at {finer} and compare — α drifts upward until the "
                "uncertain fraction is well below 1"
            )
        return tuple(lines)


@dataclass(frozen=True)
class WadaResult(AnalysisResult):
    r"""
    A grid test for the Wada property of a basin diagram.

    Attributes
    ----------
    wada : bool or None
        The verdict: ``True`` when there are at least three basins and the
        fraction of boundary cells seeing *all* basins reaches the acceptance
        threshold at the largest radius (a sufficient grid criterion, not a
        proof).  ``None`` — never ``False`` — when the test does not
        :attr:`apply <applicable>`.
    W : float or None
        The measured Wada fraction at the largest radius.
    applicable : bool
        Whether the test could be run at all (it needs ≥ 3 basins).
    n_basins : int
        Number of attractor basins (colours) considered.
    radii : ndarray
        Chebyshev radii tested.
    fractions : ndarray
        Fraction of boundary cells whose neighbourhood contains every basin, per
        radius (:math:`W` of Daza et al., 2015).
    n_boundary_cells : int
        Number of boundary cells.

    .. versionchanged:: 6.0
        ``is_wada`` (the raw stored flag) and ``threshold`` (the caller's own
        acceptance fraction) are off ``dir()``.  One verdict had **three**
        spellings — ``wada``, ``is_wada`` and ``W`` — and only ``wada`` knows
        about :attr:`applicable`, so ``is_wada`` read ``False`` on an image where
        nothing had been measured.  Both still resolve; ``is_wada`` is in
        ``to_dict()`` and ``W`` in ``to_dict(full=True)``.
    """

    #: C-5 + R1: one verdict, one spelling (:attr:`wada`, the applicable-aware
    #: one), and the acceptance threshold is an input echoed back.
    _HIDDEN_ATTRIBUTES: ClassVar[frozenset[str]] = frozenset({"is_wada", "threshold"})

    is_wada: bool = False
    n_basins: int = 0
    radii: np.ndarray = field(default_factory=lambda: np.empty(0), compare=False)
    fractions: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    n_boundary_cells: int = 0
    threshold: float = 0.0

    @property
    def applicable(self) -> bool:
        """Whether the Wada test could be run on this image at all.

        The Wada property is *defined* as every boundary point being on the
        boundary of **three or more** basins, so the test needs at least three
        basins and at least one boundary cell.  When either is missing,
        :func:`wada_property` early-returns zeros — and a ``W = 0`` reads as a
        *measured negative* ("we looked, the boundaries are not Wada") when the
        truth is that nothing was measured.  This flag separates the two, and
        :meth:`_answer` prints the reason instead of the number.

        Derived from the fields the result already carries, so no estimator
        change: it is exactly the condition ``wada_property`` tests before it
        early-returns.
        """
        return self.n_basins >= _WADA_MIN_BASINS and self.n_boundary_cells > 0

    @property
    def W(self) -> float | None:  # noqa: N802 - W is the published symbol (Daza 2015)
        """The Wada fraction at the largest radius, or ``None`` when inapplicable."""
        if not self.applicable or not self.fractions.size:
            return None
        return float(self.fractions[-1])

    def _answer(self) -> str:
        """Return the Wada fraction, or the reason the test does not apply."""
        if not self.applicable:
            if self.n_basins < _WADA_MIN_BASINS:
                return (
                    f"not applicable — the Wada test needs ≥ {_WADA_MIN_BASINS} basins, "
                    f"this image has {self.n_basins}"
                )
            return "not applicable — this image has no basin boundary cells"
        return f"W = {_sig(self.W, 4)} at radius {int(self.radii[-1])}"

    @property
    def wada(self) -> bool | None:
        """Whether the boundary is Wada — the one adjective-named verdict spelling.

        ``None`` — never ``False`` — when the test does not :attr:`apply
        <applicable>`.  This is **the** spelling: it is the one every other
        classifying result uses (contract §4.2 rule 10), and the only one that
        distinguishes a measured *no* from *nothing was measured*.  The raw
        stored flag ``is_wada`` is still readable (and still exported) but is
        off ``dir()``, because it answers ``False`` in both cases.

        Returns
        -------
        bool or None
        """
        return bool(self.is_wada) if self.applicable else None

    def _interpretation(self) -> str | None:
        """Name the verdict, but only when the test applied."""
        if not self.applicable:
            return None
        return "Wada basins" if self.is_wada else f"not Wada (W < {_sig(self.threshold, 3)})"

    def _details(self) -> tuple[str, ...]:
        """Return the boundary size the fraction was measured over."""
        if not self.applicable:
            return ()
        return (f"({self.n_boundary_cells} boundary cells, {self.n_basins} basins)",)

    def _derived(self) -> dict[str, Any]:
        """Export ``applicable`` and ``W`` — ``W`` is ``None`` when nothing was measured."""
        return {"applicable": self.applicable, "W": self.W, "wada": self.wada}


# ---------------------------------------------------------------------------
# Basin entropy
# ---------------------------------------------------------------------------


def basin_entropy(
    basins: Any, *, box_size: int = 5, base: float = np.e, include_diverged: bool = False
) -> BasinEntropy:
    r"""
    Basin entropy :math:`S_b` and boundary basin entropy :math:`S_{bb}`.

    Partition the basin image into non-overlapping boxes of ``box_size`` cells per
    axis.  Within box :math:`i`, with colour fractions :math:`p_{ij}`, the Gibbs
    entropy is :math:`S_i = -\sum_j p_{ij}\log p_{ij}`.  Then
    :math:`S_b = \langle S_i\rangle` over all boxes and
    :math:`S_{bb} = \langle S_i\rangle` over boundary boxes (more than one
    colour).  :math:`S_{bb} > \log 2` is sufficient for a fractal boundary, since
    a box straddling a smooth boundary holds at most two colours.

    Parameters
    ----------
    basins : BasinsResult or array-like of int
        The basin image (attractor ids ``>= 1``; ``-1`` marks diverged / escape).
    box_size : int, default 5
        Box side length in cells.
    base : float, default e
        Logarithm base.  With the default natural log the fractal threshold is
        :math:`\log 2 \approx 0.693`.
    include_diverged : bool, default False
        If ``False`` (the default), diverged cells (``-1``) are dropped before the
        per-box colour count — escape is not a basin, so it must not inflate the
        entropy as a spurious extra colour.  A box that is *entirely* diverged then
        holds no settled basin and contributes zero entropy, but still counts in
        the box total :math:`N` (Daza et al. (2016) average :math:`S_b` over all
        :math:`N` boxes).  Set ``True`` to count ``-1`` as its own colour (the
        legacy behaviour).

    Returns
    -------
    BasinEntropy

    Raises
    ------
    ValueError
        If ``box_size < 1`` or the label array yields no boxes (it is empty).

    References
    ----------
    A. Daza, A. Wagemakers, B. Georgeot, D. Guéry-Odelin and M. A. F. Sanjuán,
    "Basin entropy: a new tool to analyze uncertainty in dynamical systems",
    *Scientific Reports* **6**, 31416 (2016).
    """
    labels = _as_label_array(basins, analysis="basin_entropy")
    if box_size < 1:
        raise ValueError(f"box_size must be >= 1, got {box_size}")
    log = np.log(base)

    box_entropies: list[float] = []
    n_boundary = 0
    for block in _iter_blocks(labels, box_size):
        flat = block.reshape(-1)
        if flat.size == 0:
            continue  # a zero-area block can only arise from an empty grid edge.
        if not include_diverged:
            flat = flat[flat != -1]  # escape is not a colour
            if flat.size == 0:
                # A fully diverged box holds no settled basin → zero Gibbs entropy.
                # Daza et al. (2016) normalise S_b over *all* N boxes, so an empty
                # box must still count in N (as a zero), not be skipped.
                box_entropies.append(0.0)
                continue
        _, counts = np.unique(flat, return_counts=True)
        p = counts / flat.size
        s = float(-np.sum(p * np.log(p)) / log)
        box_entropies.append(s)
        if counts.size > 1:
            n_boundary += 1

    if not box_entropies:
        raise ValueError("no boxes to analyse (empty label array).")

    entropies = np.asarray(box_entropies)
    sb = float(entropies.mean())
    boundary_mask = entropies > 0.0
    sbb = float(entropies[boundary_mask].mean()) if n_boundary else float("nan")
    threshold = np.log(2.0) / log
    fractal = bool(np.isfinite(sbb) and sbb > threshold)
    return BasinEntropy(
        sb=sb,
        sbb=sbb,
        n_boxes=len(box_entropies),
        n_boundary_boxes=n_boundary,
        box_size=int(box_size),
        log_base=float(base),
        fractal_boundary=fractal,
        meta={"analysis": "basin_entropy", "box_size": int(box_size)},
    )


def _iter_blocks(labels: np.ndarray, box_size: int) -> Iterator[np.ndarray]:
    """Yield every full ``box_size``-cell block of ``labels``.

    Following the Daza et al. (2016) convention, the grid is partitioned into
    *non-overlapping* boxes of a fixed number of cells per axis and trailing
    partial boxes (a grid axis whose length is not a multiple of ``box_size``)
    are discarded — a partial box samples fewer trajectories, giving a noisier,
    biased per-box entropy.  When the grid divides ``box_size`` evenly this is
    byte-identical to keeping every block.
    """
    from itertools import product

    # Stop each axis one full box short of the end so no trailing partial block
    # is produced; ``range(0, n - box_size + 1, box_size)`` yields only origins
    # whose ``box_size`` slice stays in bounds.
    ranges = [range(0, n - box_size + 1, box_size) for n in labels.shape]
    full = (box_size,) * labels.ndim
    for origin in product(*ranges):
        sl = tuple(slice(o, o + box_size) for o in origin)
        block = labels[sl]
        if block.shape == full:  # guard against any residual partial block
            yield block


# ---------------------------------------------------------------------------
# Uncertainty exponent
# ---------------------------------------------------------------------------


def uncertainty_exponent(
    basins: Any,
    *,
    radii: tuple[int, ...] = (1, 2, 3, 4, 5),
    cell_size: float | np.ndarray = 1.0,
    include_diverged: bool = False,
) -> UncertaintyExponent:
    r"""
    Estimate the uncertainty exponent of a basin boundary.

    A cell is :math:`\varepsilon`-uncertain when at least one axis-neighbour at
    distance :math:`\varepsilon` carries a different basin label.  The fraction of
    such cells scales as :math:`f(\varepsilon)\sim\varepsilon^{\alpha}`
    (Grebogi et al., 1983); :math:`\alpha` is the slope of a log-log fit and the
    boundary box-counting dimension is :math:`D_0 = D - \alpha`.

    Parameters
    ----------
    basins : BasinsResult or array-like of int
        The basin image.
    radii : tuple of int, default (1, 2, 3, 4, 5)
        Perturbation radii in cells.
    cell_size : float or array-like, default 1.0
        Physical cell spacing (a scalar or per-axis); rescales ``epsilons`` only —
        the exponent is invariant to it.
    include_diverged : bool, default False
        If ``False`` (the default), diverged cells (``-1``) are excluded: a cell is
        :math:`\varepsilon`-uncertain only when a same-distance neighbour carries a
        *different settled basin*, and the fraction is taken over settled cells.
        Escape is not a basin, so a basin/escape interface is not a final-state
        boundary.  Set ``True`` to treat ``-1`` as just another label.

    Returns
    -------
    UncertaintyExponent

    Raises
    ------
    ValueError
        If fewer than two ``radii`` are given, or fewer than two non-zero
        :math:`f(\varepsilon)` values are available to fit a slope (no boundary).

    References
    ----------
    C. Grebogi, S. W. McDonald, E. Ott and J. A. Yorke, "Final state sensitivity:
    an obstruction to predictability", *Physics Letters A* **99**, 415 (1983).
    """
    labels = _as_label_array(basins, analysis="uncertainty_exponent")
    radii = tuple(int(r) for r in radii)
    if len(radii) < 2:
        raise ValueError("need at least two radii to fit a slope.")

    valid = None if include_diverged else (labels != -1)  # -1 == DIVERGED/escape
    fractions = np.array([_uncertain_fraction(labels, m, valid) for m in radii])
    spacing = float(np.mean(np.atleast_1d(cell_size)))
    epsilons = np.asarray(radii, dtype=float) * spacing

    positive = fractions > 0.0
    if positive.sum() < 2:
        n_basins = len({int(i) for i in np.unique(labels) if int(i) >= 1})
        why = (
            f"this image holds only {n_basins} basin, so it has no boundary between "
            f"basins to be uncertain about"
            if n_basins < 2
            else "the grid is too coarse for any cell to straddle a boundary"
        )
        raise InvalidParameterError(
            f"uncertainty_exponent measures how the uncertain fraction f(eps) shrinks "
            f"with eps, and f(eps) is zero at all but {int(positive.sum())} of the "
            f"probed radii — {why}."
            + remedy(
                "res = ts.analysis.basins(system, [(-2.0, 2.0, 400), (-2.0, 2.0, 400)])",
                "ts.analysis.uncertainty_exponent(res)",
                lead="Image a region that contains more than one attractor, more finely:",
            )
        )

    log_eps = np.log(epsilons[positive])
    log_f = np.log(fractions[positive])
    slope, intercept = np.polyfit(log_eps, log_f, 1)
    fit = slope * log_eps + intercept
    ss_res = float(np.sum((log_f - fit) ** 2))
    ss_tot = float(np.sum((log_f - log_f.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    alpha = float(slope)
    dim = labels.ndim
    # Ask the OTHER boundary test about the SAME image, and carry its answer.
    # Daza et al. (2016) prove ``Sbb > ln 2`` is *sufficient* for a fractal
    # boundary; this estimator fits a power law.  When the proof-style test fires
    # and the fit says "smooth", one of them is wrong about this picture, and a
    # reader running both — which the library invites, they sit side by side in
    # ``find("basin")`` — deserves to be told by whichever they called rather
    # than left to notice.  It is a label-image computation (no integration), so
    # it costs nothing next to the march that produced the image.
    sbb: float | None = None
    try:
        other = basin_entropy(basins, include_diverged=include_diverged)
    except (ValueError, InvalidInputError):  # pragma: no cover - a boundary-free image
        established = False
    else:
        established = bool(other.fractal_boundary)
        sbb = float(other.sbb)
    return UncertaintyExponent(
        alpha=alpha,
        boundary_dimension=float(dim - alpha),
        state_dimension=dim,
        epsilons=epsilons,
        uncertain_fractions=fractions,
        r_squared=float(r2),
        meta={
            "analysis": "uncertainty_exponent",
            "state_dimension": int(dim),
            # What the verdict's resolution hedge names when it asks for a
            # finer image — the user should never have to work out "2n" itself.
            "grid_shape": tuple(int(n) for n in labels.shape),
            "basin_entropy_sbb": sbb,
            "basin_entropy_fractal": established,
        },
    )


def _neighbor_differs(labels: np.ndarray, m: int, *, valid: np.ndarray | None = None) -> np.ndarray:
    """Boolean array: an axis-neighbour at distance ``m`` carries a different label.

    When ``valid`` (a boolean mask) is supplied, a pair only counts as differing if
    *both* cells are valid — so diverged / escape cells (``valid=False``) never
    create a spurious boundary against a settled basin.
    """
    out = np.zeros(labels.shape, dtype=bool)
    nd = labels.ndim
    for axis in range(nd):
        n = labels.shape[axis]
        if m >= n:
            continue
        lo = [slice(None)] * nd
        hi = [slice(None)] * nd
        lo[axis] = slice(0, n - m)
        hi[axis] = slice(m, n)
        lo_t, hi_t = tuple(lo), tuple(hi)
        diff = labels[lo_t] != labels[hi_t]
        if valid is not None:
            diff &= valid[lo_t] & valid[hi_t]
        out[lo_t] |= diff
        out[hi_t] |= diff
    return out


def _uncertain_fraction(labels: np.ndarray, m: int, valid: np.ndarray | None) -> float:
    r"""Fraction of :math:`\varepsilon`-uncertain cells at radius ``m``.

    With ``valid`` given (diverged excluded), the fraction is over *settled* cells
    only; otherwise over every cell (legacy behaviour).
    """
    differs = _neighbor_differs(labels, m, valid=valid)
    if valid is None:
        return float(np.mean(differs))
    if not valid.any():
        return 0.0
    return float(np.mean(differs[valid]))


# ---------------------------------------------------------------------------
# Wada property
# ---------------------------------------------------------------------------


def wada_property(
    basins: Any,
    *,
    radii: tuple[int, ...] = (1, 2, 3, 4, 5),
    threshold: float = 0.9,
    include_diverged: bool = False,
) -> WadaResult:
    r"""
    Test a basin diagram for the Wada property on a grid.

    With three or more basins, a boundary cell is "Wada-complete" at radius
    :math:`r` when its Chebyshev-:math:`r` neighbourhood contains *every* basin
    colour.  The fraction :math:`W(r)` of such boundary cells tends to one for a
    genuine Wada boundary (Daza et al., 2015).  This is a sufficient grid test,
    not a topological proof.

    Parameters
    ----------
    basins : BasinsResult or array-like of int
        The basin image.  Diverged cells (``-1``) are ignored when counting
        colours.
    radii : tuple of int, default (1, 2, 3, 4, 5)
        Chebyshev radii (in cells) at which to grow the neighbourhood.
    threshold : float, default 0.9
        Minimum :math:`W` at the largest radius to call the boundary Wada.
    include_diverged : bool, default False
        If ``False`` (the default), a basin/escape (``-1``) interface is not
        counted as a boundary cell — escape is not a basin.  Set ``True`` to let a
        cell bordering ``-1`` count as boundary.  Wada colours are always the
        settled basins (``>= 1``) regardless.

    Returns
    -------
    WadaResult

    Raises
    ------
    ValueError
        If ``basins`` carries non-integer labels (attractor ids must be integers,
        with ``-1`` marking escape).

    References
    ----------
    A. Daza, A. Wagemakers, M. A. F. Sanjuán and J. A. Yorke, "Testing for basins
    of Wada", *Scientific Reports* **5**, 16579 (2015).
    """
    from scipy.ndimage import maximum_filter

    labels = _as_label_array(basins, analysis="wada_property")
    colors = [int(c) for c in np.unique(labels) if c >= 1]
    radii = tuple(int(r) for r in radii)
    valid = None if include_diverged else (labels != -1)  # -1 == DIVERGED/escape
    boundary = _neighbor_differs(labels, 1, valid=valid)
    n_boundary = int(boundary.sum())

    if len(colors) < 3 or n_boundary == 0:
        return WadaResult(
            is_wada=False,
            n_basins=len(colors),
            radii=np.asarray(radii),
            fractions=np.zeros(len(radii)),
            n_boundary_cells=n_boundary,
            threshold=float(threshold),
            meta={"analysis": "wada_property"},
        )

    presence = {c: (labels == c) for c in colors}
    fractions = []
    for r in radii:
        size = 2 * r + 1
        has_all = np.ones(labels.shape, dtype=bool)
        for c in colors:
            within = maximum_filter(presence[c], size=size, mode="constant", cval=False)
            has_all &= within
        fractions.append(float(has_all[boundary].mean()))

    frac_arr = np.asarray(fractions)
    is_wada = bool(frac_arr[-1] >= threshold)
    return WadaResult(
        is_wada=is_wada,
        n_basins=len(colors),
        radii=np.asarray(radii),
        fractions=frac_arr,
        n_boundary_cells=n_boundary,
        threshold=float(threshold),
        meta={"analysis": "wada_property"},
    )


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


def resilience(result: BasinsResult, attractor_id: int | None = None) -> ScalarResult:
    r"""
    Minimal-fatal-shock resilience of an attractor: its distance to the boundary.

    The smallest perturbation that pushes the attractor out of its own basin,
    estimated as the state-space distance from the attractor's representative to
    the nearest cell of another basin (Halekotte & Feudel, 2020).  Larger means
    more resilient.

    Parameters
    ----------
    result : BasinsResult
        A basin image carrying its grid and attractors.
    attractor_id : int, optional
        Which attractor (basin label) to measure.  A basin image with exactly
        **one** attractor needs no choice, so it may be omitted there; with two
        or more it is required, and the error lists the ids actually present.

    Returns
    -------
    ScalarResult
        Distance from the attractor to its basin boundary (behaves as a
        ``float``), in state-space units.

    Raises
    ------
    TypeError
        If ``result`` is not a :class:`BasinsResult` (the grid + attractors are
        required to measure a state-space distance).
    ValueError
        If the labels are not laid out on the grid (a pre-squeezed slice), or
        ``attractor_id`` is absent from the basin image.

    Notes
    -----
    A *sliced* basin image (a label cube with one or more degenerate ``counts == 1``
    axes, e.g. a position-plane slice of a higher-dimensional flow) is collapsed to
    its free axes before the distance transform, so a pinned axis does not inject a
    spurious one-cell-away boundary that would cap the reported distance.

    References
    ----------
    L. Halekotte and U. Feudel, "Minimal fatal shocks in multistable complex
    networks", *Scientific Reports* **10**, 11374 (2020).  Builds on the integral
    stability of C. Mitra, J. Kurths and R. V. Donner, *Scientific Reports* **5**,
    16196 (2015).
    """
    from scipy.ndimage import distance_transform_edt

    # The shared builder (CONTRACT §5.6) answers the system door; the bespoke
    # branch below still answers the *data* door, because it names the keyword
    # (``attractor_id=``) that no generic message can know about.
    _reject_system(result, analysis="resilience")
    if not isinstance(result, BasinsResult):
        raise InvalidInputError(
            f"resilience measures a state-space distance, so it needs the full "
            f"BasinsResult (with its grid and attractors), not a "
            f"{type(result).__name__}."
            + remedy(
                "res = ts.analysis.basins(system, [(-2.0, 2.0, 200), (-2.0, 2.0, 200)])",
                "ts.analysis.resilience(res, attractor_id=1)",
            )
        )
    labels = result.labels
    attractor_id = _resolve_attractor_id(result, attractor_id)
    grid = result.grid
    assert grid is not None  # a BasinsResult fed to resilience always carries its grid
    if labels.shape != tuple(grid.shape):
        raise ValueError(
            f"resilience needs labels laid out on the grid: labels {labels.shape} vs "
            f"grid {tuple(grid.shape)} (do not pre-squeeze a sliced basin image)."
        )
    mask = labels == int(attractor_id)
    if not mask.any():
        raise ValueError(f"attractor id {attractor_id} is absent from the basin image.")

    counts = np.asarray(grid.shape, dtype=float)
    span = grid.hi - grid.lo
    spacing_full = np.where(counts > 1, span / np.maximum(counts - 1, 1), 1.0)

    # Drop degenerate (``counts == 1``) axes — a pinned slice coordinate of a
    # higher-dimensional flow.  Padding/EDT over such an axis would inject a
    # spurious one-cell-away False border (the axis has only one layer), capping
    # the reported distance.  We mirror ``_as_label_array``: collapse the slice to
    # its effective dimension, keeping only the free axes for the distance field
    # and the matching grid origin / spacing entries.
    free = np.flatnonzero(np.asarray(grid.shape) > 1)
    if free.size == 0:  # fully degenerate grid (a single cell): no boundary at all.
        free = np.arange(labels.ndim)
    mask_free = np.squeeze(mask, axis=tuple(a for a in range(labels.ndim) if a not in free))
    lo_free = grid.lo[free]
    spacing = spacing_full[free]

    # Pad the basin mask with a one-cell False border so the computational-domain
    # edge is itself a boundary.  Without this, a basin that runs to the grid edge
    # has no nearby background there and the EDT reports the (large) distance to
    # the far interior boundary instead — *overestimating* the minimal fatal shock
    # (the domain simply ends; we cannot claim resilience past what was computed).
    padded = np.pad(mask_free, 1, mode="constant", constant_values=False)
    edt = distance_transform_edt(padded, sampling=spacing)
    shape = np.asarray(mask_free.shape)

    def _edt_at(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(distance-to-boundary, clipped cell index) for each state in ``points``."""
        pts_free = np.atleast_2d(np.asarray(points, dtype=float))[:, free]
        idx = np.clip(np.rint((pts_free - lo_free) / spacing).astype(int), 0, shape - 1)
        return edt[tuple((idx + 1).T)], idx

    # Minimal fatal shock = the closest approach of the attractor to its basin
    # boundary, i.e. the MINIMUM distance-to-boundary over the attractor's spatial
    # extent (its sampled point cloud) — an extended attractor (limit cycle /
    # strange set) can graze the boundary far from its single representative.
    # ``by_id``, not ``[]``: since v6 an AttractorSet is a SEQUENCE and ``[]`` is a
    # positional lookup (CONTRACT §4.2 rule 6 / M25), while ``attractor_id`` here is
    # the basin LABEL painted into the image.
    att = result.attractors.by_id(int(attractor_id))
    pts = np.atleast_2d(np.asarray(att.points, dtype=float))
    if pts.size:
        dists, idx = _edt_at(pts)
        on_basin = mask_free[tuple(idx.T)]  # ignore stray points outside the basin
        value = float(np.min(dists[on_basin])) if np.any(on_basin) else float("nan")
    else:
        value = float("nan")
    if not np.isfinite(value):  # empty / off-basin cloud → fall back to the centre
        dist, _ = _edt_at(np.atleast_2d(att.center))
        value = float(dist[0])
    # The answer is a distance read off a CELL GRID, so it is quantised: it is
    # always an exact number of cells, and (because the grid edge and the
    # attractor's own cell both round outward) it is biased slightly high —
    # measured against a direct 72-direction bisection on a two-well oscillator,
    # +7.8% at 40x40, +6.4% at 80x80, +1.4% at 160x160.  Erring high is the
    # dangerous direction for a safety margin, so the readout carries the
    # resolution instead of six significant figures.
    quantum = float(np.max(spacing))
    grid_words = " × ".join(str(int(n)) for n in np.asarray(grid.shape)[free])
    return ScalarResult(
        value=value,
        meta={
            "analysis": "resilience",
            "attractor_id": int(attractor_id),
            "quantization": quantum,
            "quantization_reason": (
                # "1 cells" is the commonest reading of all — a margin one cell
                # wide is exactly when the reader must not be distracted by the
                # grammar of the sentence telling them so.
                f"{value / quantum:.0f} cell{'' if round(value / quantum) == 1 else 's'} "
                f"on a {grid_words} grid — "
                "refine the grid to tighten, and read it as an upper bound"
            ),
        },
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
