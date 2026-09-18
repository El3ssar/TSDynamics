"""One realistic instance of **every** :class:`AnalysisResult` subclass.

Built by calling the constructors directly with numbers taken from real runs, so
the result-layer gates (``tests/test_result_repr.py``,
``tests/test_result_plain.py``) are fast, deterministic, and independent of the
families/engine layers — a repr contract is about *rendering*, and running 32
analyses to check 32 strings would make the gate slow and couple it to every
estimator's signature.

The one thing this file must never do is drift: ``test_result_repr.py`` walks
``AnalysisResult.__subclasses__()`` and fails if any subclass is missing here, so
a new result class cannot ship without an entry.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from tsdynamics.analysis._result import (
    AnalysisResult,
    ArrayResult,
    CollectionResult,
    CountResult,
    ScalarResult,
    ScalingResult,
)
from tsdynamics.analysis.basins.attractors import Attractor, AttractorSet
from tsdynamics.analysis.basins.basins import BasinFractions, BasinsResult
from tsdynamics.analysis.basins.continuation import ContinuationResult
from tsdynamics.analysis.basins.metrics import BasinEntropy, UncertaintyExponent, WadaResult
from tsdynamics.analysis.chaos.expansion import ExpansionEntropyResult
from tsdynamics.analysis.chaos.gali import GALIResult
from tsdynamics.analysis.chaos.zero_one import ZeroOneResult
from tsdynamics.analysis.dimensions._common import DimensionResult
from tsdynamics.analysis.embedding.delay import MutualInformation
from tsdynamics.analysis.embedding.dimension import EmbeddingDimension
from tsdynamics.analysis.embedding.embed import Embedding
from tsdynamics.analysis.fixedpoints.fixed import FixedPoint, FixedPointSet
from tsdynamics.analysis.fixedpoints.periodic import OrbitSet, PeriodicOrbit
from tsdynamics.analysis.lyapunov import LyapunovSpectrum
from tsdynamics.analysis.lyapunov.from_data import LyapunovFromData
from tsdynamics.analysis.orbits.orbit_diagram import OrbitDiagram
from tsdynamics.analysis.orbits.return_map import ReturnMap
from tsdynamics.analysis.recurrence.matrix import RecurrenceMatrix
from tsdynamics.analysis.recurrence.rqa import RQAResult
from tsdynamics.analysis.recurrence.windowed import WindowedRQA
from tsdynamics.data import Grid


def _rqa(seed: int = 0) -> RQAResult:
    """One RQA readout (also the per-window payload of a :class:`WindowedRQA`)."""
    rng = np.random.default_rng(seed)
    return RQAResult(
        recurrence_rate=0.05,
        determinism=0.794,
        laminarity=0.093,
        avg_diagonal_length=3.12,
        max_diagonal_length=22,
        divergence=1.0 / 22.0,
        diagonal_entropy=1.659,
        trapping_time=2.04,
        max_vertical_length=6,
        size=251,
        epsilon=4.683,
        theiler_window=0,
        min_diagonal=2,
        min_vertical=2,
        diagonal_lengths=rng.integers(1, 20, 50),
        vertical_lengths=rng.integers(1, 8, 50),
        meta={"analysis": "rqa"},
    )


def _attractor_set() -> AttractorSet:
    """Two well-separated attractors, as the two-well Duffing produces."""
    return AttractorSet(
        attractors={
            1: Attractor(id=1, points=np.tile([-1.0, 0.0], (7, 1)), cells=3),
            2: Attractor(id=2, points=np.tile([1.0, 0.0], (7, 1)), cells=3),
        },
        diverged=0,
        seeds=900,
        meta={"system": "DuffingTwoWell", "analysis": "find_attractors"},
    )


def _basins_labels() -> np.ndarray:
    """A 30x30 two-basin label image split down the middle."""
    labels = np.ones((30, 30), dtype=int)
    labels[15:, :] = 2
    return labels


def _cascade_points(values: np.ndarray) -> list[np.ndarray]:
    """A period-1 → 2 → 4 cascade, so the repr has a real cascade to summarise.

    Each column is the recorded orbit at one parameter value, laid out as ``p``
    exactly-repeating branches — which is what ``OrbitDiagram.periods`` clusters
    and what ``bifurcation_points`` reads the transitions off.
    """
    columns = []
    for v in values:
        period = 1 if v < 3.0 else 2 if v < 3.45 else 4
        branches = np.linspace(0.3, 0.9, period)
        columns.append(np.tile(branches, 40 // period)[:40, None])
    return columns


def _scaling_curve() -> tuple[np.ndarray, np.ndarray]:
    """A clean log--log line of slope ~2, the shape every scaling estimator fits."""
    x = np.linspace(-0.43, 1.76, 20)
    return x, 2.0009 * x - 1.2


def build() -> dict[str, AnalysisResult]:
    """Return ``{class name: one instance}`` for every result class."""
    rng = np.random.default_rng(0)
    x, y = _scaling_curve()
    t = np.arange(9.0)
    lorenz: dict[str, Any] = {"system": "Lorenz"}
    henon: dict[str, Any] = {"system": "Henon", "n": 10_000}

    return {
        # -- the five generic wrappers + the base ---------------------------
        "AnalysisResult": AnalysisResult(meta={**lorenz, "analysis": "a_measurement"}),
        "ScalarResult": ScalarResult(
            value=0.4232674, meta={**henon, "analysis": "lyapunov_spectrum"}
        ),
        "CountResult": CountResult(9, meta={"analysis": "optimal_delay", "method": "mi"}),
        "ArrayResult": ArrayResult(values=np.array([1.0, 2.0, 3.0]), meta={"analysis": "a_curve"}),
        "CollectionResult": CollectionResult(
            items=(),
            meta={
                "analysis": "tipping_points",
                "means_none": "no basin annihilates over the sweep",
            },
        ),
        "ScalingResult": ScalingResult(
            estimate=2.0009,
            stderr=0.00827,
            abscissa=x,
            ordinate=y,
            fit_region=(3, 17),
            intercept=-1.2,
            meta={**lorenz, "analysis": "a_slope"},
        ),
        # -- lyapunov -------------------------------------------------------
        "LyapunovSpectrum": LyapunovSpectrum(
            values=np.array([0.9160, 0.000189, -14.583]),
            meta={**lorenz, "analysis": "lyapunov_spectrum"},
        ),
        "LyapunovFromData": LyapunovFromData(
            estimate=0.005536,
            stderr=0.00021,
            abscissa=np.arange(9) * 0.02,
            ordinate=0.005536 * np.arange(9) * 0.02 - 3.0,
            fit_region=(1, 7),
            intercept=-3.0,
            embedding_dim=5,
            delay=40,
            theiler=20,
            n_reference=7,
            method="kantz",
            trusted=True,
            meta={"analysis": "lyapunov_from_data", "method": "kantz"},
        ),
        # -- dimensions -----------------------------------------------------
        "DimensionResult": DimensionResult(
            estimate=2.0009,
            stderr=0.00827,
            abscissa=x,
            ordinate=y,
            fit_region=(3, 17),
            intercept=-1.2,
            kind="correlation",
            q=2.0,
            meta={"analysis": "correlation_dimension"},
        ),
        # -- embedding ------------------------------------------------------
        "Embedding": Embedding(
            values=rng.normal(size=(1983, 3)),
            meta={"analysis": "embed", "dimension": 3, "delay": 9},
        ),
        "MutualInformation": MutualInformation(
            # Decays to a genuine first local minimum at lag 9, then rises: the
            # Fraser--Swinney rule needs that valley to read a delay off, and a
            # mutual information is non-negative, so the curve is built rather
            # than sampled from a formula that would dip below zero.
            values=np.concatenate(
                [
                    np.linspace(1.30, 0.18, 10),
                    np.linspace(0.20, 0.35, 8),
                    np.linspace(0.34, 0.22, 13),
                ]
            ),
            meta={"analysis": "mutual_information", "max_delay": 30},
        ),
        "EmbeddingDimension": EmbeddingDimension(
            dimension=3,
            dims=np.arange(1, 6),
            method="cao",
            delay=9,
            afn_e1=np.array([0.4, 0.8, 0.97, 0.99, 1.0]),
            afn_e2=np.array([0.6, 0.7, 0.8, 0.85, 0.9]),
            meta={"analysis": "cao_dimension"},
        ),
        # -- chaos ----------------------------------------------------------
        "GALIResult": GALIResult(
            k=2,
            times=np.arange(300.0),
            values=np.logspace(0.0, -17.0, 300),
            is_discrete=True,
            meta={**henon, "analysis": "gali"},
        ),
        "ZeroOneResult": ZeroOneResult(
            value=0.998253,
            p=np.cumsum(rng.normal(size=200)),
            q=np.cumsum(rng.normal(size=200)),
            meta={"system": "Logistic", "analysis": "zero_one_test"},
        ),
        "ExpansionEntropyResult": ExpansionEntropyResult(
            estimate=0.4305,
            stderr=0.0106,
            abscissa=t,
            ordinate=0.4305 * t,
            fit_region=(0, 8),
            intercept=0.0,
            n_samples=60,
            n_survivors=47,
            meta={**henon, "analysis": "expansion_entropy"},
        ),
        # -- fixed points ---------------------------------------------------
        "FixedPoint": FixedPoint(
            x=np.array([-1.131354, -0.339406]),
            eigenvalues=np.array([3.2598, -0.09202]),
            stable=False,
            continuous=False,
            meta=dict(henon),
        ),
        "FixedPointSet": FixedPointSet(
            items=(
                FixedPoint(
                    x=np.array([-1.131354, -0.339406]),
                    eigenvalues=np.array([3.2598, -0.09202]),
                    stable=False,
                ),
                FixedPoint(
                    x=np.array([0.631354, 0.189406]),
                    eigenvalues=np.array([1.9237, -0.15595]),
                    stable=False,
                ),
            ),
            meta={**henon, "analysis": "fixed_points"},
        ),
        "PeriodicOrbit": PeriodicOrbit(
            points=np.array([[0.9758, -0.1427], [-0.4758, 0.2927]]),
            period=2,
            multipliers=np.array([3.0101, -0.0299]),
            stable=False,
            continuous=False,
            residual=2.8e-16,
            meta=dict(henon),
        ),
        "OrbitSet": OrbitSet(
            items=(
                PeriodicOrbit(
                    points=np.array([[0.9758, -0.1427], [-0.4758, 0.2927]]),
                    period=2,
                    multipliers=np.array([3.0101, -0.0299]),
                    stable=False,
                ),
            ),
            meta={**henon, "analysis": "periodic_orbits"},
        ),
        # -- orbits ---------------------------------------------------------
        "OrbitDiagram": OrbitDiagram(
            param="r",
            values=np.linspace(2.8, 4.0, 60),
            points=_cascade_points(np.linspace(2.8, 4.0, 60)),
            components=(0,),
            meta={"system": "Logistic", "analysis": "orbit_diagram"},
        ),
        "ReturnMap": ReturnMap(
            current=np.linspace(30.0, 45.0, 26),
            successor=np.linspace(31.0, 44.0, 26),
            values=np.linspace(30.0, 45.0, 27),
            times=np.linspace(0.0, 20.0, 27),
            observable=2,
            kind="max",
            meta={**lorenz, "analysis": "return_map", "variables": ("x", "y", "z")},
        ),
        # -- recurrence -----------------------------------------------------
        "RecurrenceMatrix": RecurrenceMatrix(
            matrix=sp.random(251, 251, density=0.05, format="csr", random_state=0) > 0,
            epsilon=4.683,
            metric="euclidean",
            theiler_window=0,
            meta={"analysis": "recurrence_matrix"},
        ),
        "RQAResult": _rqa(),
        "WindowedRQA": WindowedRQA(
            centers=np.array([100.0, 200.0, 300.0, 400.0]),
            results=(_rqa(1), _rqa(2), _rqa(3), _rqa(4)),
            window=200,
            step=100,
            meta={"analysis": "windowed_rqa"},
        ),
        # -- basins ---------------------------------------------------------
        "Attractor": Attractor(id=1, points=np.tile([-1.0, 0.0], (7, 1)), cells=3),
        "AttractorSet": _attractor_set(),
        "BasinsResult": BasinsResult(
            labels=_basins_labels(),
            grid=Grid(np.array([-2.0, -2.0]), np.array([2.0, 2.0]), (30, 30)),
            attractors=_attractor_set(),
            meta={"system": "DuffingTwoWell", "analysis": "basins_of_attraction"},
        ),
        "BasinFractions": BasinFractions(
            fractions={1: 0.55, 2: 0.45},
            diverged=0.0,
            n=1000,
            attractors=_attractor_set(),
            meta={"system": "DuffingTwoWell", "analysis": "basin_fractions"},
        ),
        "BasinEntropy": BasinEntropy(
            sb=0.4057,
            sbb=0.5617,
            n_boxes=36,
            n_boundary_boxes=26,
            box_size=5,
            log_base=float(np.e),
            fractal_boundary=False,
            meta={"analysis": "basin_entropy"},
        ),
        "UncertaintyExponent": UncertaintyExponent(
            alpha=0.6607,
            boundary_dimension=1.339,
            state_dimension=2,
            epsilons=np.array([0.01, 0.02, 0.04, 0.08]),
            f=np.array([0.10, 0.16, 0.25, 0.40]),
            r_squared=0.968,
            meta={"analysis": "uncertainty_exponent"},
        ),
        "WadaResult": WadaResult(
            is_wada=False,
            n_basins=2,
            radii=np.array([1, 2, 3]),
            fractions=np.zeros(3),
            n_boundary_cells=0,
            threshold=0.95,
            meta={"analysis": "wada_property"},
        ),
        "ContinuationResult": ContinuationResult(
            param="delta",
            values=np.linspace(0.2, 0.4, 3),
            fractions={1: np.array([0.42, 0.45, 0.44]), 2: np.array([0.58, 0.55, 0.56])},
            attractors=[{1: Attractor(id=1, points=np.zeros((2, 2)), cells=1)}] * 3,
            diverged=np.zeros(3),
            meta={"system": "DuffingTwoWell", "analysis": "continuation"},
        ),
    }


def wada_applicable() -> WadaResult:
    """A Wada test that **did** apply — three basins and a real boundary."""
    return WadaResult(
        is_wada=True,
        n_basins=3,
        radii=np.array([1, 2, 3]),
        fractions=np.array([0.70, 0.90, 0.98]),
        n_boundary_cells=612,
        threshold=0.95,
        meta={"system": "NewtonMap", "analysis": "wada_property"},
    )
