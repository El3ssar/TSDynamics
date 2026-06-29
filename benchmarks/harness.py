"""Benchmark harness: task catalogue, robust timing, precision scoring, tables.

This module is library-agnostic. It defines

* :data:`TASKS` — the ordered catalogue of benchmark tasks (integration, the
  Lyapunov family, correlation dimension, bifurcation diagram, basins, fixed
  points, Poincaré section), each tagged with whether it has a *precision*
  comparison and which literature reference it is scored against;
* :func:`best_of` — the best-of-N (minimum) timer, the standard estimator of
  intrinsic cost (the machine can only ever be slower, never faster);
* the :class:`Adapter` protocol every per-library adapter implements; and
* the table emitters (:func:`speed_table`, :func:`precision_table`) plus the
  results-merge helpers used by the orchestrator.

Each per-library *adapter* (in ``benchmarks/adapters/``) knows how to perform
each task it supports for its library, returning a single *comparable estimate*
(a scalar exponent / dimension, or — for the accuracy probe — the ∞-norm
deviation of its final state from the shared reference). The harness times the
call and the orchestrator scores the estimate against the literature reference.
A task an adapter does not support is reported as ``None`` and renders as a blank
cell, exactly as the brief asks.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# --------------------------------------------------------------------------- #
# Task catalogue
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskSpec:
    """A single benchmark task.

    Attributes
    ----------
    key : str
        Stable identifier used as the JSON key and adapter dispatch name.
    title : str
        Human-readable row label.
    family : str
        Coarse grouping (``integration``/``lyapunov``/``dimension``/…).
    precision : bool
        Whether the task has a precision comparison (an estimate scored against
        a reference). Speed-only tasks leave the precision table blank.
    reference_key : str or None
        Key into the config ``references`` block giving the literature value the
        estimate is scored against. ``None`` ⇒ the estimate is already an error
        (e.g. the accuracy probe's deviation) or precision is N/A.
    unit : str
        Unit of the estimate, for the precision table header.
    lower_is_better : bool
        For the accuracy probe the estimate IS the error (smaller is better) and
        there is no separate reference; rendered as a deviation.
    repeat : int
        Best-of-N timing samples (heavy tasks override the default down).
    repeat_quick : int
        Timing samples in ``--quick`` mode.
    """

    key: str
    title: str
    family: str
    precision: bool
    reference_key: str | None
    unit: str = ""
    lower_is_better: bool = False
    repeat: int = 5
    repeat_quick: int = 2


TASKS: list[TaskSpec] = [
    TaskSpec(
        "integrate_short",
        "Integration — short (Lorenz, T=100)",
        "integration",
        precision=False,
        reference_key=None,
    ),
    TaskSpec(
        "integrate_long",
        "Integration — long (Lorenz, T=10000)",
        "integration",
        precision=False,
        reference_key=None,
        repeat=1,
        repeat_quick=1,
    ),
    TaskSpec(
        "integrate_accuracy",
        "Integration accuracy (Lorenz, T=8, ‖Δ‖∞ vs 1e-13 ref)",
        "integration",
        precision=True,
        reference_key=None,
        unit="‖Δ‖∞",
        lower_is_better=True,
    ),
    TaskSpec(
        "lyapunov_spectrum",
        "Lyapunov spectrum (Lorenz, λ_max)",
        "lyapunov",
        precision=True,
        reference_key="lorenz_lambda_max",
        unit="λ_max",
    ),
    TaskSpec(
        "max_lyapunov",
        "Maximal Lyapunov (Hénon map)",
        "lyapunov",
        precision=True,
        reference_key="henon_lambda_max",
        unit="λ_max",
    ),
    TaskSpec(
        "lyapunov_from_data",
        "Maximal Lyapunov from data (Lorenz x(t))",
        "lyapunov",
        precision=True,
        reference_key="lorenz_lambda_max",
        unit="λ_max",
        repeat=3,
    ),
    TaskSpec(
        "correlation_dimension",
        "Correlation dimension (Lorenz x(t), embedded)",
        "dimension",
        precision=True,
        reference_key="lorenz_d2",
        unit="D₂",
        repeat=3,
    ),
    TaskSpec(
        "bifurcation_diagram",
        "Bifurcation diagram (logistic map sweep)",
        "bifurcation",
        precision=False,
        reference_key=None,
        repeat=3,
    ),
    TaskSpec(
        "basins_of_attraction",
        "Basins of attraction (Newton z³−1 map)",
        "basins",
        precision=False,
        reference_key=None,
        repeat=1,
        repeat_quick=1,
    ),
    TaskSpec(
        "fixed_points",
        "Fixed points (Hénon map)",
        "fixedpoints",
        precision=True,
        reference_key="henon_fp_x",
        unit="x*",
    ),
    TaskSpec(
        "poincare_section",
        "Poincaré section (Rössler, y=0)",
        "poincare",
        precision=False,
        reference_key=None,
        repeat=3,
    ),
    # --- from-data complexity / scaling / recurrence (the expanded task set) --- #
    TaskSpec(
        "sample_entropy",
        "Sample entropy (Lorenz x(t))",
        "entropy",
        precision=True,
        reference_key=None,  # no literature constant; cross-library agreement is the check
        unit="SampEn",
        repeat=3,
    ),
    TaskSpec(
        "permutation_entropy",
        "Permutation entropy (Lorenz x(t), normalized)",
        "entropy",
        precision=True,
        reference_key=None,
        unit="PermEn",
        repeat=5,
    ),
    TaskSpec(
        "multiscale_entropy",
        "Multiscale entropy (Lorenz x(t))",
        "entropy",
        precision=True,
        reference_key=None,
        unit="MSE",
        repeat=2,
    ),
    TaskSpec(
        "dfa",
        "Detrended fluctuation analysis (white noise, α=0.5)",
        "scaling",
        precision=True,
        reference_key="dfa_alpha",
        unit="α",
        repeat=5,
    ),
    TaskSpec(
        "hurst",
        "Hurst exponent (white noise, H=0.5)",
        "scaling",
        precision=True,
        reference_key="hurst_exp",
        unit="H",
        repeat=5,
    ),
    TaskSpec(
        "rqa_determinism",
        "RQA determinism (Lorenz x(t))",
        "recurrence",
        precision=True,
        reference_key=None,  # cross-library agreement is the check
        unit="DET",
        repeat=2,
    ),
    TaskSpec(
        "embedding_dimension",
        "Embedding dimension (Lorenz x(t), FNN/Cao)",
        "embedding",
        precision=True,
        reference_key="lorenz_embed_dim",
        unit="m",
        repeat=3,
    ),
    TaskSpec(
        "surrogate_generation",
        "Surrogate generation (IAAFT, Lorenz x(t))",
        "surrogate",
        precision=False,
        reference_key=None,
        repeat=3,
    ),
]

TASK_BY_KEY: dict[str, TaskSpec] = {t.key: t for t in TASKS}


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


def best_of(fn: Any, *, repeat: int) -> float:
    """Time ``fn`` ``repeat`` times and return the minimum elapsed seconds.

    The minimum is the most reproducible estimator of intrinsic cost: scheduler
    noise, GC and frequency scaling can only ever make a sample slower.

    Parameters
    ----------
    fn : callable
        Zero-argument call to time. Its return value is ignored here (the caller
        captures it separately for the precision estimate).
    repeat : int
        Number of repetitions (>= 1).

    Returns
    -------
    float
        Minimum wall-clock seconds over the repetitions.
    """
    best = math.inf
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# --------------------------------------------------------------------------- #
# Adapter protocol & per-task result record
# --------------------------------------------------------------------------- #


@dataclass
class CellResult:
    """One (library, task) measurement."""

    status: str  # "ok" | "unsupported" | "error" | "timeout"
    seconds: float | None = None
    estimate: float | None = None  # the comparable scalar (λ_max, D₂, ‖Δ‖∞, x*)
    note: str | None = None


@runtime_checkable
class Adapter(Protocol):
    """A per-library benchmark adapter.

    An adapter advertises its identity/availability and, for each task it
    supports, returns a zero-argument callable that performs the task and returns
    the comparable estimate (or ``None`` when the task has no precision estimate).
    Returning ``None`` from :meth:`build` marks the task unsupported (a blank
    cell). The callable is timed by the harness; it must do the *whole* unit of
    work each call (no caching of the result across calls).
    """

    name: str
    language: str

    @property
    def available(self) -> bool: ...

    @property
    def version(self) -> str: ...

    def build(self, task_key: str, *, quick: bool) -> Any | None:
        """Return a zero-arg callable for ``task_key`` (``None`` ⇒ unsupported)."""
        ...


# --------------------------------------------------------------------------- #
# Table emission
# --------------------------------------------------------------------------- #


@dataclass
class Merged:
    """The merged results across all libraries, ready for table emission."""

    libraries: list[dict[str, Any]]  # [{name, language, version, available, tasks}]
    references: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


def _fmt_time(seconds: float | None) -> str:
    """Format a wall-clock time with an adaptive unit, or ``—`` when missing."""
    if seconds is None:
        return "—"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _cell(tasks: dict[str, Any], key: str) -> dict[str, Any] | None:
    cell = tasks.get(key)
    if not cell or cell.get("status") != "ok":
        return None
    return cell


def speed_table(merged: Merged) -> str:
    """Render the speed comparison as a Markdown table (rows=tasks, cols=libs)."""
    libs = merged.libraries
    header = "| Task | " + " | ".join(lib["name"] for lib in libs) + " |"
    rule = "|---|" + "|".join("---:" for _ in libs) + "|"
    lines = [header, rule]
    for task in TASKS:
        cells = []
        for lib in libs:
            cell = _cell(lib.get("tasks", {}), task.key)
            cells.append(_fmt_time(cell["seconds"]) if cell else "—")
        lines.append(f"| {task.title} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _score(task: TaskSpec, estimate: float | None, references: dict[str, Any]) -> str:
    """Format a precision cell: the estimate and (parenthesised) error vs ref."""
    if estimate is None:
        return "—"
    if task.lower_is_better:  # the estimate IS the error (accuracy deviation)
        return f"{estimate:.2e}"
    ref = references.get(task.reference_key) if task.reference_key else None
    if ref is None:
        return f"{estimate:.4g}"
    err = abs(estimate - ref)
    return f"{estimate:.4g} (Δ {err:.3g})"


def precision_table(merged: Merged) -> str:
    """Render the precision comparison as a Markdown table (precision tasks only)."""
    libs = merged.libraries
    header = "| Task (reference) | " + " | ".join(lib["name"] for lib in libs) + " |"
    rule = "|---|" + "|".join("---:" for _ in libs) + "|"
    lines = [header, rule]
    for task in TASKS:
        if not task.precision:
            continue
        ref = merged.references.get(task.reference_key) if task.reference_key else None
        if task.lower_is_better:
            label = f"{task.title}"
        elif ref is not None:
            label = f"{task.title} = {ref:.4g}"
        else:
            label = task.title
        cells = []
        for lib in libs:
            cell = _cell(lib.get("tasks", {}), task.key)
            est = cell["estimate"] if cell else None
            cells.append(_score(task, est, merged.references))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def availability_table(merged: Merged) -> str:
    """Render a per-library availability / version / language summary table."""
    lines = [
        "| Library | Language | Version | Status |",
        "|---|---|---|---|",
    ]
    for lib in merged.libraries:
        status = (
            "available" if lib.get("available") else f"**unavailable** — {lib.get('reason', '?')}"
        )
        lines.append(
            f"| {lib['name']} | {lib['language']} | {lib.get('version', '—')} | {status} |"
        )
    return "\n".join(lines)
