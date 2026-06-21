"""Tests for the analysis-layer performance-regression harness.

These tests guard the *harness*, not a wall-clock budget: they assert that
``benches/analysis_bench.py`` builds its fixed inputs, runs every case to a
finite time on a valid schema, and that the regression-comparison logic flags a
slowdown beyond tolerance while clearing a noise-level one. The actual timing
gate is advisory and lives in ``.github/workflows/perf-analysis.yml`` — CI
wall-clock numbers are too noisy to assert on here.

The harness is imported by file path (``benches/`` is not an installed package)
so the test runs from a plain checkout.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_BENCH_PATH = pathlib.Path(__file__).resolve().parents[1] / "benches" / "analysis_bench.py"


def _load_bench():
    """Import ``benches/analysis_bench.py`` by path (not an installed module)."""
    spec = importlib.util.spec_from_file_location("analysis_bench", _BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's @dataclass introspection (which reads
    # sys.modules[cls.__module__].__dict__) resolves.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    """The imported benchmark harness module."""
    return _load_bench()


def test_bench_file_is_executable_module(bench):
    """The harness exposes the documented entry points."""
    for name in ("build_inputs", "all_cases", "time_case", "run", "compare", "main"):
        assert hasattr(bench, name), f"analysis_bench missing {name}"


def test_all_cases_are_named_and_unique(bench):
    """Every registered case has a unique, non-empty name."""
    names = [c.name for c in bench.all_cases()]
    assert names, "no benchmark cases registered"
    assert len(names) == len(set(names)), "duplicate case names"
    assert all(isinstance(n, str) and n for n in names)


def test_build_inputs_is_deterministic(bench):
    """Fixed inputs are reproducible: the same series two builds running."""
    a = bench.build_inputs(quick=True)
    b = bench.build_inputs(quick=True)
    # The seeded, fixed-grid Lorenz series must be byte-identical run to run, or
    # the benchmark measures different *work* each time and a "regression" is
    # meaningless.
    assert a.x.shape == b.x.shape
    assert (a.x == b.x).all()
    assert a.x.size > 0


def test_run_quick_produces_valid_schema(bench):
    """A quick run returns the documented JSON-able schema, all times finite."""
    data = bench.run(quick=True, repeat=1)
    assert set(data) == {"meta", "cases"}
    meta = data["meta"]
    for key in ("quick", "repeat", "python", "engine", "platform", "tsdynamics"):
        assert key in meta, f"meta missing {key}"
    assert meta["quick"] is True

    registered = {c.name for c in bench.all_cases()}
    assert set(data["cases"]) == registered, "every case must be measured"
    for name, row in data["cases"].items():
        assert set(row) == {"seconds", "repeat", "n"}, name
        assert row["seconds"] >= 0.0 and row["seconds"] != float("inf"), name
        assert row["repeat"] == 1
        assert isinstance(row["n"], int)


def test_run_case_filter_selects_subset(bench):
    """``only=`` narrows the suite to matching case names."""
    data = bench.run(quick=True, repeat=1, only="rqa")
    assert set(data["cases"]) == {"rqa"}


def test_run_unknown_case_raises(bench):
    """An ``only=`` filter that matches nothing fails loudly."""
    with pytest.raises(SystemExit):
        bench.run(quick=True, repeat=1, only="no-such-analysis")


def test_time_case_returns_minimum(bench):
    """``time_case`` keeps the minimum over repetitions (a non-negative float)."""
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    seconds = bench.time_case(fn, repeat=3)
    assert calls["n"] == 3
    assert isinstance(seconds, float) and seconds >= 0.0


def _result(seconds: dict[str, float]) -> dict:
    return {
        "meta": {},
        "cases": {k: {"seconds": v, "repeat": 1, "n": 0} for k, v in seconds.items()},
    }


def test_compare_flags_regression_beyond_tolerance(bench):
    """A slowdown past tolerance is flagged; within tolerance is not."""
    baseline = _result({"a": 1.0, "b": 1.0})
    current = _result({"a": 2.0, "b": 1.1})  # a doubled, b +10%
    report = bench.compare(baseline, current, tolerance=0.5)
    assert report["a"]["status"] == "regressed"
    assert report["b"]["status"] == "ok"
    assert report["a"]["ratio"] == pytest.approx(2.0)


def test_compare_speedup_is_ok(bench):
    """A faster current run is never flagged."""
    report = bench.compare(_result({"a": 2.0}), _result({"a": 1.0}), tolerance=0.5)
    assert report["a"]["status"] == "ok"
    assert report["a"]["ratio"] == pytest.approx(0.5)


def test_compare_added_and_removed_cases(bench):
    """Cases on only one side are reported, never as regressions."""
    report = bench.compare(_result({"old": 1.0}), _result({"new": 1.0}), tolerance=0.5)
    assert report["old"]["status"] == "removed"
    assert report["new"]["status"] == "added"


def test_compare_zero_baseline_is_infinite_ratio(bench):
    """A zero baseline time yields an infinite ratio (and a flag)."""
    report = bench.compare(_result({"a": 0.0}), _result({"a": 1.0}), tolerance=0.5)
    assert report["a"]["ratio"] == float("inf")
    assert report["a"]["status"] == "regressed"
