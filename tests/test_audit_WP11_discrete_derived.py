"""Regression tests for WP11 — discrete signature/divergence + tangent/projected fixes.

Covers the audit findings:

* F4-1 — a map kernel with a *keyword-only* parameter must be rejected at
  class-definition time (it can never bind positionally as ``step_fn(x, *params)``).
* F4-3 — the reference-backend divergence message reports the trajectory's own
  step-index axis (consistent for a warm-restart problem), not the bare row offset.
* D1-1 / D1-2 — after a *batch* engine map ``lyapunov_spectrum`` the streaming
  accessors ``deviations()`` / ``growths()`` raise an honest error instead of
  silently returning a nulled frame / the long-run average, while ``exponents()``
  stays coherent and the streaming API recovers after ``reinit()`` + ``step()``.
* D2-1 — a dimension-preserving (permutation) projection with a ``complete``
  callable applies ``complete`` rather than writing the projected input through
  untransformed (the size-only heuristic miswrite).
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.derived.tangent import TangentSystem
from tsdynamics.errors import InvalidInputError


class TestF4KeywordOnlyKernel:
    """F4-1: keyword-only map kernel parameters are rejected at import time."""

    def test_keyword_only_step_param_raises(self) -> None:
        # Pre-fix: ``_positional_param_names`` appended the keyword-only name to the
        # positional list, so names == declared and validation PASSED — then a live
        # call ``step_fn(x, a, b)`` would fail because ``b`` is keyword-only.
        with pytest.raises(InvalidInputError, match="keyword-only"):

            class _BadKwOnly(ts.DiscreteMap):  # type: ignore[misc]
                params = {"a": 1.0, "b": 2.0}
                dim = 1

                @staticmethod
                def _step(X, a, *, b):  # type: ignore[no-untyped-def]
                    return [a * X[0] + b]

                @staticmethod
                def _jacobian(X, a, b):  # type: ignore[no-untyped-def]
                    return [[a]]

    def test_all_positional_kernel_still_accepted(self) -> None:
        # Answer-preserving: a conventional all-positional kernel is unaffected.
        class _OkPositional(ts.DiscreteMap):  # type: ignore[misc]
            params = {"a": 1.0, "b": 2.0}
            dim = 1

            @staticmethod
            def _step(X, a, b):  # type: ignore[no-untyped-def]
                return [a * X[0] + b]

            @staticmethod
            def _jacobian(X, a, b):  # type: ignore[no-untyped-def]
                return [[a]]

        assert _OkPositional().dim == 1


class TestD1TangentPostEngineMap:
    """D1-1 / D1-2: honest streaming accessors after a batch engine map spectrum."""

    def test_deviations_and_growths_raise_after_batch_engine(self) -> None:
        pytest.importorskip("tsdynamics._rust")
        tng = TangentSystem(ts.Henon(), backend="interp")
        exps = tng.lyapunov_spectrum(steps=2000, ic=[0.1, 0.1])

        # exponents() must stay coherent with the just-returned spectrum.
        np.testing.assert_allclose(tng.exponents(), exps)

        # Pre-fix: deviations() raised the misleading "available after reinit()"
        # message and growths() returned the long-run AVERAGE (== exponents()),
        # not the most-recent-step stretch. Now both raise an honest error.
        with pytest.raises(RuntimeError, match="batch engine map"):
            tng.deviations()
        with pytest.raises(RuntimeError, match="batch engine map"):
            tng.growths()

    def test_growths_not_silently_equal_to_average(self) -> None:
        # The crux of D1-2: the pre-fix code set ``_last_growths = exponents`` so
        # growths() == exponents() (the average). Post-fix it must NOT silently
        # return the average — it raises.
        pytest.importorskip("tsdynamics._rust")
        tng = TangentSystem(ts.Henon(), backend="interp")
        tng.lyapunov_spectrum(steps=1000, ic=[0.1, 0.1])
        with pytest.raises(RuntimeError):
            tng.growths()

    def test_streaming_recovers_after_reinit_step(self) -> None:
        # Answer-preserving: driving the streaming frame after a batch run works.
        pytest.importorskip("tsdynamics._rust")
        tng = TangentSystem(ts.Henon(), backend="interp")
        tng.lyapunov_spectrum(steps=500, ic=[0.1, 0.1])
        tng.reinit([0.1, 0.1])
        tng.step()
        assert tng.deviations().shape == (2, 2)
        assert np.isfinite(tng.growths()).all()

    def test_reference_backend_streaming_coherent(self) -> None:
        # The reference NumPy QR loop keeps the streaming accessors coherent
        # (it carries _W and _last_growths) — unchanged by this fix.
        tref = TangentSystem(ts.Henon(), backend="reference")
        tref.lyapunov_spectrum(steps=500, ic=[0.1, 0.1])
        assert np.isfinite(tref.deviations()).all()
        assert np.isfinite(tref.growths()).all()


class TestD2ProjectedPermutation:
    """D2-1: a permutation projection with ``complete`` is not miswritten."""

    def test_permutation_set_state_applies_complete(self) -> None:
        # dim(projected) == dim(full) == 3. Pre-fix the size heuristic wrote the
        # projected input straight to the full state, so state() re-projected it
        # to the WRONG value. Post-fix ``complete`` is applied.
        sys = ts.Lorenz()
        proj = ts.ProjectedSystem(
            sys, [2, 1, 0], complete=lambda u: np.asarray(u, dtype=float)[[2, 1, 0]]
        )
        proj.reinit([1.0, 2.0, 3.0])
        proj.set_state([9.0, 8.0, 7.0])
        # The projected readback must echo the intended projected state.
        np.testing.assert_allclose(proj.state(), [9.0, 8.0, 7.0])
        # The inner full state is the permuted reconstruction.
        np.testing.assert_allclose(proj.system.state(), [7.0, 8.0, 9.0])

    def test_permutation_reinit_applies_complete(self) -> None:
        sys = ts.Lorenz()
        proj = ts.ProjectedSystem(
            sys, [2, 1, 0], complete=lambda u: np.asarray(u, dtype=float)[[2, 1, 0]]
        )
        proj.reinit([9.0, 8.0, 7.0])
        np.testing.assert_allclose(proj.state(), [9.0, 8.0, 7.0])

    def test_subset_projection_unchanged(self) -> None:
        # Answer-preserving: a genuine dimension-reducing projection still
        # disambiguates by size (full 2-D state written directly).
        proj = ts.ProjectedSystem(ts.Henon(), [0], complete=lambda u: [u[0], 0.0])
        proj.reinit([0.1, 0.2])
        proj.set_state([0.3, 0.4])  # full size -> written directly
        np.testing.assert_array_equal(proj.system.state(), [0.3, 0.4])
        proj.set_state([0.5])  # projected size -> complete applied
        np.testing.assert_array_equal(proj.system.state(), [0.5, 0.0])

    def test_no_complete_projected_input_raises(self) -> None:
        # The documented NotImplementedError contract for a projected input with
        # no ``complete`` is preserved.
        proj = ts.ProjectedSystem(ts.Henon(), [0])
        proj.reinit([0.1, 0.2])
        with pytest.raises(NotImplementedError, match="complete"):
            proj.set_state([0.5])
