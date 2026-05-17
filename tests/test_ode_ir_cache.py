"""Regression: SymEngine lowering runs once per ODE `(class, dim, structural hash)` bucket."""

from __future__ import annotations

import pytest

import tsdynamics as ts
import tsdynamics.base.ode_base as ode_base_mod
from tsdynamics.base._ode_lowering import lower_ode_to_ir


def test_lower_ode_to_ir_cached_across_second_integrate(monkeypatch: pytest.MonkeyPatch) -> None:
    captures: dict[str, int] = {"n": 0}
    inner = lower_ode_to_ir

    def tracing_lower(
        cls,
        *,
        dim: int,
        params: dict,
        structural_params: frozenset = frozenset(),
    ):
        captures["n"] += 1
        return inner(cls, dim=dim, params=params, structural_params=structural_params)

    monkeypatch.setattr(ode_base_mod, "lower_ode_to_ir", tracing_lower)
    ode_base_mod.ContinuousSystem._compiled_ode_ir.clear()

    lor = ts.Lorenz()
    lor.integrate(final_time=1.0, dt=0.2, ic=[1.0, 1.0, 1.0], method="DP8")
    lor.integrate(final_time=2.0, dt=0.1, ic=[0.9, 0.9, 0.9], method="DP5")

    assert captures["n"] == 1


def test_ir_cache_key_splits_on_structural_params(monkeypatch: pytest.MonkeyPatch) -> None:
    counts: dict[str, int] = {"n": 0}
    inner = lower_ode_to_ir

    def tracing_lower(cls, *, dim: int, params: dict, structural_params: frozenset = frozenset()):
        counts["n"] += 1
        return inner(cls, dim=dim, params=params, structural_params=structural_params)

    monkeypatch.setattr(ode_base_mod, "lower_ode_to_ir", tracing_lower)

    ode_base_mod.ContinuousSystem._compiled_ode_ir.clear()

    lor_a = ts.Lorenz96(N=8)
    lor_b = ts.Lorenz96(N=9)
    lor_a.integrate(final_time=0.05, dt=0.025, ic=[1.0] * 8, method="DP5")
    lor_b.integrate(final_time=0.05, dt=0.025, ic=[1.0] * 9, method="DP5")

    assert counts["n"] == 2
