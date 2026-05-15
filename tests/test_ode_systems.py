"""
Tests for continuous ODE systems (``ContinuousSystem`` subclasses).

Instantiation tests run for every system (fast, no JiT compilation).
A representative subset is integration-tested (slow, requires JiTCODE C compile).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# (module_path, class_name) for every built-in ODE system.
# KuramotoSivashinsky has a non-standard constructor and is tested separately.
# ---------------------------------------------------------------------------

_CHAOTIC_ATTRACTORS = "tsdynamics.systems.continuous.chaotic_attractors"
_CHEM_BIO = "tsdynamics.systems.continuous.chem_bio_systems"
_CLIMATE = "tsdynamics.systems.continuous.climate_geophysics"
_COUPLED = "tsdynamics.systems.continuous.coupled_systems"
_EXOTIC = "tsdynamics.systems.continuous.exotic_systems"
_OSCILLATORY = "tsdynamics.systems.continuous.oscillatory_systems"
_PHYSICAL = "tsdynamics.systems.continuous.physical_systems"
_POPULATION = "tsdynamics.systems.continuous.population_dynamics"


ALL_ODE_SYSTEMS: list[tuple[str, str]] = [
    # Chaotic attractors
    (_CHAOTIC_ATTRACTORS, "Lorenz"),
    (_CHAOTIC_ATTRACTORS, "LorenzBounded"),
    (_CHAOTIC_ATTRACTORS, "LorenzCoupled"),
    (_CHAOTIC_ATTRACTORS, "Lorenz84"),
    (_CHAOTIC_ATTRACTORS, "Rossler"),
    (_CHAOTIC_ATTRACTORS, "Thomas"),
    (_CHAOTIC_ATTRACTORS, "Halvorsen"),
    (_CHAOTIC_ATTRACTORS, "Chua"),
    (_CHAOTIC_ATTRACTORS, "Duffing"),
    (_CHAOTIC_ATTRACTORS, "RabinovichFabrikant"),
    (_CHAOTIC_ATTRACTORS, "Dadras"),
    (_CHAOTIC_ATTRACTORS, "PehlivanWei"),
    (_CHAOTIC_ATTRACTORS, "SprottTorus"),
    (_CHAOTIC_ATTRACTORS, "SprottA"),
    (_CHAOTIC_ATTRACTORS, "SprottB"),
    (_CHAOTIC_ATTRACTORS, "SprottC"),
    (_CHAOTIC_ATTRACTORS, "SprottD"),
    (_CHAOTIC_ATTRACTORS, "SprottE"),
    (_CHAOTIC_ATTRACTORS, "SprottF"),
    (_CHAOTIC_ATTRACTORS, "SprottG"),
    (_CHAOTIC_ATTRACTORS, "SprottH"),
    (_CHAOTIC_ATTRACTORS, "SprottI"),
    (_CHAOTIC_ATTRACTORS, "SprottJ"),
    (_CHAOTIC_ATTRACTORS, "SprottK"),
    (_CHAOTIC_ATTRACTORS, "SprottL"),
    (_CHAOTIC_ATTRACTORS, "SprottM"),
    (_CHAOTIC_ATTRACTORS, "SprottN"),
    (_CHAOTIC_ATTRACTORS, "SprottO"),
    (_CHAOTIC_ATTRACTORS, "SprottP"),
    (_CHAOTIC_ATTRACTORS, "SprottQ"),
    (_CHAOTIC_ATTRACTORS, "SprottR"),
    (_CHAOTIC_ATTRACTORS, "SprottS"),
    (_CHAOTIC_ATTRACTORS, "SprottMore"),
    (_CHAOTIC_ATTRACTORS, "SprottJerk"),
    (_CHAOTIC_ATTRACTORS, "Arneodo"),
    (_CHAOTIC_ATTRACTORS, "Rucklidge"),
    (_CHAOTIC_ATTRACTORS, "HyperRossler"),
    (_CHAOTIC_ATTRACTORS, "HyperLorenz"),
    (_CHAOTIC_ATTRACTORS, "HyperYangChen"),
    (_CHAOTIC_ATTRACTORS, "HyperYan"),
    (_CHAOTIC_ATTRACTORS, "GuckenheimerHolmes"),
    (_CHAOTIC_ATTRACTORS, "HenonHeiles"),
    (_CHAOTIC_ATTRACTORS, "NoseHoover"),
    (_CHAOTIC_ATTRACTORS, "RikitakeDynamo"),
    # Chemical / biological
    (_CHEM_BIO, "GlycolyticOscillation"),
    (_CHEM_BIO, "Oregonator"),
    (_CHEM_BIO, "IsothermalChemical"),
    (_CHEM_BIO, "ForcedBrusselator"),
    (_CHEM_BIO, "CircadianRhythm"),
    (_CHEM_BIO, "CaTwoPlus"),
    (_CHEM_BIO, "ExcitableCell"),
    (_CHEM_BIO, "CellCycle"),
    (_CHEM_BIO, "HindmarshRose"),
    (_CHEM_BIO, "ForcedVanDerPol"),
    (_CHEM_BIO, "ForcedFitzHughNagumo"),
    (_CHEM_BIO, "TurchinHanski"),
    (_CHEM_BIO, "HastingsPowell"),
    (_CHEM_BIO, "ItikBanksTumor"),
    # Climate / geophysics
    (_CLIMATE, "VallisElNino"),
    (_CLIMATE, "RayleighBenard"),
    (_CLIMATE, "Hadley"),
    (_CLIMATE, "DoubleGyre"),
    (_CLIMATE, "BlinkingRotlet"),
    (_CLIMATE, "OscillatingFlow"),
    (_CLIMATE, "ArnoldBeltramiChildress"),
    (_CLIMATE, "AtmosphericRegime"),
    (_CLIMATE, "SaltonSea"),
    # Coupled systems
    (_COUPLED, "Sakarya"),
    (_COUPLED, "Bouali2"),
    (_COUPLED, "LuChenCheng"),
    (_COUPLED, "LuChen"),
    (_COUPLED, "QiChen"),
    (_COUPLED, "ZhouChen"),
    (_COUPLED, "BurkeShaw"),
    (_COUPLED, "Chen"),
    (_COUPLED, "ChenLee"),
    (_COUPLED, "WangSun"),
    (_COUPLED, "YuWang"),
    (_COUPLED, "YuWang2"),
    (_COUPLED, "SanUmSrisuchinwong"),
    (_COUPLED, "DequanLi"),
    # Exotic / hyperchaotic
    (_EXOTIC, "NuclearQuadrupole"),
    (_EXOTIC, "HyperCai"),
    (_EXOTIC, "HyperBao"),
    (_EXOTIC, "HyperJha"),
    (_EXOTIC, "HyperQi"),
    (_EXOTIC, "HyperXu"),
    (_EXOTIC, "HyperWang"),
    (_EXOTIC, "HyperPang"),
    (_EXOTIC, "HyperLu"),
    (_EXOTIC, "LorenzStenflo"),
    (_EXOTIC, "Qi"),
    (_EXOTIC, "ArnoldWeb"),
    (_EXOTIC, "NewtonLiepnik"),
    (_EXOTIC, "Robinson"),
    (_EXOTIC, "CellularNeuralNetwork"),
    # Oscillatory
    (_OSCILLATORY, "ShimizuMorioka"),
    (_OSCILLATORY, "MooreSpiegel"),
    (_OSCILLATORY, "AnishchenkoAstakhov"),
    (_OSCILLATORY, "Aizawa"),
    (_OSCILLATORY, "StickSlipOscillator"),
    (_OSCILLATORY, "Torus"),
    (_OSCILLATORY, "Lissajous3D"),
    (_OSCILLATORY, "Lissajous2D"),
    # Physical
    (_PHYSICAL, "DoublePendulum"),
    (_PHYSICAL, "SwingingAtwood"),
    (_PHYSICAL, "Colpitts"),
    (_PHYSICAL, "Laser"),
    (_PHYSICAL, "Blasius"),
    (_PHYSICAL, "FluidTrampoline"),
    (_PHYSICAL, "JerkCircuit"),
    (_PHYSICAL, "WindmiReduced"),
    # Population dynamics
    (_POPULATION, "CoevolvingPredatorPrey"),
    (_POPULATION, "KawczynskiStrizhak"),
    (_POPULATION, "Finance"),
]

_IDS = [name for _, name in ALL_ODE_SYSTEMS]


# ---------------------------------------------------------------------------
# Instantiation tests — fast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_path,class_name", ALL_ODE_SYSTEMS, ids=_IDS)
def test_ode_instantiation(module_path: str, class_name: str) -> None:
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    sys = cls()
    assert sys.dim is not None and sys.dim > 0
    assert len(sys.params) == len(type(sys).params)
    # Default ICs may or may not be set at class level — both cases are valid.


@pytest.mark.parametrize("module_path,class_name", ALL_ODE_SYSTEMS, ids=_IDS)
def test_ode_params_as_attributes(module_path: str, class_name: str) -> None:
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    sys = cls()
    for key in sys.params:
        assert hasattr(sys, key), f"{class_name} missing attribute for param {key!r}"


# ---------------------------------------------------------------------------
# Lorenz96 / KuramotoSivashinsky / MultiChua — variable-dim constructors
# (these were broken before the structural-params fix)
# ---------------------------------------------------------------------------


def test_lorenz96_default_constructor() -> None:
    import tsdynamics as ts

    lor = ts.Lorenz96()
    assert lor.dim == 20
    assert lor.params["N"] == 20
    assert lor.params["f"] == 8.0


def test_lorenz96_dim_follows_n() -> None:
    import tsdynamics as ts

    lor = ts.Lorenz96(N=12)
    assert lor.dim == 12
    assert lor.params["N"] == 12


def test_kuramoto_sivashinsky_default_ic_is_zero_mean() -> None:
    import tsdynamics as ts

    ks = ts.KuramotoSivashinsky(N=8, L=8.0)
    assert ks.dim == 8
    assert ks.ic is not None
    assert ks.ic.shape == (8,)
    assert abs(ks.ic.mean()) < 1e-10


def test_kuramoto_sivashinsky_default_ic_is_reproducible() -> None:
    """Two instances with the same (N, L) must yield byte-identical default ICs."""
    import tsdynamics as ts

    a = ts.KuramotoSivashinsky(N=32, L=22.0).ic
    b = ts.KuramotoSivashinsky(N=32, L=22.0).ic
    np.testing.assert_array_equal(a, b)


def test_kuramoto_sivashinsky_default_ic_has_target_rms() -> None:
    """Broadband IC is normalised to RMS=0.5 (the documented target)."""
    import tsdynamics as ts

    ks = ts.KuramotoSivashinsky(N=64, L=22.0)
    rms = float(np.sqrt(np.mean(ks.ic**2)))
    assert rms == pytest.approx(0.5, rel=1e-9)


def test_kuramoto_sivashinsky_rejects_small_n() -> None:
    import tsdynamics as ts

    with pytest.raises(ValueError):
        ts.KuramotoSivashinsky(N=4)


def test_multichua_dim_follows_n_circuits() -> None:
    import tsdynamics as ts

    mc = ts.MultiChua(n_circuits=4)
    assert mc.dim == 12
    assert mc.params["n_circuits"] == 4


# ---------------------------------------------------------------------------
# Integration tests — slow (JiTCODE compile + ODE solve)
# ---------------------------------------------------------------------------

# Representative integration sample.  Oregonator is excluded — stiff system
# needing very tight tolerances that slow the test suite excessively.
_INTEGRATION_SAMPLE: list[tuple[str, str, int]] = [
    (_CHAOTIC_ATTRACTORS, "Lorenz", 3),
    (_CHAOTIC_ATTRACTORS, "Rossler", 3),
    (_CHAOTIC_ATTRACTORS, "Halvorsen", 3),
    (_CHAOTIC_ATTRACTORS, "HyperRossler", 4),
    (_CHAOTIC_ATTRACTORS, "SprottA", 3),
    (_CHEM_BIO, "HindmarshRose", 3),
    (_CHEM_BIO, "CircadianRhythm", 5),
    (_CHEM_BIO, "ForcedVanDerPol", 3),
    (_CLIMATE, "RayleighBenard", 3),
    (_CLIMATE, "ArnoldBeltramiChildress", 3),
    (_COUPLED, "Chen", 3),
    (_COUPLED, "LuChen", 3),
    (_EXOTIC, "HyperCai", 4),
    (_EXOTIC, "HyperBao", 4),
    (_OSCILLATORY, "ShimizuMorioka", 3),
    (_OSCILLATORY, "Aizawa", 3),
    (_OSCILLATORY, "Torus", 3),
    (_OSCILLATORY, "Lissajous2D", 2),
    (_PHYSICAL, "DoublePendulum", 4),
    (_PHYSICAL, "Colpitts", 3),
    (_PHYSICAL, "Laser", 3),
    (_POPULATION, "Finance", 3),
    (_POPULATION, "CoevolvingPredatorPrey", 3),
]
_INTEG_IDS = [name for _, name, _ in _INTEGRATION_SAMPLE]


@pytest.mark.slow
@pytest.mark.parametrize("module_path,class_name,expected_dim", _INTEGRATION_SAMPLE, ids=_INTEG_IDS)
def test_ode_integration_shape_and_finiteness(
    module_path: str, class_name: str, expected_dim: int
) -> None:
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    sys = cls()
    traj = sys.integrate(final_time=2.0, dt=0.1, rtol=1e-5, atol=1e-7)
    assert traj.t.ndim == 1
    assert traj.y.ndim == 2
    assert traj.y.shape[0] == traj.t.shape[0]
    assert traj.y.shape[1] == expected_dim
    assert np.all(np.isfinite(traj.y))


@pytest.mark.slow
def test_ode_time_starts_at_zero() -> None:
    import tsdynamics as ts

    traj = ts.Lorenz().integrate(final_time=1.0, dt=0.1)
    assert traj.t[0] == pytest.approx(0.0)


@pytest.mark.slow
def test_ode_custom_ic_stored() -> None:
    import tsdynamics as ts

    ic = [1.0, 1.0, 1.0]
    lor = ts.Lorenz()
    lor.integrate(final_time=1.0, dt=0.1, ic=ic)
    np.testing.assert_array_almost_equal(lor.ic, ic)


@pytest.mark.slow
def test_ode_random_ic_stored_when_none_supplied() -> None:
    import tsdynamics as ts

    r = ts.Rossler()
    assert r.ic is None
    r.integrate(final_time=0.5, dt=0.1)
    assert r.ic is not None
    assert r.ic.shape == (3,)


@pytest.mark.slow
def test_ode_dop853_integrator() -> None:
    import tsdynamics as ts

    traj = ts.Lorenz().integrate(final_time=2.0, dt=0.1, method="dop853")
    assert np.all(np.isfinite(traj.y))


@pytest.mark.slow
def test_lorenz96_integrates() -> None:
    """Lorenz-96 with non-default N — was broken before the structural-params fix."""
    import tsdynamics as ts

    lor = ts.Lorenz96(N=10, f=8.0)
    traj = lor.integrate(final_time=2.0, dt=0.1)
    assert traj.y.shape == (traj.t.shape[0], 10)
    assert np.all(np.isfinite(traj.y))


@pytest.mark.slow
def test_kuramoto_sivashinsky_integrates() -> None:
    """KS with default IC — was broken before the structural-params fix."""
    import tsdynamics as ts

    ks = ts.KuramotoSivashinsky(N=8, L=8.0)
    traj = ks.integrate(final_time=2.0, dt=0.1)
    assert traj.y.shape == (traj.t.shape[0], 8)
    assert np.all(np.isfinite(traj.y))


@pytest.mark.slow
@pytest.mark.parametrize("L", [22.0, 30.0, 60.0])
def test_kuramoto_sivashinsky_large_l_is_nontrivial(L: float) -> None:
    """
    Regression test for the "horizontal stripes" bug: with the old default
    IC (``0.01 cos(2π x / L)``), KS at ``L >= 30`` stayed near zero for the
    whole integration window because only the lowest, marginally unstable
    Fourier mode was excited.  The broadband default IC drives the system
    into the chaotic attractor for every ``L`` tested.

    We check (after discarding the transient) that:
      - the per-cell temporal standard deviation is well above floor noise;
      - the trajectory range covers at least ±1 (canonical KS values).
    """
    import tsdynamics as ts

    N = max(64, 4 * int(np.ceil(L)))
    ks = ts.KuramotoSivashinsky(N=N, L=L)
    traj = ks.integrate(final_time=120.0, dt=0.5, rtol=1e-6, atol=1e-9)
    # Drop transient.
    y_post = traj.y[traj.t > 60.0]
    assert np.all(np.isfinite(y_post))
    # Temporal variance per cell — flat = "horizontal stripe" = bug.
    temporal_std = float(np.sqrt(y_post.var(axis=0)).mean())
    assert temporal_std > 0.5, (
        f"L={L}: temporal_std={temporal_std:.3e} suggests near-frozen dynamics "
        f"(horizontal stripes regression). Expected >0.5 on the KS attractor."
    )
    # Amplitude floor — the canonical attractor reaches ~±3.
    assert y_post.max() - y_post.min() > 2.0


@pytest.mark.slow
def test_multichua_integrates() -> None:
    """MultiChua with default n_circuits — was broken before the structural-params fix."""
    import tsdynamics as ts

    mc = ts.MultiChua()
    traj = mc.integrate(final_time=2.0, dt=0.1, ic=0.1 * np.ones(mc.dim))
    assert traj.y.shape == (traj.t.shape[0], 9)
    assert np.all(np.isfinite(traj.y))
