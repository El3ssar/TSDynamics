"""
Tests for continuous ODE systems (DynSys subclasses).

Instantiation tests run for ALL systems (no JiT compilation, fast).
Integration tests run for a representative sample (JiT compilation, marked slow).
"""

import importlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Complete list of (module_path, class_name) for all ODE systems.
# KuramotoSivashinsky is excluded here and tested separately (special constructor).
# ---------------------------------------------------------------------------

_CHAOTIC_ATTRACTORS = "tsdynamics.systems.continuous.chaotic_attractors"
_CHEM_BIO = "tsdynamics.systems.continuous.chem_bio_systems"
_CLIMATE = "tsdynamics.systems.continuous.climate_geophysics"
_COUPLED = "tsdynamics.systems.continuous.coupled_systems"
_EXOTIC = "tsdynamics.systems.continuous.exotic_systems"
_NEURAL = "tsdynamics.systems.continuous.neural_cognitive"
_OSCILLATORY = "tsdynamics.systems.continuous.oscillatory_systems"
_PHYSICAL = "tsdynamics.systems.continuous.physical_systems"
_POPULATION = "tsdynamics.systems.continuous.population_dynamics"

ALL_ODE_SYSTEMS = [
    # ── Chaotic attractors ──────────────────────────────────────────────────
    (_CHAOTIC_ATTRACTORS, "Lorenz"),
    (_CHAOTIC_ATTRACTORS, "LorenzBounded"),
    (_CHAOTIC_ATTRACTORS, "LorenzCoupled"),
    (_CHAOTIC_ATTRACTORS, "Lorenz96"),
    (_CHAOTIC_ATTRACTORS, "Lorenz84"),
    (_CHAOTIC_ATTRACTORS, "Rossler"),
    (_CHAOTIC_ATTRACTORS, "Thomas"),
    (_CHAOTIC_ATTRACTORS, "Halvorsen"),
    (_CHAOTIC_ATTRACTORS, "Chua"),
    (_CHAOTIC_ATTRACTORS, "MultiChua"),
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
    # ── Chemical / biological ───────────────────────────────────────────────
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
    # ── Climate / geophysics ────────────────────────────────────────────────
    (_CLIMATE, "VallisElNino"),
    (_CLIMATE, "RayleighBenard"),
    (_CLIMATE, "Hadley"),
    (_CLIMATE, "DoubleGyre"),
    (_CLIMATE, "BlinkingRotlet"),
    (_CLIMATE, "OscillatingFlow"),
    (_CLIMATE, "BickleyJet"),
    (_CLIMATE, "ArnoldBeltramiChildress"),
    (_CLIMATE, "AtmosphericRegime"),
    (_CLIMATE, "SaltonSea"),
    # ── Coupled systems ─────────────────────────────────────────────────────
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
    # ── Exotic / hyperchaotic ───────────────────────────────────────────────
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
    # ── Neural / cognitive ──────────────────────────────────────────────────
    (_NEURAL, "Hopfield"),
    (_NEURAL, "CellularNeuralNetwork"),
    (_NEURAL, "BeerRNN"),
    # ── Oscillatory ─────────────────────────────────────────────────────────
    (_OSCILLATORY, "ShimizuMorioka"),
    (_OSCILLATORY, "GenesioTesi"),
    (_OSCILLATORY, "MooreSpiegel"),
    (_OSCILLATORY, "AnishchenkoAstakhov"),
    (_OSCILLATORY, "Aizawa"),
    (_OSCILLATORY, "StickSlipOscillator"),
    (_OSCILLATORY, "Torus"),
    (_OSCILLATORY, "Lissajous3D"),
    (_OSCILLATORY, "Lissajous2D"),
    # ── Physical ────────────────────────────────────────────────────────────
    (_PHYSICAL, "DoublePendulum"),
    (_PHYSICAL, "SwingingAtwood"),
    (_PHYSICAL, "Colpitts"),
    (_PHYSICAL, "Laser"),
    (_PHYSICAL, "Blasius"),
    (_PHYSICAL, "FluidTrampoline"),
    (_PHYSICAL, "JerkCircuit"),
    (_PHYSICAL, "InteriorSquirmer"),
    (_PHYSICAL, "WindmiReduced"),
    # ── Population dynamics ─────────────────────────────────────────────────
    (_POPULATION, "CoevolvingPredatorPrey"),
    (_POPULATION, "KawczynskiStrizhak"),
    (_POPULATION, "Finance"),
]

_IDS = [name for _, name in ALL_ODE_SYSTEMS]


# ---------------------------------------------------------------------------
# Instantiation tests — fast, no JiT compilation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_path,class_name", ALL_ODE_SYSTEMS, ids=_IDS)
def test_ode_instantiation(module_path, class_name):
    """Every ODE system must instantiate with default arguments."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    sys = cls()
    assert sys.n_dim is not None and sys.n_dim > 0, (
        f"{class_name}.n_dim must be a positive integer, got {sys.n_dim}"
    )
    assert isinstance(sys.params, dict), f"{class_name}.params must be a dict"
    assert sys.initial_conds is None  # no IC set at class level by default


@pytest.mark.parametrize("module_path,class_name", ALL_ODE_SYSTEMS, ids=_IDS)
def test_ode_params_as_attributes(module_path, class_name):
    """Every param key must be accessible as an instance attribute."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    sys = cls()
    for key in sys.params:
        assert hasattr(sys, key), f"{class_name} missing attribute for param '{key}'"


# ---------------------------------------------------------------------------
# KuramotoSivashinsky — special constructor
# ---------------------------------------------------------------------------

def test_kuramoto_sivashinsky_instantiation():
    from tsdynamics.systems.continuous.chaotic_attractors import KuramotoSivashinsky
    ks = KuramotoSivashinsky(N=8, L=8.0)
    assert ks.n_dim == 8



def test_kuramoto_sivashinsky_raises_small_n():
    from tsdynamics.systems.continuous.chaotic_attractors import KuramotoSivashinsky
    with pytest.raises(ValueError):
        KuramotoSivashinsky(N=4, L=8.0)


# ---------------------------------------------------------------------------
# Integration tests — selected systems, marked slow
# ---------------------------------------------------------------------------

# Systems to integration-test: (module_path, class_name, expected_n_dim)
_INTEGRATION_SAMPLE = [
    (_CHAOTIC_ATTRACTORS, "Lorenz", 3),
    (_CHAOTIC_ATTRACTORS, "Rossler", 3),
    (_CHAOTIC_ATTRACTORS, "Halvorsen", 3),
    (_CHAOTIC_ATTRACTORS, "HyperRossler", 4),
    (_CHAOTIC_ATTRACTORS, "Lorenz96", 20),
    (_CHAOTIC_ATTRACTORS, "SprottA", 3),
    (_CHEM_BIO, "HindmarshRose", 3),
    (_CLIMATE, "RayleighBenard", 3),
    (_COUPLED, "Chen", 3),
    (_EXOTIC, "HyperCai", 4),
    # Hopfield omitted: _rhs uses np.random.randn() + matmul on JiTCODE symbols — uncompilable
    (_OSCILLATORY, "ShimizuMorioka", 3),
    (_PHYSICAL, "DoublePendulum", 4),
    (_POPULATION, "Finance", 3),
]
_INTEG_IDS = [name for _, name, _ in _INTEGRATION_SAMPLE]


@pytest.mark.slow
@pytest.mark.parametrize("module_path,class_name,expected_n_dim", _INTEGRATION_SAMPLE, ids=_INTEG_IDS)
def test_ode_integration_shape_and_finiteness(module_path, class_name, expected_n_dim):
    """Integration returns correct shapes and finite values."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    sys = cls()

    t, X = sys.integrate(dt=0.1, final_time=2.0, rtol=1e-4, atol=1e-6)

    assert t.ndim == 1, "time array must be 1D"
    assert X.ndim == 2, "trajectory must be 2D"
    assert X.shape[0] == t.shape[0], "time and trajectory lengths must match"
    assert X.shape[1] == expected_n_dim, (
        f"{class_name}: expected n_dim={expected_n_dim}, got {X.shape[1]}"
    )
    assert np.all(np.isfinite(X)), f"{class_name}: trajectory contains NaN or Inf"


@pytest.mark.slow
def test_ode_time_starts_at_zero():
    from tsdynamics.systems.continuous.chaotic_attractors import Lorenz
    t, X = Lorenz().integrate(dt=0.1, final_time=1.0)
    assert t[0] == pytest.approx(0.0)


@pytest.mark.slow
def test_ode_steps_overrides_final_time():
    """steps= takes precedence over final_time=."""
    from tsdynamics.systems.continuous.chaotic_attractors import Lorenz
    t1, X1 = Lorenz().integrate(dt=0.1, steps=10)
    t2, X2 = Lorenz().integrate(dt=0.1, steps=10, final_time=9999.0)
    assert X1.shape[0] == X2.shape[0]


@pytest.mark.slow
def test_ode_custom_initial_conditions_stored():
    """Initial conditions passed to integrate() are stored on the object."""
    from tsdynamics.systems.continuous.chaotic_attractors import Lorenz
    ic = [1.0, 1.0, 1.0]
    lor = Lorenz()
    lor.integrate(dt=0.1, final_time=1.0, initial_conds=ic)
    np.testing.assert_array_almost_equal(lor.initial_conds, ic)


@pytest.mark.slow
def test_ode_integration_method_dop853():
    """dop853 integrator produces finite output."""
    from tsdynamics.systems.continuous.chaotic_attractors import Lorenz
    t, X = Lorenz().integrate(dt=0.1, final_time=2.0, method="dop853")
    assert np.all(np.isfinite(X))


@pytest.mark.slow
def test_ode_initial_conds_set_if_none():
    """When no IC is provided, random IC is generated and stored."""
    from tsdynamics.systems.continuous.chaotic_attractors import Rossler
    r = Rossler()
    assert r.initial_conds is None
    r.integrate(dt=0.1, final_time=0.5)
    assert r.initial_conds is not None
    assert r.initial_conds.shape == (3,)


@pytest.mark.slow
def test_hopfield_integration():
    """Hopfield integrates with pre-computed symbolic weight matrix."""
    from tsdynamics.systems.continuous.neural_cognitive import Hopfield
    h = Hopfield()
    t, X = h.integrate(dt=0.1, final_time=1.0)
    assert X.shape == (t.shape[0], 3)
    assert np.all(np.isfinite(X))


@pytest.mark.slow
def test_kuramoto_sivashinsky_integrates():
    """KuramotoSivashinsky integrates without error."""
    from tsdynamics.systems.continuous.chaotic_attractors import KuramotoSivashinsky
    ic = 1e-2 * np.random.default_rng(0).standard_normal(8)
    ks = KuramotoSivashinsky(N=8, L=8.0, initial_conds=ic)
    t, X = ks.integrate(dt=0.1, final_time=2.0)
    assert X.shape == (t.shape[0], 8)
    assert np.all(np.isfinite(X))
