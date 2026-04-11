"""
Fast symbolic-evaluation tests for every ODE system.

For each DynSys subclass: call sys.rhs(y, t) with the JiTCODE symbolic objects.
This exercises the entire _rhs body without triggering C compilation.
Tests run in < 1 s total even for 100+ systems.
"""

import importlib
import warnings

import pytest
from jitcode import t, y

# ---------------------------------------------------------------------------
# Full system list — must stay in sync with test_ode_systems.py ALL_ODE_SYSTEMS
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
    # InteriorSquirmer excluded: list-param × symbolic-phase multiplication is
    # structurally incompatible with JiTCODE; tested separately as xfail.
    (_PHYSICAL, "WindmiReduced"),
    # ── Population dynamics ─────────────────────────────────────────────────
    (_POPULATION, "CoevolvingPredatorPrey"),
    (_POPULATION, "KawczynskiStrizhak"),
    (_POPULATION, "Finance"),
]

_IDS = [name for _, name in ALL_ODE_SYSTEMS]


# ---------------------------------------------------------------------------
# Symbolic rhs evaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_path,class_name", ALL_ODE_SYSTEMS, ids=_IDS)
def test_ode_rhs_returns_correct_length(module_path, class_name):
    """
    sys.rhs(y, t) must return exactly n_dim symbolic expressions.

    No C compilation is triggered — pure symbolic evaluation of _rhs.
    """
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    sys = cls()

    # Hopfield / BeerRNN generate random weights in __init__; rhs() is valid.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sys.rhs(y, t)

    assert hasattr(result, "__len__") or hasattr(result, "__iter__"), (
        f"{class_name}.rhs() must return an iterable, got {type(result)}"
    )
    expr_list = list(result)
    assert len(expr_list) == sys.n_dim, (
        f"{class_name}.rhs() returned {len(expr_list)} expressions, expected {sys.n_dim}"
    )


@pytest.mark.xfail(reason="InteriorSquirmer: list param × symbolic phase is not JiTCODE-compatible")
def test_interior_squirmer_rhs_xfail():
    """InteriorSquirmer._rhs multiplies list params by a symbolic expression."""
    from tsdynamics.systems.continuous.physical_systems import InteriorSquirmer

    sys = InteriorSquirmer()
    sys.rhs(y, t)


# ---------------------------------------------------------------------------
# KuramotoSivashinsky (special constructor — excluded from parametrize list)
# ---------------------------------------------------------------------------


def test_kuramoto_sivashinsky_rhs_symbolic():
    """KuramotoSivashinsky.rhs(y, t) must return N symbolic expressions."""
    from tsdynamics.systems.continuous.chaotic_attractors import KuramotoSivashinsky

    ks = KuramotoSivashinsky(N=8, L=8.0)
    result = ks.rhs(y, t)
    assert len(list(result)) == 8


# ---------------------------------------------------------------------------
# Optional Jacobian: where _jac is defined, verify its shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_path,class_name", ALL_ODE_SYSTEMS, ids=_IDS)
def test_ode_jac_shape_or_not_implemented(module_path, class_name):
    """
    If a Jacobian is defined, jac(y, t) must return an (n_dim × n_dim) structure.
    Systems returning NotImplemented are skipped silently.
    """
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    sys = cls()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sys.jac(y, t)

    if result is NotImplemented or result is None:
        return  # Jacobian not implemented — that's fine

    rows = list(result)
    assert len(rows) == sys.n_dim, (
        f"{class_name}.jac() must have {sys.n_dim} rows, got {len(rows)}"
    )
    for i, row in enumerate(rows):
        cols = list(row)
        assert len(cols) == sys.n_dim, (
            f"{class_name}.jac() row {i} must have {sys.n_dim} entries, got {len(cols)}"
        )
