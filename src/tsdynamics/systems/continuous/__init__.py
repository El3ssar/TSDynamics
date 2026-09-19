from .chaotic_attractors import (
    Arneodo,
    Chua,
    Coullet,
    Dadras,
    Duffing,
    GenesioTesi,
    GuckenheimerHolmes,
    Halvorsen,
    HenonHeiles,
    HyperLorenz,
    HyperRossler,
    HyperYan,
    HyperYangChen,
    KuramotoSivashinsky,
    Lorenz,
    Lorenz84,
    Lorenz96,
    LorenzBounded,
    LorenzCoupled,
    MultiChua,
    NoseHoover,
    PehlivanWei,
    RabinovichFabrikant,
    RikitakeDynamo,
    Rossler,
    Rucklidge,
    SprottA,
    SprottB,
    SprottC,
    SprottD,
    SprottE,
    SprottF,
    SprottG,
    SprottH,
    SprottI,
    SprottJ,
    SprottJerk,
    SprottK,
    SprottL,
    SprottM,
    SprottMore,
    SprottN,
    SprottO,
    SprottP,
    SprottQ,
    SprottR,
    SprottS,
    SprottTorus,
    Thomas,
    ThomasLabyrinth,
)
from .chem_bio_systems import (
    BelousovZhabotinsky,
    Brusselator,
    CaTwoPlus,
    CaTwoPlusQuasiperiodic,
    CellCycle,
    CircadianRhythm,
    ExcitableCell,
    FitzHughNagumo,
    ForcedBrusselator,
    ForcedFitzHughNagumo,
    ForcedVanDerPol,
    GlycolyticOscillation,
    HastingsPowell,
    HindmarshRose,
    IsothermalChemical,
    ItikBanksTumor,
    Oregonator,
    Selkov,
    TurchinHanski,
    VanDerPol,
)
from .climate_geophysics import (
    ArnoldBeltramiChildress,
    AtmosphericRegime,
    BickleyJet,
    BlinkingRotlet,
    BlinkingVortex,
    DoubleGyre,
    Hadley,
    InteriorSquirmer,
    LidDrivenCavityFlow,
    OscillatingFlow,
    RayleighBenard,
    SaltonSea,
    VallisElNino,
)
from .coupled_systems import (
    Bouali,
    Bouali2,
    BurkeShaw,
    Chen,
    ChenLee,
    DequanLi,
    LiuChen,
    LuChen,
    LuChenCheng,
    PanXuZhou,
    QiChen,
    Sakarya,
    SanUmSrisuchinwong,
    Tsucs2,
    WangSun,
    YuWang,
    YuWang2,
    ZhouChen,
)
from .delayed_systems import (
    IkedaDelay,
    MackeyGlass,
    PiecewiseCircuit,
    ScrollDelay,
    SprottDelay,
    VossDelay,
)
from .exotic_systems import (
    ArnoldWeb,
    BeerRNN,
    CellularNeuralNetwork,
    Hopfield,
    HyperBao,
    HyperCai,
    HyperJha,
    HyperLu,
    HyperPang,
    HyperQi,
    HyperWang,
    HyperXu,
    LorenzStenflo,
    NewtonLiepnik,
    NuclearQuadrupole,
    Qi,
    Robinson,
)
from .oscillatory_systems import (
    Aizawa,
    AnishchenkoAstakhov,
    Lissajous2D,
    Lissajous3D,
    MooreSpiegel,
    ShimizuMorioka,
    StickSlipOscillator,
    StuartLandau,
    Torus,
)
from .physical_systems import (
    Blasius,
    Colpitts,
    DoublePendulum,
    FluidTrampoline,
    JerkCircuit,
    Laser,
    SwingingAtwood,
    WindmiReduced,
)
from .population_dynamics import (
    CoevolvingPredatorPrey,
    Finance,
    KawczynskiStrizhak,
    LotkaVolterra,
    MacArthur,
)
from .spatial_fields import GrayScott, SwiftHohenberg
from .stochastic_systems import DoubleWell, GeometricBrownianMotion, OrnsteinUhlenbeck

__all__ = []


# chaotic_attractors
__all__ += [
    "Lorenz",
    "LorenzBounded",
    "LorenzCoupled",
    "Lorenz96",
    "Lorenz84",
    "Rossler",
    "Thomas",
    "ThomasLabyrinth",
    "Coullet",
    "GenesioTesi",
    "KuramotoSivashinsky",
    "Halvorsen",
    "Chua",
    "MultiChua",
    "Duffing",
    "RabinovichFabrikant",
    "Dadras",
    "PehlivanWei",
    "Arneodo",
    "Rucklidge",
    "HyperRossler",
    "HyperLorenz",
    "HyperYangChen",
    "HyperYan",
    "GuckenheimerHolmes",
    "HenonHeiles",
    "NoseHoover",
    "RikitakeDynamo",
    "SprottTorus",
    "SprottA",
    "SprottB",
    "SprottC",
    "SprottD",
    "SprottE",
    "SprottF",
    "SprottG",
    "SprottH",
    "SprottI",
    "SprottJ",
    "SprottK",
    "SprottL",
    "SprottM",
    "SprottN",
    "SprottO",
    "SprottP",
    "SprottQ",
    "SprottR",
    "SprottS",
    "SprottMore",
    "SprottJerk",
]

# chem_bio_systems
__all__ += [
    "GlycolyticOscillation",
    "Selkov",
    "Oregonator",
    "IsothermalChemical",
    "Brusselator",
    "ForcedBrusselator",
    "CircadianRhythm",
    "CaTwoPlus",
    "CaTwoPlusQuasiperiodic",
    "BelousovZhabotinsky",
    "ExcitableCell",
    "CellCycle",
    "HindmarshRose",
    "VanDerPol",
    "ForcedVanDerPol",
    "FitzHughNagumo",
    "ForcedFitzHughNagumo",
    "TurchinHanski",
    "HastingsPowell",
    "ItikBanksTumor",
]


# climate_geophysics
__all__ += [
    "VallisElNino",
    "RayleighBenard",
    "Hadley",
    "DoubleGyre",
    "BlinkingRotlet",
    "BlinkingVortex",
    "OscillatingFlow",
    "LidDrivenCavityFlow",
    "BickleyJet",
    "InteriorSquirmer",
    "ArnoldBeltramiChildress",
    "AtmosphericRegime",
    "SaltonSea",
]


# coupled_systems
__all__ += [
    "Sakarya",
    "Bouali",
    "Bouali2",
    "LuChenCheng",
    "LuChen",
    "LiuChen",
    "QiChen",
    "ZhouChen",
    "BurkeShaw",
    "Chen",
    "ChenLee",
    "WangSun",
    "YuWang",
    "YuWang2",
    "SanUmSrisuchinwong",
    "DequanLi",
    "PanXuZhou",
    "Tsucs2",
]


# delayed_systems
__all__ += [
    "MackeyGlass",
    "IkedaDelay",
    "SprottDelay",
    "ScrollDelay",
    "PiecewiseCircuit",
    "VossDelay",
]


# exotic_systems
__all__ += [
    "NuclearQuadrupole",
    "HyperCai",
    "HyperBao",
    "HyperJha",
    "HyperQi",
    "HyperXu",
    "HyperWang",
    "HyperPang",
    "HyperLu",
    "LorenzStenflo",
    "Qi",
    "ArnoldWeb",
    "CellularNeuralNetwork",
    "NewtonLiepnik",
    "Robinson",
    "BeerRNN",
    "Hopfield",
]


# oscillatory_systems
__all__ += [
    "StuartLandau",
    "ShimizuMorioka",
    "MooreSpiegel",
    "AnishchenkoAstakhov",
    "Aizawa",
    "StickSlipOscillator",
    "Torus",
    "Lissajous3D",
    "Lissajous2D",
]


# physical_systems
__all__ += [
    "DoublePendulum",
    "SwingingAtwood",
    "Colpitts",
    "Laser",
    "Blasius",
    "FluidTrampoline",
    "JerkCircuit",
    "WindmiReduced",
]


# population_dynamics
__all__ += [
    "LotkaVolterra",
    "CoevolvingPredatorPrey",
    "KawczynskiStrizhak",
    "Finance",
    "MacArthur",
]

# spatial_fields (2-D method-of-lines PDEs — spatial-field movies)
__all__ += ["GrayScott", "SwiftHohenberg"]

# stochastic_systems (diagonal-Itô SDEs)
__all__ += ["OrnsteinUhlenbeck", "GeometricBrownianMotion", "DoubleWell"]


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
