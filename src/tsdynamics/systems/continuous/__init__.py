from .chaotic_attractors import (
    Arneodo,
    Chua,
    Dadras,
    Duffing,
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
)
from .chem_bio_systems import (
    CaTwoPlus,
    CellCycle,
    CircadianRhythm,
    ExcitableCell,
    ForcedBrusselator,
    ForcedFitzHughNagumo,
    ForcedVanDerPol,
    GlycolyticOscillation,
    HastingsPowell,
    HindmarshRose,
    IsothermalChemical,
    ItikBanksTumor,
    Oregonator,
    TurchinHanski,
)
from .climate_geophysics import (
    ArnoldBeltramiChildress,
    AtmosphericRegime,
    BickleyJet,
    BlinkingRotlet,
    DoubleGyre,
    Hadley,
    OscillatingFlow,
    RayleighBenard,
    SaltonSea,
    VallisElNino,
)
from .coupled_systems import (
    Bouali2,
    BurkeShaw,
    Chen,
    ChenLee,
    DequanLi,
    LuChen,
    LuChenCheng,
    QiChen,
    Sakarya,
    SanUmSrisuchinwong,
    WangSun,
    YuWang,
    YuWang2,
    ZhouChen,
)
from .delayed_systems import (
    ENSODelay,
    IkedaDelay,
    MackeyGlass,
    PiecewiseCircuit,
    ScrollDelay,
    SprottDelay,
)
from .exotic_systems import (
    ArnoldWeb,
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
from .neural_cognitive import BeerRNN, CellularNeuralNetwork, Hopfield
from .oscillatory_systems import (
    Aizawa,
    AnishchenkoAstakhov,
    GenesioTesi,
    Lissajous2D,
    Lissajous3D,
    MooreSpiegel,
    ShimizuMorioka,
    StickSlipOscillator,
    Torus,
)
from .physical_systems import (
    Blasius,
    Colpitts,
    DoublePendulum,
    FluidTrampoline,
    InteriorSquirmer,
    JerkCircuit,
    Laser,
    SwingingAtwood,
    WindmiReduced,
)
from .population_dynamics import CoevolvingPredatorPrey, Finance, KawczynskiStrizhak

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
    "Oregonator",
    "IsothermalChemical",
    "ForcedBrusselator",
    "CircadianRhythm",
    "CaTwoPlus",
    "ExcitableCell",
    "CellCycle",
    "HindmarshRose",
    "ForcedVanDerPol",
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
    "OscillatingFlow",
    "BickleyJet",
    "ArnoldBeltramiChildress",
    "AtmosphericRegime",
    "SaltonSea",
]


# coupled_systems
__all__ += [
    "Sakarya",
    "Bouali2",
    "LuChenCheng",
    "LuChen",
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
]


# delayed_systems
__all__ += [
    "MackeyGlass",
    "IkedaDelay",
    "SprottDelay",
    "ScrollDelay",
    "PiecewiseCircuit",
    "ENSODelay",
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
    "NewtonLiepnik",
    "Robinson",
]


# neural_cognitive
__all__ += ["Hopfield", "CellularNeuralNetwork", "BeerRNN"]


# oscillatory_systems
__all__ += [
    "ShimizuMorioka",
    "GenesioTesi",
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
    "InteriorSquirmer",
    "WindmiReduced",
]


# population_dynamics
__all__ += ["CoevolvingPredatorPrey", "KawczynskiStrizhak", "Finance"]
