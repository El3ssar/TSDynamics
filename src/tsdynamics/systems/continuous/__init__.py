from .chaotic_attractors import (
    Lorenz,
    LorenzBounded,
    LorenzCoupled,
    Lorenz96,
    Lorenz84,
    Rossler,
    Thomas,
    Halvorsen,
    Chua,
    MultiChua,
    Duffing,
    RabinovichFabrikant,
    Dadras,
    PehlivanWei,
    Arneodo,
    Rucklidge,
    HyperRossler,
    HyperLorenz,
    HyperYangChen,
    HyperYan,
    GuckenheimerHolmes,
    HenonHeiles,
    NoseHoover,
    RikitakeDynamo,
    SprottTorus,
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
    SprottK,
    SprottL,
    SprottM,
    SprottN,
    SprottO,
    SprottP,
    SprottQ,
    SprottR,
    SprottS,
    SprottMore,
    SprottJerk
)


from .chem_bio_systems import (
    GlycolyticOscillation,
    Oregonator,
    IsothermalChemical,
    ForcedBrusselator,
    CircadianRhythm,
    CaTwoPlus,
    ExcitableCell,
    CellCycle,
    HindmarshRose,
    ForcedVanDerPol,
    ForcedFitzHughNagumo,
    TurchinHanski,
    HastingsPowell,
    MacArthur,
    ItikBanksTumor,
)


from .climate_geophysics import (
    VallisElNino,
    RayleighBenard,
    Hadley,
    DoubleGyre,
    BlinkingRotlet,
    OscillatingFlow,
    BickleyJet,
    ArnoldBeltramiChildress,
    AtmosphericRegime,
    SaltonSea
)


from .coupled_systems import (
    Sakarya,
    Bouali2,
    LuChenCheng,
    LuChen,
    QiChen,
    ZhouChen,
    BurkeShaw,
    Chen,
    ChenLee,
    WangSun,
    YuWang,
    YuWang2,
    SanUmSrisuchinwong,
    DequanLi
)


from .delayed_systems import (
    MackeyGlass,
    IkedaDelay,
    SprottDelay,
    VossDelay,
    ScrollDelay,
    PiecewiseCircuit,
    ENSODelay
)


from .exotic_systems import (
    NuclearQuadrupole,
    HyperCai,
    HyperBao,
    HyperJha,
    HyperQi,
    HyperXu,
    HyperWang,
    HyperPang,
    HyperLu,
    LorenzStenflo,
    Qi,
    ArnoldWeb,
    NewtonLiepnik,
    Robinson
)


from .neural_cognitive import (
    Hopfield,
    CellularNeuralNetwork,
    BeerRNN
)


from .oscillatory_systems import (
    ShimizuMorioka,
    GenesioTesi,
    MooreSpiegel,
    AnishchenkoAstakhov,
    Aizawa,
    StickSlipOscillator,
    Torus,
    Lissajous3D,
    Lissajous2D
)


from .physical_systems import (
    DoublePendulum,
    SwingingAtwood,
    Colpitts,
    Laser,
    Blasius,
    FluidTrampoline,
    JerkCircuit,
    InteriorSquirmer,
    WindmiReduced
)


from .population_dynamics import (
    CoevolvingPredatorPrey,
    KawczynskiStrizhak,
    Finance
)


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
            "SprottJerk"
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
            "MacArthur",
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
            "SaltonSea"
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
            "DequanLi"
        ]


# delayed_systems
__all__ += [
            "MackeyGlass",
            "IkedaDelay",
            "SprottDelay",
            "VossDelay",
            "ScrollDelay",
            "PiecewiseCircuit",
            "ENSODelay"
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
            "Robinson"
        ]


# neural_cognitive
__all__ += [
            "Hopfield",
            "CellularNeuralNetwork",
            "BeerRNN"
        ]


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
            "Lissajous2D"
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
            "WindmiReduced"
        ]


# population_dynamics
__all__ += [
            "CoevolvingPredatorPrey",
            "KawczynskiStrizhak",
            "Finance"
        ]