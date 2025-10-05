from .chaotic_maps import (
    Chirikov,
    FoldedTowel,
    GeneralizedHenon,
    Gingerbreadman,
    Henon,
    Ikeda,
    Tinkerbell,
    Ulam,
    Zaslavskii,
)
from .exotic_maps import (
    Bedhead,
    Bogdanov,
    GumowskiMira,
    Hopalong,
    Pickover,
    Svensson,
    ZeraouliaSprott,
)
from .geometric_maps import (
    Baker,
    Chebyshev,
    Circle,
    Tent,
)
from .polynomial_maps import (
    DeJong,
    Gauss,
    KaplanYorke,
)
from .population_maps import (
    Logistic,
    MaynardSmith,
    Ricker,
)

__all__ = []


# chaotic_maps
__all__ += [
    "Chirikov",
    "FoldedTowel",
    "GeneralizedHenon",
    "Gingerbreadman",
    "Henon",
    "Ikeda",
    "Tinkerbell",
    "Ulam",
    "Zaslavskii",
]


# exotic_maps
__all__ += [
    "Bogdanov",
    "Bedhead",
    "Svensson",
    "GumowskiMira",
    "Hopalong",
    "Pickover",
    "ZeraouliaSprott",
]


# geometric_maps
__all__ += [
    "Baker",
    "Chebyshev",
    "Circle",
    "Tent",
]


# polynomial_maps
__all__ += [
    "DeJong",
    "Gauss",
    "KaplanYorke",
]


# population_maps
__all__ += [
    "Logistic",
    "MaynardSmith",
    "Ricker",
]
