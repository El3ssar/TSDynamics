from .chaotic_maps import (
    Henon,
    Ikeda,
    Tinkerbell,
    Gingerbreadman,
    Zaslavskii,
    Chirikov,
)


from .exotic_maps import (
    Bogdanov,
    Svensson,
    Bedhead,
    ZeraouliaSprott,
    GumowskiMira,
    Hopalong,
    Pickover,
    BlinkingVortexMap,
)


from .geometric_maps import (
    Tent,
    Baker,
    Circle,
    Chebyshev,
)


from .polynomial_maps import (
    Gauss,
    DeJong,
    KaplanYorke,
)


from .population_maps import (
    Logistic,
    Ricker,
    MaynardSmith,
)

__all__ = []


# chaotic_maps
__all__ += [
    "Henon",
    "Ikeda",
    "Tinkerbell",
    "Gingerbreadman",
    "Zaslavskii",
    "Chirikov",
]


# exotic_maps
__all__ += [
    "Bogdanov",
    "Svensson",
    "Bedhead",
    "ZeraouliaSprott",
    "GumowskiMira",
    "Hopalong",
    "Pickover",
    "BlinkingVortexMap",
]


# geometric_maps
__all__ += [
    "Tent",
    "Baker",
    "Circle",
    "Chebyshev",
]


# polynomial_maps
__all__ += [
    "Gauss",
    "DeJong",
    "KaplanYorke",
]


# population_maps
__all__ += [
    "Logistic",
    "Ricker",
    "MaynardSmith",
]
