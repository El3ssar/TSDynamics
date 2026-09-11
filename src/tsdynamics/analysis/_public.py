"""The declared name lists behind ``ts.analysis.__all__``.

One tuple per owner, sorted, so several builders adding a name in the same
release merge cleanly instead of colliding inside one literal in
``analysis/__init__.py`` (CONTRACT §9.2).

``__all__`` itself is **computed** — from :data:`tsdynamics.registry.analyses`
plus :data:`VERBS` — never hand-written, so a registered analysis cannot be
missing from the tab surface and an exported name cannot be missing from the
registry.  These tuples carry only the facts a registry cannot: which names are
*not* analyses, and which analyses live in a module the door has to reach into
because their definition site is somebody else's file.
"""

from __future__ import annotations

__all__ = ["AREA_SUBPACKAGES", "PLANAR_ANALYSES", "VERBS"]

#: The three names on ``ts.analysis`` that are not analyses: the search, the
#: registration door, and the module holding the 32 result classes.
VERBS: tuple[str, ...] = ("find", "register", "results")

#: The capability subpackages, which are **unbound** from this namespace after
#: they have imported and self-registered (CONTRACT §5.4): ``ts.analysis.lyapunov``
#: must not answer ``TypeError: 'module' object is not callable``.  They stay
#: importable through :data:`sys.modules`.
AREA_SUBPACKAGES: tuple[str, ...] = (
    "basins",
    "chaos",
    "dimensions",
    "embedding",
    "fixedpoints",
    "lyapunov",
    "orbits",
    "planar",
    "recurrence",
    "sampling",
)

#: The eight field analyses in :mod:`tsdynamics.analysis.planar`, promoted to the
#: public surface in v6 with their registry metadata.  They live here rather than
#: at their definition site only because ``planar.py`` is a single module, not a
#: subpackage that can self-register; everything else about them is ordinary.
#:
#: Seven need a **vector field** — they evaluate or integrate the right-hand side
#: at points that are not in any trajectory — so they declare ``flow`` rather
#: than ``system``: that is what makes ``find(henon)`` answer 14 and not 21.
PLANAR_ANALYSES: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    ("escape_time_field", ("flow",), "fields", "escape leaking transient window region"),
    ("flow_field", ("flow",), "fields", "vector field arrows quiver direction"),
    ("ftle_field", ("flow",), "fields", "lagrangian coherent structures stretching lcs"),
    (
        "invariant_density",
        ("trajectory", "array"),
        "fields",
        "natural measure histogram occupation",
    ),
    ("nullclines", ("flow",), "fields", "nullcline isocline equilibria planar"),
    ("streamlines", ("flow",), "fields", "integral curves trajectories planar"),
    ("trace_determinant", ("flow",), "fields", "linearisation stability classification saddle"),
    ("transient_time_field", ("flow",), "fields", "settling relaxation approach attractor"),
)
