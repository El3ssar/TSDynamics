"""Built-in system catalogue.

Every built-in system class is re-exported flat here, so the canonical path to a
model is ``tsdynamics.systems.<Name>`` (e.g. ``tsdynamics.systems.Lorenz``) — no
need to remember whether it lives under ``continuous`` or ``discrete``. The
per-category submodules (``continuous`` / ``discrete``) remain available for
finer navigation.
"""

from . import continuous, discrete

# Flat re-export of every catalogue class. Adding a system to a category module's
# ``__all__`` automatically surfaces it here (and at ``tsdynamics.<Name>`` via the
# top-level lazy accessor) — no manual edit needed.
for _name in continuous.__all__:
    globals()[_name] = getattr(continuous, _name)
for _name in discrete.__all__:
    globals()[_name] = getattr(discrete, _name)
del _name

_SYSTEM_NAMES: tuple[str, ...] = (*continuous.__all__, *discrete.__all__)

__all__ = ["continuous", "discrete", *_SYSTEM_NAMES]


def __dir__() -> list[str]:
    """Expose the catalogue surface (``__all__``) to ``dir()`` / autocomplete.

    That is the two category subpackages plus every flat-re-exported model class;
    private scan helpers (``_SYSTEM_NAMES``) stay hidden.
    """
    return sorted(__all__)
