"""Common machinery for derived-system wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["DerivedSystem"]


class DerivedSystem:
    """
    Base for wrappers that present an existing system through a new lens.

    A derived system implements the :class:`~tsdynamics.base.System` protocol
    by delegating to a wrapped system, transforming what "one step" or "the
    state" means (Poincaré crossings, stroboscopic samples, projections...).

    Parameters and metadata are forwarded to the wrapped system, and
    ``with_params`` re-parametrizes the *inner* system and rebuilds the
    wrapper, so parameter sweeps compose: an orbit diagram over a
    ``PoincareMap`` is a bifurcation diagram of the underlying flow.
    """

    def __init__(self, system: Any) -> None:
        self.system = system

    # --- forwarded surface ---

    @property
    def dim(self) -> int:
        return self.system.dim

    @property
    def params(self):
        return self.system.params

    @property
    def meta(self):
        return self.system.meta

    @property
    def variables(self):
        return getattr(type(self.system), "variables", None)

    def with_params(self, **overrides: Any) -> DerivedSystem:
        """Return a new wrapper of the same kind around a re-parametrized copy."""
        return self._rebuild(self.system.with_params(**overrides))

    def _rebuild(self, inner: Any) -> DerivedSystem:
        """Construct a new wrapper of the same kind around ``inner``."""
        raise NotImplementedError

    # --- default protocol delegation (subclasses override what differs) ---

    @property
    def is_discrete(self) -> bool:
        return self.system.is_discrete

    def state(self) -> np.ndarray:
        return self.system.state()

    def set_state(self, u: Any) -> None:
        self.system.set_state(u)

    def time(self) -> float:
        return self.system.time()

    def reinit(self, u: Any | None = None, **kwargs: Any) -> None:
        self.system.reinit(u, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.system!r})"
