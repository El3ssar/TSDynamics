r"""
Recurrence & RQA — stream **A-RQA**.

Recurrence plots (Eckmann, Kamphorst & Ruelle, 1981/1987) and the recurrence
quantification analysis built on them (Zbilut & Webber 1992; Marwan, Romano,
Thiel & Kurths, *Phys. Rep.* **438**, 237, 2007) probe a trajectory through *when
it revisits its own past*:

- :func:`recurrence_matrix` — the binary matrix :math:`R_{ij}=\Theta(\varepsilon
  - \lVert x_i - x_j\rVert)`, thresholded by a fixed :math:`\varepsilon` or a
  target recurrence rate, stored sparse (a k-d tree range search, not the dense
  :math:`N\times N` array).
- :func:`rqa` — the standard scalar measures of its line structure: recurrence
  rate, **determinism** and line entropy / length (diagonal lines), **laminarity**
  and **trapping time** (vertical lines).
- :func:`windowed_rqa` — those measures in a sliding window, so a regime change
  shows up as a step in determinism or laminarity.

Every entry reads a :class:`~tsdynamics.data.Trajectory` or a raw array
interchangeably; a scalar series is accepted directly, or embed it first
(:func:`tsdynamics.analysis.embed`) for phase-space recurrence.  The headline
functions self-register into :data:`tsdynamics.registry.analyses`.
"""

from __future__ import annotations

from ... import registry as _registry
from .matrix import RecurrenceMatrix, recurrence_matrix
from .rqa import RQAResult, rqa
from .windowed import WindowedRQA, windowed_rqa

__all__ = [
    "RQAResult",
    "RecurrenceMatrix",
    "WindowedRQA",
    "recurrence_matrix",
    "rqa",
    "windowed_rqa",
]

# Self-register the headline analyses (D4 / §4e: in-tree analyses register from
# their own subpackage).  Idempotent across re-imports.
for _name, _fn in (
    ("recurrence_matrix", recurrence_matrix),
    ("rqa", rqa),
    ("windowed_rqa", windowed_rqa),
):
    _registry.analyses.register(_name, _fn, needs="trajectory", family="recurrence")
del _name, _fn


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
