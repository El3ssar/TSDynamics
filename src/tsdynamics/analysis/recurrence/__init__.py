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

from .._discovery import register as _register
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

# Self-register the analyses: the definition site is the registration site
# (CONTRACT §7.7), through the public ``ts.analysis.register`` door.
_DATA = ("trajectory", "array")
_register(
    recurrence_matrix,
    subjects=_DATA,
    area="recurrence",
    returns=RecurrenceMatrix,
    keywords="recurrence plot eckmann threshold epsilon",
    cite="Eckmann, Kamphorst & Ruelle (1987), Europhys. Lett. 4, 973",
    doi="10.1209/0295-5075/4/9/004",
)
_register(
    rqa,
    subjects=_DATA,
    area="recurrence",
    returns=RQAResult,
    keywords="recurrence quantification determinism laminarity plot",
    cite="Marwan, Romano, Thiel & Kurths (2007), Phys. Rep. 438, 237",
    doi="10.1016/j.physrep.2006.11.001",
)
_register(
    windowed_rqa,
    subjects=_DATA,
    area="recurrence",
    returns=WindowedRQA,
    keywords="recurrence nonstationarity sliding window regime plot",
    cite="Marwan, Romano, Thiel & Kurths (2007), Phys. Rep. 438, 237",
    doi="10.1016/j.physrep.2006.11.001",
)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
