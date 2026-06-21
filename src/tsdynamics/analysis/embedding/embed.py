r"""
Time-delay (Takens) embedding.

A scalar measurement :math:`x(t)` of a deterministic system is reconstructed into
a state-space trajectory by the *delay-coordinate map*

.. math::

    y_i = \big(x_i,\; x_{i+\tau},\; x_{i+2\tau},\; \dots,\; x_{i+(m-1)\tau}\big),

with embedding dimension :math:`m` and delay :math:`\tau` (in samples).  Takens'
theorem (Takens, 1981) guarantees that for a generic observable and
:math:`m > 2d` — where :math:`d` is the box-counting dimension of the original
attractor — this map is an embedding: it preserves the attractor's topology, so
invariants such as the correlation dimension and the Lyapunov spectrum are
recovered from the single series.

:func:`embed` builds the :math:`(N - (m-1)\tau)\times m` matrix of delay vectors,
and also performs **multivariate** embedding — stacking delay coordinates of
several synchronous channels into one reconstruction (a per-channel ``dimension``
and ``delay`` are allowed).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._result import ArrayResult
from ._common import _as_channels, _as_series, _is_trajectory

__all__ = ["Embedding", "embed"]


@dataclass(frozen=True, eq=False)
class Embedding(ArrayResult):
    """A delay-coordinate reconstruction — the embedded matrix and its provenance.

    An :class:`~tsdynamics.analysis._result.ArrayResult`, so it is a drop-in for
    the bare ``(N - (m-1)·τ, m)`` matrix: ``np.asarray(result)``, indexing,
    slicing (``result[:, 0]``), ``result.shape`` and iteration all defer to the
    wrapped array, while it also carries ``.meta`` / ``.summary()`` / the ``.plot``
    seam.

    Attributes
    ----------
    values : numpy.ndarray
        The embedded delay-vector matrix.  ``np.asarray(result)`` returns it.
    """


def _as_per_channel(value: int | Sequence[int], n_channels: int, name: str) -> list[int]:
    """Broadcast an int (or validate a per-channel sequence) to a length-``n_channels`` list."""
    if isinstance(value, (int, np.integer)):
        vals = [int(value)] * n_channels
    else:
        vals = [int(v) for v in value]
        if len(vals) != n_channels:
            raise ValueError(f"{name} has {len(vals)} entries but there are {n_channels} channels.")
    return vals


def embed(
    data: Any,
    dimension: int | Sequence[int],
    delay: int | Sequence[int],
    *,
    component: int | str | None = None,
) -> Embedding:
    r"""Time-delay embedding of a scalar series (or a multivariate bundle).

    Parameters
    ----------
    data : array-like or Trajectory
        The source signal.  A 1-D series (or a single selected ``component`` of a
        :class:`~tsdynamics.data.Trajectory` / 2-D array) gives a univariate
        embedding.  Pass a 2-D ``(N, d)`` array, a list of equal-length series, or
        a multi-component trajectory **without** ``component`` to embed every
        channel jointly (multivariate embedding).
    dimension : int or sequence of int
        Embedding dimension :math:`m`.  A single int applies to every channel; a
        per-channel sequence sets each channel's dimension (multivariate only).
        Must be ``>= 1``.
    delay : int or sequence of int
        Delay :math:`\tau` in samples.  A single int applies to every channel; a
        per-channel sequence sets each channel's delay (multivariate only).  Must
        be ``>= 1``.
    component : int or str, optional
        Select a single channel from a multi-component ``data`` for a univariate
        embedding.  When omitted, a multi-component input is embedded across all
        of its channels.

    Returns
    -------
    ndarray, shape (M, sum(dimension))
        The delay-coordinate matrix, one reconstructed state per row, in temporal
        order.  ``M = N - max_c (m_c - 1) * tau_c`` is the number of rows for which
        every channel's full delay window is in range.  The columns are grouped by
        channel: channel ``c`` contributes ``m_c`` consecutive columns
        ``[x_c(i), x_c(i+tau_c), ...]``.

    Raises
    ------
    ValueError
        If ``dimension``/``delay`` are not positive, a per-channel sequence is
        given for a univariate embedding, or the series is too short for the
        requested window.

    Notes
    -----
    The matrix is consumable directly by the point-set analyses — e.g.
    ``correlation_dimension(embed(x, m, tau))`` estimates :math:`D_2` of the
    reconstructed attractor.  Rows are index-ordered with the original sampling,
    so an index-based Theiler window still removes temporally-correlated pairs.

    References
    ----------
    F. Takens, "Detecting strange attractors in turbulence", in *Dynamical
    Systems and Turbulence*, Lecture Notes in Mathematics **898**, 366 (1981).
    """
    # Univariate path: a 1-D series, or an explicitly selected single component.
    univariate = component is not None or _looks_univariate(data)
    if univariate:
        if not isinstance(dimension, (int, np.integer)):
            raise ValueError("a per-channel `dimension` sequence needs a multivariate input.")
        if not isinstance(delay, (int, np.integer)):
            raise ValueError("a per-channel `delay` sequence needs a multivariate input.")
        series = _as_series(data, component=component)
        embedded = _embed_single(series, int(dimension), int(delay))
        return Embedding(values=embedded, meta=_embed_meta(dimension, delay))

    channels = _as_channels(data)
    n_channels = channels.shape[1]
    dims = _as_per_channel(dimension, n_channels, "dimension")
    delays = _as_per_channel(delay, n_channels, "delay")

    n = channels.shape[0]
    spans = [(m - 1) * tau for m, tau in zip(dims, delays, strict=True)]
    _validate(dims, delays, n, max(spans))
    rows = n - max(spans)

    blocks = [_embed_single(channels[:, c], dims[c], delays[c])[:rows] for c in range(n_channels)]
    embedded = np.ascontiguousarray(np.hstack(blocks))
    return Embedding(values=embedded, meta=_embed_meta(dimension, delay))


def _as_json_int(value: int | Sequence[int]) -> int | list[int]:
    """Coerce an int or int sequence to a JSON-friendly int / list of ints."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return [int(v) for v in value]


def _embed_meta(dimension: int | Sequence[int], delay: int | Sequence[int]) -> dict[str, Any]:
    """Build the provenance mapping for an :class:`Embedding` (JSON-friendly)."""
    return {
        "analysis": "embed",
        "dimension": _as_json_int(dimension),
        "delay": _as_json_int(delay),
    }


def _looks_univariate(data: Any) -> bool:
    """Whether ``data`` is a single scalar series (1-D, or a 1-column 2-D / trajectory)."""
    if _is_trajectory(data):
        return data.y.ndim == 1 or data.y.shape[1] == 1
    if isinstance(data, (list, tuple)):
        # A list of 1-D series is multivariate; a flat numeric list is univariate.
        return not (len(data) > 0 and all(np.ndim(c) == 1 for c in data))
    arr = np.asarray(data)
    return arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)


def _validate(dims: Sequence[int], delays: Sequence[int], n: int, max_span: int) -> None:
    if any(m < 1 for m in dims):
        raise ValueError(f"embedding dimension must be >= 1, got {list(dims)}.")
    if any(tau < 1 for tau in delays):
        raise ValueError(f"delay must be >= 1, got {list(delays)}.")
    if max_span >= n:
        raise ValueError(
            f"series of length {n} is too short for the embedding window "
            f"(needs > {max_span} samples); reduce dimension or delay."
        )


def _embed_single(series: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Delay-embed one scalar series into an ``(N-(m-1)tau, m)`` matrix."""
    n = series.size
    span = (m - 1) * tau
    _validate([m], [tau], n, span)
    rows = n - span
    # Column j is the series shifted by j*tau; build without a Python loop over rows.
    out = np.empty((rows, m), dtype=float)
    for j in range(m):
        start = j * tau
        out[:, j] = series[start : start + rows]
    return out


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
