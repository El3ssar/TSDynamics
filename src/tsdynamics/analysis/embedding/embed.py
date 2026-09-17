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
from typing import Any, cast

import numpy as np

from ...errors import invalid_value
from .._result import ArrayResult
from ._common import _as_channels, _as_series, _delay_columns, _is_trajectory

__all__ = ["Embedding", "embed"]


@dataclass(frozen=True, eq=False)
class Embedding(ArrayResult):
    """A delay-coordinate reconstruction — the embedded matrix and its provenance.

    An :class:`~tsdynamics.analysis._result.ArrayResult`, so it is a drop-in for
    the bare ``(N - (m-1)·τ, m)`` matrix: ``np.asarray(result)``, indexing,
    slicing (``result[:, 0]``), ``result.shape`` and iteration all defer to the
    wrapped array, while it also carries ``.meta`` / the readout ``repr`` / the ``.plot``
    seam.

    Attributes
    ----------
    values : numpy.ndarray
        The embedded delay-vector matrix.  ``np.asarray(result)`` returns it.
    """

    def _answer(self) -> str:
        """Return ``<N> points in <m>-D`` — the shape of the reconstruction."""
        mat = np.atleast_2d(np.asarray(self.values))
        if not mat.size:
            return "empty reconstruction"
        return f"{mat.shape[0]} points in {mat.shape[1]}-D"

    def _context(self) -> str | None:
        """Return the reconstruction parameters that define the embedding."""
        m = self.meta.get("dimension") if self.meta else None
        tau = self.meta.get("delay") if self.meta else None
        bits = []
        if m is not None:
            bits.append(f"m={m}")
        if tau is not None:
            bits.append(f"τ={tau} samples")
        system = self._system_label()
        if system:
            bits.append(system)
        return ", ".join(bits) or None

    def __plot_spec__(self, kind: str | None = None) -> Any:
        r"""Describe the reconstructed attractor as a backend-agnostic :class:`PlotSpec`.

        Builds a phase portrait of the delay-coordinate trajectory: a 3-D
        ``LINE3D`` of the first three coordinates :math:`(x_i, x_{i+\tau},
        x_{i+2\tau})` for an embedding of dimension :math:`m \ge 3`, a 2-D
        ``LINE`` of the first two for :math:`m = 2`, and a 1-D ``LINE`` against
        the row index for :math:`m = 1`.  The :mod:`tsdynamics.viz.spec` import is
        lazy, so building a spec never pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (a :class:`~tsdynamics.viz.spec.PlotKind`
            value).  ``None`` picks ``PHASE_PORTRAIT_3D`` / ``PHASE_PORTRAIT_2D``
            / ``TIME_SERIES`` from the embedding dimension.

        Returns
        -------
        PlotSpec
        """
        from .. import _plotbuilder as pb

        mat = np.atleast_2d(np.asarray(self.values, dtype=float))
        m = mat.shape[1] if mat.size else 1
        if m >= 3:
            return pb.spec(
                kind,
                "phase_portrait_3d",
                layers=[pb.line3d(mat[:, 0], mat[:, 1], mat[:, 2], label="reconstruction")],
                aspect="equal",
                xlabel="$x_i$",
                ylabel=r"$x_{i+\tau}$",
                zlabel=r"$x_{i+2\tau}$",
                title="delay embedding",
            )
        if m == 2:
            return pb.spec(
                kind,
                "phase_portrait_2d",
                layers=[pb.line(mat[:, 0], mat[:, 1], label="reconstruction")],
                aspect="equal",
                xlabel="$x_i$",
                ylabel=r"$x_{i+\tau}$",
                title="delay embedding",
            )
        series = mat[:, 0] if mat.ndim == 2 and mat.size else np.ravel(mat).astype(float)
        return pb.spec(
            kind,
            "time_series",
            layers=[pb.line(np.arange(series.size, dtype=float), series)],
            xlabel="index",
            ylabel="$x_i$",
            title="delay embedding",
        )


def _as_per_channel(value: int | Sequence[int], n_channels: int, name: str) -> list[int]:
    """Broadcast an int (or validate a per-channel sequence) to a length-``n_channels`` list."""
    if isinstance(value, (int, np.integer)):
        vals = [int(value)] * n_channels
    else:
        vals = [int(v) for v in value]
        if len(vals) != n_channels:
            raise ValueError(f"{name} has {len(vals)} entries but there are {n_channels} channels.")
    return vals


def _as_count(value: Any, name: str) -> Any:
    """Coerce a single integer-like ``dimension`` / ``delay`` — results included.

    ``operator.index`` is the protocol for "this is an integer": it accepts an
    ``int``, a numpy integer, and any result that defines ``__index__`` — which
    is what makes ``embed(x, dimension=embedding_dimension(x))``, the idiom the
    docstrings teach, work for *both* keywords instead of only one.  A genuine
    per-channel sequence has no ``__index__`` and is handed straight back for
    the multivariate path (or for that path's own refusal) to deal with.
    """
    import operator

    try:
        return operator.index(value)
    except TypeError:
        pass
    try:  # a scaling result that is a float in an int's clothing (FNN's m)
        as_float = float(value)
    except (TypeError, ValueError):
        return value
    if float(as_float).is_integer():
        return int(as_float)
    raise invalid_value(
        name,
        value,
        rule="must be a whole number of samples (or, for multivariate input, one per channel)",
        hint=f"round it, or pass a per-channel sequence, e.g. {name}=[3, 3].",
    )


def embed(
    data: Any,
    dimension: int | Sequence[int] | None = None,
    delay: int | Sequence[int] | None = None,
    *,
    components: int | str | None = None,
    **renamed: Any,
) -> Embedding:
    r"""Time-delay embedding of a scalar series (or a multivariate bundle).

    Parameters
    ----------
    data : array-like or Trajectory
        The source signal.  A 1-D series (or a single selected ``components`` of a
        :class:`~tsdynamics.data.Trajectory` / 2-D array) gives a univariate
        embedding.  Pass a 2-D ``(N, d)`` array, a list of equal-length series, or
        a multi-component trajectory **without** ``components`` to embed every
        channel jointly (multivariate embedding).
    dimension : int or sequence of int, optional
        Embedding dimension :math:`m`.  A single int applies to every channel; a
        per-channel sequence sets each channel's dimension (multivariate only).
        Must be ``>= 1``.  **Omit it** and it is estimated with
        :func:`~tsdynamics.analysis.embedding.embedding_dimension` (Cao's
        averaged false neighbours) at the resolved ``delay``.
    delay : int or sequence of int, optional
        Delay :math:`\tau` in samples.  A single int applies to every channel; a
        per-channel sequence sets each channel's delay (multivariate only).  Must
        be ``>= 1``.  **Omit it** and it is estimated with
        :func:`~tsdynamics.analysis.embedding.optimal_delay` (the first minimum of
        the time-delayed mutual information).
    components : int or str, optional
        Select a single channel from a multi-component ``data`` for a univariate
        embedding.  When omitted, a multi-component input is embedded across all
        of its channels.
    **renamed
        Not a real parameter: it exists so that a *near-miss* spelling of one of
        the two above — ``dim=``, ``m=``, ``tau=``, ``lag=``, all of them banned
        by the naming glossary and all of them the first thing a reader tries —
        is answered with the canonical name instead of Python's bare
        ``unexpected keyword argument``.

    Returns
    -------
    Embedding
        The delay-coordinate matrix (behaves as an ``(M, sum(dimension))``
        ``ndarray``), one reconstructed state per row, in temporal
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
    tsdynamics.errors.InvalidParameterError
        If a keyword is a banned spelling of ``dimension`` / ``delay`` (or is not
        a parameter at all).

    Notes
    -----
    The matrix is consumable directly by the point-set analyses — e.g.
    ``correlation_dimension(embed(x, m, tau))`` estimates :math:`D_2` of the
    reconstructed attractor.  Rows are index-ordered with the original sampling,
    so an index-based Theiler window still removes temporally-correlated pairs.

    **Estimated parameters are recorded, never silent.**  ``meta["delay_auto"]``
    / ``meta["dimension_auto"]`` say which of the two the caller supplied, so a
    reconstruction always carries how it was parameterised.  Estimating both
    costs a mutual-information scan and a Cao sweep; pass either one to skip its
    estimate.  Auto-selection is for the univariate case (one series has one
    :math:`\tau` and one :math:`m`) — a multivariate bundle needs its per-channel
    values named.

    References
    ----------
    F. Takens, "Detecting strange attractors in turbulence", in *Dynamical
    Systems and Turbulence*, Lecture Notes in Mathematics **898**, 366 (1981).

    Examples
    --------
    >>> import numpy as np
    >>> from tsdynamics.analysis.embedding import embed
    >>> x = np.arange(10.0)
    >>> y = embed(x, dimension=3, delay=2)
    >>> y.shape
    (6, 3)
    >>> y[0]
    array([0., 2., 4.])
    """
    if renamed:
        raise _renamed_keyword_error(renamed)
    auto = {"dimension_auto": dimension is None, "delay_auto": delay is None}

    # Univariate path: a 1-D series, or an explicitly selected single component.
    univariate = components is not None or _looks_univariate(data)
    if univariate:
        series = _as_series(data, component=components, analysis="embed")
        if delay is None or dimension is None:
            dimension, delay = _estimate_parameters(series, dimension, delay)
        # ``operator.index`` FIRST: chaining is the documented idiom
        # (``embed(x, dimension=embedding_dimension(x), delay=optimal_delay(x))``)
        # and a ``CountResult`` is an ``int`` subclass that ``isinstance(..., int)``
        # accepts — but a bare ``ScalingResult`` is not, so ``dimension=fnn_result``
        # was refused with a message about "a per-channel sequence" and
        # "multivariate input", two concepts the caller never mentioned, while
        # ``delay=`` (whose result IS an int subclass) sailed through.  The
        # asymmetry was invisible until it bit.
        dimension = _as_count(dimension, "dimension")
        delay = _as_count(delay, "delay")
        for label, value in (("dimension", dimension), ("delay", delay)):
            if not isinstance(value, (int, np.integer)):
                raise ValueError(f"a per-channel `{label}` sequence needs a multivariate input.")
        m_int, tau_int = int(cast(int, dimension)), int(cast(int, delay))
        embedded = _embed_single(series, m_int, tau_int)
        return Embedding(values=embedded, meta={**_embed_meta(dimension, delay), **auto})

    if dimension is None or delay is None:
        missing = [n for n, v in (("dimension", dimension), ("delay", delay)) if v is None]
        raise invalid_value(
            "/".join(missing),
            None,
            rule=(
                "must be given for a multivariate embedding (each channel has its own, so "
                "there is nothing to estimate from one series)"
            ),
            hint=(
                "pass a per-channel sequence, e.g. embed(data, dimension=[3, 3], "
                "delay=[7, 5]) — or select one channel with components= to get the "
                "estimated univariate reconstruction."
            ),
        )

    channels = _as_channels(data, analysis="embed")
    n_channels = channels.shape[1]
    dims = _as_per_channel(dimension, n_channels, "dimension")
    delays = _as_per_channel(delay, n_channels, "delay")

    n = channels.shape[0]
    spans = [(m - 1) * tau for m, tau in zip(dims, delays, strict=True)]
    _validate(dims, delays, n, max(spans))
    rows = n - max(spans)

    blocks = [_embed_single(channels[:, c], dims[c], delays[c])[:rows] for c in range(n_channels)]
    embedded = np.ascontiguousarray(np.hstack(blocks))
    return Embedding(values=embedded, meta={**_embed_meta(dimension, delay), **auto})


#: Banned near-miss spellings of :func:`embed`'s two parameters, mapped to the
#: canonical name.  Straight from the frozen naming glossary (§2) — the point is
#: that the glossary's decision is *enforced with an explanation* rather than
#: with Python's bare ``unexpected keyword argument``, which names neither the
#: right spelling nor the fact that a rule was applied.
_RENAMED: dict[str, str] = {
    "dim": "dimension",
    "m": "dimension",
    "emb_dim": "dimension",
    "tau": "delay",
    "lag": "delay",
}


def _renamed_keyword_error(renamed: dict[str, Any]) -> Exception:
    """Explain a rejected keyword: the canonical spelling, or that it is not a parameter."""
    parts = []
    for key, value in renamed.items():
        canonical = _RENAMED.get(key)
        if canonical is not None:
            parts.append(f"{key}={value!r} → {canonical}={value!r}")
        else:
            parts.append(f"{key}= is not a parameter of embed")
    return invalid_value(
        "keyword argument",
        sorted(renamed),
        rule="is not accepted by embed",
        hint=(
            "; ".join(parts)
            + ". embed(data, dimension, delay, *, components=None) — `dimension` (m) and "
            "`delay` (tau, in samples) are the canonical spellings, and either may be "
            "omitted to have it estimated."
        ),
    )


def _estimate_parameters(
    series: np.ndarray, dimension: Any, delay: Any
) -> tuple[int | Sequence[int], int | Sequence[int]]:
    """Fill in whichever of ``(dimension, delay)`` was not given, from the series itself.

    The delay comes first (the first minimum of the time-delayed mutual
    information, Fraser & Swinney) because the dimension estimate needs one: Cao's
    false-neighbour ratio is computed *at* a delay.  Estimating in the other order
    would evaluate the dimension at an arbitrary lag and then move the lag out
    from under it.

    A failing *estimator* is re-raised in ``embed``'s own terms.  Its message is
    written for its own caller and names its own parameters (``max_dim``), which
    a reader who typed ``embed(x)`` never passed and cannot find in ``embed``'s
    signature — so the estimator's diagnosis is kept and the advice is the one
    that applies here: give the numbers instead.

    The advice **follows the diagnosis** rather than assuming it.  An estimator
    can fail for a reason that has nothing to do with length — a constant series
    has undefined mutual information at *any* record length — and telling that
    caller "500 samples is not enough" contradicts, in the same sentence, the
    reason quoted immediately before it.
    """
    from .delay import optimal_delay
    from .dimension import embedding_dimension

    try:
        if delay is None:
            delay = int(optimal_delay(series))
        if dimension is None:
            dimension = int(embedding_dimension(series, delay=int(delay)))
    except ValueError as err:
        which = "delay" if delay is None else "dimension"
        too_short = "short" in str(err) or "enough" in str(err)
        raise invalid_value(
            which,
            None,
            rule=f"could not be estimated from this series ({err})",
            hint=(
                (f"{series.size} samples is not enough for the estimator to work with. ")
                if too_short
                else ""
            )
            + "Pass the value explicitly — e.g. embed(data, dimension=3, delay=1)"
            + (" — or embed a longer record." if too_short else "."),
        ) from err
    return dimension, delay


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
        return bool(data.y.ndim == 1 or data.y.shape[1] == 1)
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
    return _delay_columns(series, m, tau, n - span)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
