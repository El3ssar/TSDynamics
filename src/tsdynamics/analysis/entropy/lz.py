r"""
Lempel–Ziv (LZ76) complexity and entropy-rate estimate.

The LZ76 complexity ``c(S)`` of a symbol sequence is the number of factors in its
exhaustive-history parse — each factor a previously unseen substring.  The
normalised entropy density ``h ≈ c(S)·log_k(n)/n`` (``k`` = alphabet size)
converges to the source entropy rate for an ergodic source, so LZ76 doubles as a
fast, threshold-free complexity and entropy estimator for symbolic series.

Continuous signals are symbolised first (median-threshold binarisation is the
common neuroscience choice).  A native, dependency-free parser is the default
provider; ``provider="lzcomplexity"`` delegates to the C++ ``lzcomplexity``
library (Aragon Perez & Estevez Rams) when it is installed, for large sequences.

References
----------
Lempel, A. & Ziv, J. (1976). On the complexity of finite sequences.
*IEEE Trans. Inf. Theory* **22**, 75–81.

Kaspar, F. & Schuster, H. G. (1987). Easily calculable measure for the
complexity of spatiotemporal patterns. *Phys. Rev. A* **36**, 842–848.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import as_series

__all__ = ["binarize", "lz76_complexity", "lz76_entropy", "lz76_factors"]


def binarize(x: Any, threshold: str | float = "median", *, component: int | str | None = None):
    r"""
    Map a real-valued series to a binary symbol sequence.

    Parameters
    ----------
    x : array-like or Trajectory
        Scalar time series.
    threshold : {"median", "mean"} or float, default "median"
        Symbols are ``1`` where the series exceeds the threshold, else ``0``.
        Median thresholding is the standard choice for LZ analysis because it is
        robust to outliers and yields a balanced alphabet.
    component : int or str, optional
        Component selector for multi-component input.

    Returns
    -------
    numpy.ndarray
        Integer array of ``0``/``1`` symbols.
    """
    series = as_series(x, component)
    if threshold == "median":
        thr = float(np.median(series))
    elif threshold == "mean":
        thr = float(series.mean())
    else:
        thr = float(threshold)
    return (series > thr).astype(np.intp)


def _symbols(x: Any, symbolize: Any, threshold: str | float, component: int | str | None):
    """Resolve arbitrary input to a 1-D symbol array (no code densification yet)."""
    if isinstance(x, str):  # already symbolic; component is meaningless
        return np.frombuffer(x.encode("utf-8", "surrogatepass"), dtype=np.uint8)
    if callable(symbolize):
        return np.asarray(symbolize(as_series(x, component))).ravel()
    if symbolize in ("median", "mean"):
        return binarize(x, symbolize, component=component)

    arr = np.asarray(x)
    is_traj = hasattr(x, "component") and hasattr(x, "y") and not isinstance(x, np.ndarray)
    is_float = np.issubdtype(arr.dtype, np.floating)
    # Pass already-symbolic input through: integer arrays always, and (for
    # symbolize=None) any non-float array.  Float-as-symbol is rejected below
    # because every distinct value would become its own symbol.
    passthrough = not is_traj and (
        np.issubdtype(arr.dtype, np.integer) or (symbolize is None and not is_float)
    )
    if passthrough:
        if arr.ndim == 1:
            if component not in (None, 0):
                raise ValueError("component= is meaningless for a 1-D symbolic series.")
            return arr
        if arr.ndim == 2:
            if component is None:
                raise ValueError("2-D symbolic input needs component= to select a column.")
            return arr[:, int(component)]
        raise ValueError("expected 1-D or 2-D symbolic input.")
    if symbolize == "auto":  # float / trajectory → median-binarise the series
        return binarize(as_series(x, component), threshold)
    raise ValueError(
        f"cannot symbolise floating-point input with symbolize={symbolize!r}; "
        "pass 'median'/'mean', a callable, or pre-symbolised data."
    )


def _to_codes(
    x: Any, symbolize: Any, threshold: str | float, component: int | str | None = None
) -> np.ndarray:
    """Reduce arbitrary input to a contiguous ``0..k-1`` integer code array."""
    # Dense integer codes — the parse only needs equality + indexing.
    _, codes = np.unique(_symbols(x, symbolize, threshold, component), return_inverse=True)
    return codes.ravel().astype(np.intp, copy=False)


def _lz76_parse(codes: np.ndarray) -> tuple[int, list[int]]:
    """
    Exhaustive-history LZ76 parse (Lempel–Ziv 1976; Kaspar–Schuster 1987).

    Returns ``(complexity, factor_start_indices)``.  The complexity is the number
    of factors; the starts mark where each factor begins (the ``i``-th factor
    spans ``[starts[i], starts[i+1])``, the last running to the end).
    """
    n = codes.size
    if n == 0:
        return 0, []
    if n == 1:
        return 1, [0]

    s = codes.tolist()  # plain-list indexing is faster than numpy scalar access
    c = 1
    lengths = [1]  # the first symbol is always factor 1
    i = 0  # scan pointer into the already-parsed prefix
    lp = 1  # start of the factor currently being formed (consumed-prefix length)
    k = 1  # candidate match length
    k_max = 1  # longest reproducible prefix from position lp
    while True:
        if s[i + k - 1] == s[lp + k - 1]:
            k += 1
            if lp + k > n:  # ran off the end while still matching → close final factor
                c += 1
                lengths.append(n - lp)
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == lp:  # no earlier position reproduces a longer prefix
                c += 1
                lengths.append(k_max)
                lp += k_max
                if lp + 1 > n:  # consumed the whole sequence (lp == n)
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1

    # Factor starts are the cumulative sums of all but the last factor length.
    starts = [0]
    acc = 0
    for length in lengths[:-1]:
        acc += length
        starts.append(acc)
    return c, starts


def lz76_factors(
    x: Any,
    *,
    symbolize: Any = "auto",
    threshold: str | float = "median",
    component: int | str | None = None,
) -> list[int]:
    """
    Factor-boundary start indices of the LZ76 exhaustive-history parse.

    See :func:`lz76_complexity` for the symbolisation arguments.  ``len(...)`` of
    the result equals the LZ76 complexity.
    """
    codes = _to_codes(x, symbolize, threshold, component)
    return _lz76_parse(codes)[1]


def lz76_complexity(
    data: Any,
    *,
    symbolize: Any = "auto",
    threshold: str | float = "median",
    normalize: bool = False,
    provider: str = "native",
    component: int | str | None = None,
) -> float:
    r"""
    Lempel–Ziv (LZ76) complexity of a sequence.

    Parameters
    ----------
    data : array-like, str, or Trajectory
        The sequence.  Strings and integer arrays are treated as already
        symbolic; a real-valued series is symbolised first (see ``symbolize``).
    symbolize : {"auto", "median", "mean", None} or callable, default "auto"
        How to turn the input into symbols.  ``"auto"`` passes strings and
        integer arrays through unchanged and median-binarises floating-point
        series; ``"median"``/``"mean"`` force binarisation; a callable is applied
        to the (float) series and must return symbols; ``None`` insists the input
        is already symbolic.
    threshold : {"median", "mean"} or float, default "median"
        Threshold used when binarising (see :func:`binarize`).
    normalize : bool, default False
        Return the normalised complexity ``c·log_k(n)/n`` (= :func:`lz76_entropy`)
        instead of the raw factor count.
    provider : {"native", "lzcomplexity"}, default "native"
        ``"native"`` uses the built-in parser; ``"lzcomplexity"`` delegates to the
        optional C++ ``lzcomplexity`` package (faster for very long sequences).
    component : int or str, optional
        Component selector for multi-component input.

    Returns
    -------
    float
        The LZ76 complexity (an integer count, or the normalised density when
        ``normalize=True``).

    Examples
    --------
    >>> lz76_complexity("0001101001000101", symbolize=None)   # Kaspar–Schuster
    6.0
    >>> lz76_complexity("aaaaaaaa", symbolize=None)            # constant → 2
    2.0
    """
    codes = _to_codes(data, symbolize, threshold, component)
    n = codes.size
    k = int(np.unique(codes).size) if n else 0

    if provider == "native":
        c = _lz76_parse(codes)[0]
    elif provider == "lzcomplexity":
        c = _lz76_via_lzcomplexity(codes, k)
    else:
        raise ValueError(f"unknown provider {provider!r}; use 'native' or 'lzcomplexity'.")

    if normalize:
        return _normalized_density(c, n, k)
    return float(c)


def lz76_entropy(
    data: Any,
    *,
    symbolize: Any = "auto",
    threshold: str | float = "median",
    provider: str = "native",
    component: int | str | None = None,
) -> float:
    r"""
    LZ76 entropy-rate estimate ``h ≈ c(S)·log_k(n)/n``.

    The normalised LZ76 complexity, which converges to the entropy rate of an
    ergodic source (Lempel & Ziv 1976).  Equivalent to
    :func:`lz76_complexity` with ``normalize=True``.  Arguments are as in
    :func:`lz76_complexity`.

    Returns
    -------
    float
        Entropy density in units of ``log_k`` (i.e. normalised to ``[0, ~1]`` for
        a ``k``-symbol source).
    """
    return lz76_complexity(
        data,
        symbolize=symbolize,
        threshold=threshold,
        normalize=True,
        provider=provider,
        component=component,
    )


def _normalized_density(c: int, n: int, k: int) -> float:
    """``c·log_k(n)/n`` with the degenerate (k ≤ 1 or n ≤ 1) cases guarded."""
    if n <= 1 or k <= 1:
        return 0.0
    return float(c * (np.log(n) / np.log(k)) / n)


def _lz76_via_lzcomplexity(codes: np.ndarray, k: int) -> int:
    """Delegate the factor count to the optional ``lzcomplexity`` C++ library."""
    try:
        import lzcomplexity as lz
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "provider='lzcomplexity' needs the optional 'lzcomplexity' package; "
            "install it (pip install lzcomplexity) or use provider='native'."
        ) from exc
    if k > 90:
        raise ValueError(
            "the lzcomplexity bridge supports alphabets up to 90 symbols; "
            "use provider='native' for larger alphabets."
        )
    # Map dense codes to distinct printable ASCII characters (avoids the library's
    # list[int] → concatenated-decimals gotcha) and pass the resulting string.
    text = "".join(chr(0x30 + int(code)) for code in codes)
    return int(lz.lz76Factorization(text))


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
