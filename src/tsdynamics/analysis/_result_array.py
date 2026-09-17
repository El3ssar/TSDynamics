"""The :class:`ArrayResult` ndarray-valued result.

Split out of ``analysis/_result.py``; wraps a bare ``ndarray`` return so it
stays a drop-in for its array while carrying the result surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from tsdynamics.analysis._result_base import AnalysisResult, _unknown_result_attribute
from tsdynamics.analysis._result_json import _jsonify, _state
from tsdynamics.analysis._result_viz import VisualizationNotInstalled


@dataclass(frozen=True, eq=False)
class ArrayResult(np.lib.mixins.NDArrayOperatorsMixin, AnalysisResult):
    """An array-valued measurement that stays a drop-in for its ``ndarray``.

    Wraps a bare ``ndarray`` return (a Lyapunov spectrum, a delay-embedded point
    cloud, a mutual-information-vs-lag curve, a surrogate ensemble) so it carries
    the result surface while ``np.asarray(result)``, indexing/slicing, ``len`` /
    iteration, elementwise comparisons (``result >= 0``) and attribute access
    (``result.shape``, ``result.max()``) all defer to the underlying array.

    Indexing and slicing return the *raw* array element/sub-array (never another
    wrapper), so ``result[:, 0]`` flows straight into NumPy as before.  Operators
    are resolved on the type (``__getattr__`` cannot intercept them), so the
    comparison/arithmetic dunders are spelled out and return raw arrays;
    :class:`numpy.lib.mixins.NDArrayOperatorsMixin` supplies the rest
    (``/``, ``**``, ``//``, ``%``, unary ``-``, the in-place forms) through
    :meth:`__array_ufunc__` — before v6 ``result / 2`` raised ``TypeError`` while
    ``result * 2`` worked, which is the worst of both worlds for a drop-in.

    Attributes
    ----------
    values : numpy.ndarray
        The wrapped array.  ``np.asarray(result)`` returns it.
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ()

    values: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)

    # -- the readout ---------------------------------------------------------

    def _answer(self) -> str:
        """Return the array itself when it is small, else its shape."""
        arr = np.asarray(self.values)
        if arr.ndim <= 1:
            return f"= {_state(arr)}"
        return f"{arr.shape[0]} × {int(np.prod(arr.shape[1:]))} values"

    # -- ndarray protocol ----------------------------------------------------

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """Unwrap every result operand to its array and defer to NumPy."""
        args = [np.asarray(x) if isinstance(x, ArrayResult) else x for x in inputs]
        out = kwargs.get("out")
        if out is not None:
            kwargs["out"] = tuple(np.asarray(o) if isinstance(o, ArrayResult) else o for o in out)
        return getattr(ufunc, method)(*args, **kwargs)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:  # noqa: D105
        # NumPy 2.0 passes ``copy`` into ``__array__``; honor it so ``np.array``
        # (which defaults to copying) does not error under ``filterwarnings=error``.
        arr = np.asarray(self.values)
        if dtype is not None:
            arr = arr.astype(dtype, copy=bool(copy))
        elif copy:
            arr = arr.copy()
        return arr

    def __getitem__(self, key: Any) -> Any:  # noqa: D105
        return self.values[key]

    def __len__(self) -> int:  # noqa: D105
        return len(self.values)

    def __iter__(self) -> Any:  # noqa: D105
        return iter(self.values)

    def __getattr__(self, name: str) -> Any:
        """Forward to the wrapped array — but answer a WRONG guess as this class.

        ``.shape`` / ``.max`` / ``.sum`` are the point of wrapping an array and
        keep forwarding.  A name numpy does not have used to be answered by
        numpy anyway (``AttributeError: 'numpy.ndarray' object has no attribute
        'chaotic'``), naming a type the caller never typed and offering no
        replacement — so every wrong guess on every result was the worst error
        message in the library.  Now the answer names THIS result and what it
        does carry.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            values = object.__getattribute__(self, "values")
        except AttributeError as exc:  # pragma: no cover - during unpickling
            raise AttributeError(name) from exc
        if hasattr(values, name):
            return getattr(values, name)
        raise _unknown_result_attribute(self, name)

    # -- elementwise comparisons / arithmetic (return raw arrays) ------------

    def __eq__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) == other

    def __ne__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) != other

    def __lt__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) < other

    def __le__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) <= other

    def __gt__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) > other

    def __ge__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) >= other

    __hash__ = None  # type: ignore[assignment]  # array-valued → unhashable, like ndarray

    def __add__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) + other

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) - other

    def __mul__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) * other

    __rmul__ = __mul__

    def to_dict(self, full: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly mapping (the array as a nested list + ``meta``).

        Parameters
        ----------
        full : bool, default False
            Also emit the derived quantities the repr reports (see
            :meth:`AnalysisResult._derived`).

        Returns
        -------
        dict
            ``{"values": <nested list>, "meta": <provenance>}``.
        """
        data = {"values": _jsonify(self.values), "meta": _jsonify(self.meta)}
        if full:
            data.update({k: _jsonify(v) for k, v in self._derived().items()})
        return data

    def to_frame(self) -> Any:
        """Return a :class:`pandas.DataFrame` tabulating the wrapped array.

        The base :meth:`AnalysisResult.to_frame` would return an *empty* frame
        here, because ``values`` is declared ``repr=False`` and so is not a
        display field — it tabulates display scalars, of which an
        :class:`ArrayResult` has none.  This override puts the array itself in the
        table instead: a 1-D array becomes a single ``value`` column (one row per
        element); a 2-D array becomes one ``c{j}`` column per channel (column).
        ``meta`` rides on ``frame.attrs["meta"]`` like the other subclasses.

        ``pandas`` is a soft dependency, imported lazily; a missing install raises
        an :class:`ImportError` pointing at ``pip install pandas`` (there is no
        ``tsdynamics[frame]`` extra — this docstring used to name one).

        Returns
        -------
        pandas.DataFrame
            One column (``value``) for a 1-D array, one per channel for a 2-D
            array, with ``meta`` on ``frame.attrs["meta"]``.

        Raises
        ------
        ImportError
            If :mod:`pandas` is not installed (the message points at
            ``pip install pandas``).
        """
        pd = self._require_pandas()
        arr = np.asarray(self.values)
        if arr.ndim <= 1:
            frame = pd.DataFrame({"value": arr.ravel()})
        else:
            flat = arr.reshape(arr.shape[0], -1)
            frame = pd.DataFrame({f"c{j}": flat[:, j] for j in range(flat.shape[1])})
        frame.attrs["meta"] = dict(self.meta) if self.meta else {}
        return frame

    def __plot_spec__(self, kind: str | None = None) -> Any:
        """Describe the wrapped array as a :class:`PlotSpec` (safe generic view).

        A 1-D array becomes a ``DIAGNOSTIC_CURVE`` ``LINE`` against its index; a
        2-column / 3-column array becomes a phase-portrait point cloud (2-D
        ``SCATTER`` / 3-D ``LINE3D``); any other 2-D array becomes a
        ``LINE_FAMILY`` (one line per column).  Subclasses with a natural figure
        (a Lyapunov spectrum, a mutual-information curve) override this.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.  ``None`` picks the natural kind above.

        Raises
        ------
        VisualizationNotInstalled
            If the array is empty or higher-than-2-D (nothing generic to draw).
        """
        from . import _plotbuilder as pb

        arr = np.asarray(self.values, dtype=float)
        title = type(self).__name__

        if arr.ndim == 1 and arr.size:
            return pb.spec(
                kind,
                "diagnostic_curve",
                layers=[pb.line(np.arange(arr.size, dtype=float), arr)],
                xlabel="index",
                ylabel="value",
                title=title,
                meta=self.meta,
            )

        if arr.ndim == 2 and arr.shape[0] and arr.shape[1] in (2, 3):
            if arr.shape[1] == 3:
                return pb.spec(
                    kind,
                    "phase_portrait_3d",
                    layers=[pb.line3d(arr[:, 0], arr[:, 1], arr[:, 2])],
                    xlabel="x1",
                    ylabel="x2",
                    zlabel="x3",
                    title=title,
                    meta=self.meta,
                )
            return pb.spec(
                kind,
                "phase_portrait_2d",
                layers=[pb.scatter(arr[:, 0], arr[:, 1])],
                aspect="equal",
                xlabel="x1",
                ylabel="x2",
                title=title,
                meta=self.meta,
            )

        if arr.ndim == 2 and arr.shape[0]:
            idx = np.arange(arr.shape[0], dtype=float)
            layers = [pb.line(idx, arr[:, j], label=f"[{j}]") for j in range(min(arr.shape[1], 12))]
            return pb.spec(
                kind,
                "line_family",
                layers=layers,
                xlabel="index",
                ylabel="value",
                title=title,
                legend=True,
                meta=self.meta,
            )

        raise VisualizationNotInstalled(
            f"{title} wraps an empty or higher-than-2-D array, so the generic ArrayResult "
            "__plot_spec__() has nothing to draw; export it with .to_dict() instead."
        )
