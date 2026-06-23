"""Unit tests for the shared :class:`AnalysisResult` base (WS-RESULT)."""

from __future__ import annotations

import dataclasses
import json
import sys
import types
from dataclasses import dataclass, field

import numpy as np
import pytest

from tsdynamics.analysis._result import AnalysisResult, VisualizationNotInstalled

# ---------------------------------------------------------------------------
# Test result subclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Spectrum(AnalysisResult):
    """A scalar-and-array result with an interpretation line."""

    _repr_fields = ("exponents", "kaplan_yorke")

    exponents: np.ndarray = field(repr=False)
    kaplan_yorke: float = 0.0

    def _interpretation(self) -> str:
        n_pos = int((np.asarray(self.exponents) > 0).sum())
        return f"chaotic: {n_pos} positive exponent(s)" if n_pos else "regular"


@dataclass(frozen=True)
class _Auto(AnalysisResult):
    """No ``_repr_fields`` — repr is introspected from the fields."""

    scalar: float = 0.0
    curve: np.ndarray = field(default_factory=lambda: np.zeros(8), repr=False)


@dataclass(frozen=True)
class _Custom(AnalysisResult):
    """A subclass that writes its own ``__repr__``."""

    value: float = 1.0

    def __repr__(self) -> str:
        return f"CUSTOM<{self.value}>"


class _FakeSystem:
    def _provenance(self, **extra):
        return {"system": "Lorenz", "params": {"sigma": 10.0}, "tsdynamics": "4.0", **extra}


def _spectrum() -> _Spectrum:
    return _Spectrum(
        np.array([0.906, 0.0, -14.57]),
        kaplan_yorke=2.06,
        meta=AnalysisResult.build_meta(_FakeSystem(), final_time=300.0),
    )


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_repr_honours_repr_fields_and_survives_dataclass():
    r = _spectrum()
    text = repr(r)
    assert text.startswith("_Spectrum(")
    assert "exponents=" in text  # listed in _repr_fields despite field(repr=False)
    assert "kaplan_yorke=2.06" in text


def test_repr_introspects_fields_when_no_repr_fields():
    text = repr(_Auto(scalar=5.0))
    assert "scalar=5" in text
    assert "curve" not in text  # field(repr=False) excluded
    assert "meta" not in text  # meta never shown in repr


def test_repr_formats_floats_compactly():
    assert "kaplan_yorke=2.06" in repr(_spectrum())


def test_custom_repr_is_respected():
    assert repr(_Custom(3.0)) == "CUSTOM<3.0>"


def test_grandchild_inherits_repr_fields_repr():
    @dataclass(frozen=True)
    class _Mid(AnalysisResult):
        _repr_fields = ("a",)
        a: float = 0.0

    @dataclass(frozen=True)
    class _Grand(_Mid):
        _repr_fields = ("a", "b")
        a: float = 0.0
        b: float = 0.0

    text = repr(_Grand(a=1.0, b=2.0))
    assert "a=1" in text and "b=2" in text


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def test_summary_has_header_fields_and_interpretation():
    out = _spectrum().summary()
    assert out.splitlines()[0] == "_Spectrum  (Lorenz)"  # header + system label
    assert "kaplan_yorke = 2.06" in out
    assert "→ chaotic: 1 positive exponent(s)" in out


def test_summary_omits_interpretation_when_none():
    out = _Auto(scalar=1.0).summary()
    assert "→" not in out
    assert out.splitlines()[0] == "_Auto"  # no system label in meta


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_jsonifies_arrays_and_is_json_serializable():
    d = _spectrum().to_dict()
    assert d["exponents"] == [0.906, 0.0, -14.57]  # ndarray -> list
    assert d["kaplan_yorke"] == 2.06
    json.dumps(d)  # must not raise


def test_to_dict_includes_meta():
    d = _spectrum().to_dict()
    assert d["meta"]["system"] == "Lorenz"
    assert d["meta"]["final_time"] == 300.0


def test_to_dict_coerces_numpy_scalars():
    @dataclass(frozen=True)
    class _R(AnalysisResult):
        v: float = 0.0

    d = _R(v=np.float64(1.5)).to_dict()
    assert d["v"] == 1.5 and isinstance(d["v"], float)
    json.dumps(d)


# ---------------------------------------------------------------------------
# to_frame
# ---------------------------------------------------------------------------


def test_to_frame_missing_pandas_gives_install_hint(monkeypatch):
    # Force `import pandas` to fail regardless of the environment.
    import builtins

    real_import = builtins.__import__

    def _fail(name, *args, **kw):
        if name == "pandas" or name.startswith("pandas."):
            raise ImportError("no pandas")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", _fail)
    with pytest.raises(ImportError, match=r"tsdynamics\[frame\]"):
        _spectrum().to_frame()


def test_to_frame_with_pandas_builds_scalar_row():
    pd = pytest.importorskip("pandas")
    frame = _spectrum().to_frame()
    assert isinstance(frame, pd.DataFrame)
    # Only scalar display fields become columns; arrays are excluded.
    assert "kaplan_yorke" in frame.columns
    assert "exponents" not in frame.columns
    assert len(frame) == 1
    assert frame.attrs["meta"]["system"] == "Lorenz"


# ---------------------------------------------------------------------------
# plot seam
# ---------------------------------------------------------------------------


def test_visualization_not_installed_is_importerror():
    assert issubclass(VisualizationNotInstalled, ImportError)


@pytest.fixture
def _no_backend(monkeypatch):
    """Force an empty renderers registry so the ``.plot`` seam raises.

    The matplotlib backend lazily auto-registers on first render as of stream
    VIZ-MPL-CORE, so the no-backend path is exercised by clearing the registry
    and stubbing :func:`register_builtin_renderers` to a no-op, then restoring it.
    """
    from tsdynamics import registry
    from tsdynamics.viz import render as render_mod

    saved = registry.renderers.all()
    registry.renderers.clear()
    monkeypatch.setattr(render_mod, "register_builtin_renderers", lambda *a, **k: [])
    try:
        yield
    finally:
        registry.renderers.clear()
        for entry in saved:
            registry.renderers.register(entry.name, entry.obj, replace=True)


def test_plot_call_raises_without_backend(_no_backend):
    with pytest.raises(VisualizationNotInstalled):
        _spectrum().plot()


@pytest.mark.parametrize(
    "method",
    [
        "scaling",
        "diagnostic",
        "time_series",
        "phase",
        "image",
        "bifurcation",
        "return_map",
        "histogram",
        "spectrum",
        "section",
    ],
)
def test_plot_typed_methods_raise_without_backend(method, _no_backend):
    accessor = _spectrum().plot
    assert hasattr(accessor, method)
    with pytest.raises(VisualizationNotInstalled):
        getattr(accessor, method)()


def test_plot_accessor_repr():
    assert "plot accessor" in repr(_spectrum().plot)


def test_plot_renders_when_a_renderer_is_registered(monkeypatch):
    """Forward-compat: once a backend registers, the seam renders the spec.

    Pins the documented PlotSpec contract — ``to_plot_spec(kind=...)`` carries the
    semantic kind and ``render(backend, **backend_kw)`` does the drawing (``kind``
    is never forwarded to ``render``).
    """

    # render() matches the documented signature: render(self, backend="...", **backend_kw).
    class _FakeSpec:
        def __init__(self, kind):
            self.kind = kind

        def render(self, backend="matplotlib", **backend_kw):
            return {"backend": backend, "kind": self.kind, "backend_kw": backend_kw}

    @dataclass(frozen=True)
    class _Plottable(AnalysisResult):
        value: float = 0.0

        def to_plot_spec(self, kind=None):
            return _FakeSpec(kind)

    # Inject a non-empty renderer registry (registry.renderers does not exist yet).
    import tsdynamics.registry as reg

    monkeypatch.setattr(reg, "renderers", ["matplotlib-stub"], raising=False)

    r = _Plottable(value=1.0)
    out = r.plot(backend="plotly")
    assert out == {"backend": "plotly", "kind": None, "backend_kw": {}}
    # A typed method routes its kind into to_plot_spec; backend kwargs reach render.
    out2 = r.plot.scaling(backend="mpl", ax="axes-handle")
    assert out2["kind"] == "scaling_fit"
    assert out2["backend"] == "mpl"
    assert out2["backend_kw"] == {"ax": "axes-handle"}


def test_empty_renderer_registry_still_raises(monkeypatch):
    import tsdynamics.registry as reg

    monkeypatch.setattr(reg, "renderers", [], raising=False)  # registered, but empty
    with pytest.raises(VisualizationNotInstalled):
        _spectrum().plot()


# ---------------------------------------------------------------------------
# _repr_html_
# ---------------------------------------------------------------------------


def test_repr_html_has_caption_and_fields():
    html_out = _spectrum()._repr_html_()
    assert "<table>" in html_out and "</table>" in html_out
    assert "_Spectrum (Lorenz)" in html_out
    assert "kaplan_yorke" in html_out
    assert "chaotic" in html_out  # interpretation footer


def test_repr_html_escapes_markup():
    @dataclass(frozen=True)
    class _R(AnalysisResult):
        label: str = ""

    out = _R(label="<b>x</b>")._repr_html_()
    assert "<b>x</b>" not in out
    assert "&lt;b&gt;" in out


# ---------------------------------------------------------------------------
# provenance / build_meta
# ---------------------------------------------------------------------------


def test_build_meta_delegates_to_provenance():
    meta = AnalysisResult.build_meta(_FakeSystem(), transient=50)
    assert meta["system"] == "Lorenz"
    assert meta["params"] == {"sigma": 10.0}
    assert meta["transient"] == 50


def test_build_meta_fallback_for_plain_object():
    class Plain:
        pass

    meta = AnalysisResult.build_meta(Plain(), n=10)
    assert meta == {"system": "Plain", "n": 10}


def test_build_meta_none_system():
    assert AnalysisResult.build_meta(None, k=3) == {"k": 3}


# ---------------------------------------------------------------------------
# dataclass mechanics (frozen, kw-only meta)
# ---------------------------------------------------------------------------


def test_result_is_frozen():
    r = _spectrum()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.kaplan_yorke = 9.0  # type: ignore[misc]


def test_meta_is_keyword_only_and_defaults_empty():
    r = _Spectrum(np.array([1.0]), 1.0)  # positional: exponents, kaplan_yorke
    assert r.meta == {}
    with pytest.raises(TypeError):
        _Spectrum(np.array([1.0]), 1.0, {"system": "X"})  # meta cannot be positional


def test_base_result_alone_has_clean_repr_and_summary():
    b = AnalysisResult(meta={"system": "X"})
    assert repr(b) == "AnalysisResult()"
    assert b.summary().splitlines()[0] == "AnalysisResult  (X)"
    assert b.to_dict() == {"meta": {"system": "X"}}


# ---------------------------------------------------------------------------
# equality / hashing (meta and arrays are not identity)
# ---------------------------------------------------------------------------


def test_base_result_is_hashable_and_meta_excluded_from_identity():
    # meta is provenance, not identity: hashing works and equality ignores it.
    a = AnalysisResult(meta={"system": "A"})
    b = AnalysisResult(meta={"system": "B", "run": 7})
    assert hash(a) == hash(b)
    assert a == b
    assert {a, b} == {a}  # usable as a set element / dict key


def test_scalar_subclass_equality_ignores_meta():
    @dataclass(frozen=True)
    class _R(AnalysisResult):
        value: float = 0.0

    p = _R(1.0, meta={"run": 1})
    q = _R(1.0, meta={"run": 2})
    assert p == q and hash(p) == hash(q)
    assert _R(2.0) != _R(1.0)


def test_array_field_subclass_is_comparable_and_hashable_with_compare_false():
    # The documented subclass shape: array fields declared field(compare=False).
    @dataclass(frozen=True)
    class _R(AnalysisResult):
        _repr_fields = ("scalar",)
        exponents: np.ndarray = field(repr=False, compare=False)
        scalar: float = 0.0

    a = _R(np.array([1.0, 2.0, 3.0]), scalar=5.0)
    b = _R(np.array([9.0, 9.0]), scalar=5.0)  # different array, same scalar
    assert a == b  # would raise ValueError if the array were in __eq__
    assert hash(a) == hash(b)  # would raise TypeError if the array were in __hash__


# ---------------------------------------------------------------------------
# review-hardening: repr/formatting/to_dict/to_frame edge cases
# ---------------------------------------------------------------------------


def test_fmt_numpy_bool_renders_as_plain_bool():
    @dataclass(frozen=True)
    class _R(AnalysisResult):
        flag: object = None

    assert "flag=True" in repr(_R(flag=np.bool_(True)))
    assert "flag=False" in repr(_R(flag=np.bool_(False)))


def test_grandchild_inherits_a_parents_custom_repr():
    @dataclass(frozen=True)
    class _Mid(AnalysisResult):
        v: float = 0.0

        def __repr__(self) -> str:
            return f"MID<{self.v}>"

    @dataclass(frozen=True)
    class _Grand(_Mid):
        v: float = 0.0
        w: float = 0.0

    assert repr(_Grand(v=5.0, w=9.0)) == "MID<5.0>"


def test_repr_summary_html_skip_undeclared_repr_fields():
    @dataclass(frozen=True)
    class _Ghost(AnalysisResult):
        _repr_fields = ("ghost", "real")
        real: float = 1.0

    r = _Ghost(real=2.0)
    for text in (repr(r), r.summary(), r._repr_html_()):
        assert "ghost" not in text  # undeclared attribute silently skipped
    assert "real=2" in repr(r)
    assert "real = 2" in r.summary()
    assert "real" in r._repr_html_()


def test_to_dict_recurses_through_nested_mappings_and_containers():
    @dataclass(frozen=True)
    class _Nested(AnalysisResult):
        payload: dict = field(default_factory=dict)

    r = _Nested(
        payload={
            "arr": np.array([1.0, 2.0]),
            "mixed": (np.float64(3.0), [np.int64(4)]),
            "inner": {1: np.arange(3)},  # non-str key + nested array
            "uniq": {np.int64(9)},  # set -> list
        },
        meta={"params": {"sigma": 10.0}, "trace": np.arange(2)},
    )
    d = r.to_dict()
    assert d["payload"]["arr"] == [1.0, 2.0]
    assert d["payload"]["mixed"] == [3.0, [4]]  # tuple -> list
    assert d["payload"]["inner"] == {"1": [0, 1, 2]}  # int key -> "1", array -> list
    assert d["payload"]["uniq"] == [9]  # set -> list, numpy scalar -> int
    assert d["meta"]["trace"] == [0, 1]
    json.dumps(d)  # fully serializable


def test_to_frame_excludes_ragged_container_field_without_raising(monkeypatch):
    @dataclass(frozen=True)
    class _R(AnalysisResult):
        _repr_fields = ("scalar", "ragged")
        scalar: float = 0.0
        ragged: tuple = ()

    monkeypatch.setitem(sys.modules, "pandas", _pandas_stub())
    r = _R(scalar=1.0, ragged=(1, [2, 3]))  # inhomogeneous -> np.ndim would have raised
    out = r.to_frame()
    assert list(out.columns) == ["scalar"]  # ragged tuple excluded, no raise


def _pandas_stub() -> types.ModuleType:
    """A minimal stand-in for the bits of pandas that ``to_frame`` touches."""

    class _Frame:
        def __init__(self, data=None):
            rows = data if isinstance(data, list) else ([] if data is None else [data])
            self._cols = list(rows[0].keys()) if rows else []
            self._n = len(rows)
            self.attrs: dict = {}

        @property
        def columns(self):
            return self._cols

        def __len__(self):
            return self._n

    mod = types.ModuleType("pandas")
    mod.DataFrame = _Frame
    return mod


def test_to_frame_builds_scalar_row_with_stub(monkeypatch):
    # Always-on coverage of the frame-building body (no real pandas needed).
    monkeypatch.setitem(sys.modules, "pandas", _pandas_stub())
    frame = _spectrum().to_frame()
    assert list(frame.columns) == ["kaplan_yorke"]
    assert "exponents" not in frame.columns  # arrays excluded
    assert len(frame) == 1
    assert frame.attrs["meta"]["system"] == "Lorenz"
