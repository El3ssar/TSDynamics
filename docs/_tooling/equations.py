"""
Render system equations as LaTeX, straight from their symbolic definitions.

The docs build calls :func:`equations_markdown` per registered system.  The
renderer chain degrades gracefully — symbolic LaTeX where possible, the
docstring's math block if present, a source-code fence otherwise — so a
single exotic system can never break a ``--strict`` docs build.
"""

from __future__ import annotations

import inspect
import textwrap

_MAX_RENDER_DIM = 8  # variable-dim/loop systems read better as source code


def _unwrap(obj):
    fn = getattr(obj, "__func__", obj)
    return getattr(fn, "py_func", fn)


def _state_names(cls, dim: int) -> list[str]:
    names = getattr(cls, "variables", None)
    if names and len(names) == dim:
        return list(names)
    return [f"y_{{{i}}}" for i in range(dim)]


def _param_symbols(sys_obj):
    import symengine

    cls = type(sys_obj)
    structural = getattr(cls, "_structural_params", frozenset())
    out = {}
    for k, v in sys_obj.params.items():
        out[k] = v if k in structural else symengine.Symbol(k)
    return out


def _aligned(lines: list[str]) -> str:
    body = " \\\\\n".join(lines)
    return f"$$\n\\begin{{aligned}}\n{body}\n\\end{{aligned}}\n$$"


def _ode_latex(entry) -> str:
    import symengine
    import sympy

    sys_obj = entry.cls()
    dim = sys_obj.dim
    if dim is None or dim > _MAX_RENDER_DIM:
        raise ValueError("dimension too large for symbolic rendering")

    names = _state_names(entry.cls, dim)
    syms = [symengine.Symbol(n) for n in names]
    t = symengine.Symbol("t")

    def y(i):
        return syms[int(i)]

    exprs = list(entry.cls._equations(y, t, **_param_symbols(sys_obj)))
    lines = [
        rf"\dot{{{n}}} &= {sympy.latex(symengine.sympify(e)._sympy_())}"
        for n, e in zip(names, exprs, strict=True)
    ]
    return _aligned(lines)


def _dde_latex(entry) -> str:
    import symengine
    import sympy

    sys_obj = entry.cls()
    dim = sys_obj.dim
    if dim is None or dim > _MAX_RENDER_DIM:
        raise ValueError("dimension too large for symbolic rendering")

    names = _state_names(entry.cls, dim)
    syms = [symengine.Symbol(n) for n in names]
    t = symengine.Symbol("t")

    def y(i, time=None):
        if time is None:
            return syms[int(i)]
        return symengine.Function(names[int(i)])(time)

    exprs = list(entry.cls._equations(y, t, **_param_symbols(sys_obj)))
    lines = [
        rf"\dot{{{n}}} &= {sympy.latex(symengine.sympify(e)._sympy_())}"
        for n, e in zip(names, exprs, strict=True)
    ]
    return _aligned(lines)


def _map_latex(entry) -> str:
    import sympy

    sys_obj = entry.cls()
    dim = sys_obj.dim
    if dim is None or dim > _MAX_RENDER_DIM:
        raise ValueError("dimension too large for symbolic rendering")

    names = _state_names(entry.cls, dim)
    syms = [sympy.Symbol(n) for n in names]
    X = syms[0] if dim == 1 else syms  # 1-D maps unpack `x = X` directly
    params = [sympy.Symbol(k) for k in sys_obj.params]

    fn = _unwrap(entry.cls.__dict__["_step"])
    out = fn(X, *params)  # NumPy calls on sympy symbols raise → fallback
    out_list = [out] if dim == 1 and not isinstance(out, tuple | list) else list(out)
    lines = [
        rf"{n}' &= {sympy.latex(sympy.sympify(e))}" for n, e in zip(names, out_list, strict=True)
    ]
    return _aligned(lines)


def _source_fence(entry) -> str:
    method = "_step" if entry.family == "map" else "_equations"
    fn = _unwrap(entry.cls.__dict__[method])
    src = textwrap.dedent(inspect.getsource(fn))
    return f"```python\n{src}```"


def equations_markdown(entry) -> str:
    """Best-effort equations block for a registry entry (never raises)."""
    renderer = {"ode": _ode_latex, "dde": _dde_latex, "map": _map_latex}[entry.family]
    try:
        return renderer(entry)
    except Exception:  # noqa: BLE001 — any failure falls back to source
        try:
            return _source_fence(entry)
        except Exception:  # noqa: BLE001
            return "_Equations could not be rendered — see the source._"
