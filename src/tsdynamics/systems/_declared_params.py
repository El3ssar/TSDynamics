"""The parameter merge a **variable-dimension** system's ``__init__`` must perform.

Five catalogue systems resolve their own ``dim`` from a structural parameter —
``GrayScott`` / ``SwiftHohenberg`` / ``KuramotoSivashinsky`` from ``N``,
``Lorenz96`` from ``N``, ``MultiChua`` from ``n_circuits`` — so each needs a
custom ``__init__`` that computes ``dim`` before calling ``super().__init__``.
That constructor exists *only* to do the arithmetic, and it must not become a
**second, laxer front door** than :meth:`SystemBase.__init__`.  Each hand-rolled
copy had drifted into being exactly that, in two ways:

**A parameter given twice was silently preferred, not refused.**
:meth:`SystemBase.__init__` refuses a name passed both in ``params=`` and as a
keyword — "there is deliberately no precedence rule".  These constructors merged
their channels *before* calling ``super()``, so the base never saw the
duplication: ``KuramotoSivashinsky(L=40.0, params={"L": 22.0})`` built at
``L = 40.0`` with no word said.

**``dim=`` and ``field_shape=`` were rejected as unknown parameters.**
:meth:`SystemBase.with_params` and :meth:`SystemBase.copy` forward both on every
rebuild.  A free ``**param_kwargs`` swallowed them, the unknown-name check then
refused them, and so ``with_params`` / ``copy`` — and with them continuation,
orbit diagrams and every parameter sweep — raised on all five systems.  Both are
*derived* here, so they are accepted and discarded.

One merge, one set of rules, five callers.  This module is private and its
contents are in no ``__all__``, so the catalogue modules' curated ``__dir__``
(and the gates over it) are unaffected.
"""

from __future__ import annotations

from tsdynamics.errors import InvalidParameterError

__all__ = ["merge_declared_params"]


def merge_declared_params(
    cls_name: str,
    defaults: dict[str, float],
    params: dict[str, float] | None,
    param_kwargs: dict[str, float],
    named: dict[str, float | None],
) -> dict[str, float]:
    """Merge a system's three parameter channels the way the base constructor does.

    Parameters
    ----------
    cls_name : str
        The system class name, quoted in every message.
    defaults : dict
        The class's declared ``params`` (a fresh copy — this is mutated).
    params : dict or None
        The caller's ``params=`` mapping.
    param_kwargs : dict
        The caller's free ``**param_kwargs``.  ``dim`` / ``field_shape`` must be
        bound by the constructor's own signature and **not** reach here.
    named : dict
        The constructor's own explicit arguments, ``None`` where omitted.  A
        non-``None`` entry that is *also* in ``params=`` or ``**param_kwargs``
        is the same duplication wearing a positional hat.

    Returns
    -------
    dict
        The merged parameters, ready for ``super().__init__(params=...)``.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If a parameter is given twice through two different channels, or if a
        name is not one the class declares.
    """
    overrides: dict[str, float] = dict(params) if params else {}
    if param_kwargs:
        duplicated = sorted(set(param_kwargs) & set(overrides))
        if duplicated:
            raise InvalidParameterError(
                f"{cls_name}: parameter(s) {duplicated} given twice — "
                f"once in params= and once as a keyword. Pass each parameter "
                f"exactly once (there is deliberately no precedence rule)."
            )
        overrides.update(param_kwargs)

    given = sorted(name for name, value in named.items() if value is not None)
    both = sorted(set(given) & set(overrides))
    if both:
        spellings = ", ".join(f"{cls_name}({name}=...)" for name in both)
        raise InvalidParameterError(
            f"{cls_name}: parameter(s) {both} given twice — once as a named "
            f"constructor argument ({spellings}) and once in params= / as a "
            f"keyword. Pass each parameter exactly once (there is deliberately "
            f"no precedence rule)."
        )

    merged = dict(defaults)
    if overrides:
        unknown = set(overrides) - set(merged)
        if unknown:
            raise InvalidParameterError(
                f"{cls_name}: unknown parameter(s) {sorted(unknown)}. Declared: {sorted(merged)}"
            )
        merged.update(overrides)
    for name in given:
        value = named[name]
        assert value is not None
        merged[name] = value
    return merged
