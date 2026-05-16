"""
Single-source-of-truth registration for trajectory operations.

Every analysis primitive — decimation, derivatives, peak finding, event
detection, Poincaré sections, return maps — is defined exactly *once* as a
free function with the universal calling convention

    fn(t, y, *args, **kwargs) -> result

and decorated with :func:`trajectory_op`.  The decorator:

1. Wraps the function so it accepts a :class:`Trajectory`, a ``(t, y)``
   tuple, *or* bare ``t, y`` ndarrays as its leading argument(s).  This
   is the public free-function form.
2. Appends an entry to a module-level registry.  When :class:`Trajectory`
   is defined, :func:`install_methods` drains the registry and installs
   one method per registered op on the class.  The method strips
   ``self.t, self.y`` from its arguments and forwards to the original
   function, then wraps the result the same way the polymorphic
   free-function form does.

The net effect: **one decorated function gives you both**

    decimate(traj, every=5)           # free function, Trajectory input
    decimate((t, y), every=5)         # free function, (t, y) tuple
    decimate(t, y, every=5)           # free function, bare arrays
    traj.decimate(every=5)            # fluent method

There is no duplicated docstring, no wrapper class, no manual ``self.t,
self.y`` extraction anywhere downstream.  Adding the next analysis
primitive is one decorator + one function.

Wrapping behaviour
------------------

``returns`` declares what the function's return value represents:

- ``"trajectory"`` — the function returns ``(t_new, y_new)``.  Both
  forms wrap it as a fresh ``Trajectory``; the back-reference
  ``Trajectory.system`` is inherited from the input Trajectory or set to
  ``None`` when the caller supplied raw arrays / a ``(t, y)`` tuple.
- ``"ndarray_keep_t"`` — the function returns an ndarray ``y_new`` with
  the same time axis as the input.  Both forms return
  ``Trajectory(t_in.copy(), y_new, system)``.  (Currently only
  :func:`derivative`.)
- ``"passthrough"`` — the function returns an arbitrary object that
  should not be reinterpreted (e.g. ``EventResult``, ``ReturnMap``,
  ``(t_peaks, y_peaks)``, scalar arrays, dicts).  No wrapping is ever
  applied — neither free-function nor method form touches the value.

The returned Trajectory supports tuple-unpacking (``t, y = result``) so
call sites that previously assigned ``t, y = fn(...)`` still work.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

#: How to wrap a registered function's return value.  See module docstring.
ReturnShape = Literal["trajectory", "ndarray_keep_t", "passthrough"]

_Entry = tuple[str, Callable[..., Any], ReturnShape]
_REGISTRY: list[_Entry] = []
_INSTALLED_INTO: set[type] = set()

#: The Trajectory class, set by :func:`install_methods` so the polymorphic
#: free-function form can always rewrap ``(t, y)`` results as Trajectory
#: regardless of input type.  Before :func:`install_methods` runs (e.g.
#: when ``_trajectory_ops`` is imported in isolation from a unit test) the
#: polymorphic form falls back to returning the raw ``(t, y)`` tuple.
_TRAJECTORY_CLS: type | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def trajectory_op(
    *,
    returns: ReturnShape = "trajectory",
    name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorate ``fn(t, y, *args, **kw)`` as a polymorphic trajectory operation.

    Parameters
    ----------
    returns : {"trajectory", "ndarray_keep_t", "passthrough"}, default ``"trajectory"``
        How to wrap the function's return value.  See module docstring.
    name : str, optional
        Method name to install on :class:`Trajectory`.  Defaults to
        ``fn.__name__``.

    Returns
    -------
    polymorphic : callable
        A wrapper around ``fn`` that accepts a :class:`Trajectory`, a
        ``(t, y)`` tuple, or bare ``t, y`` arrays as its leading argument(s).
        Calling ``polymorphic.__wrapped__`` (set by :func:`functools.update_wrapper`)
        recovers the raw ``(t, y, ...)`` function.

    Examples
    --------
    >>> @trajectory_op(returns="trajectory")
    ... def decimate(t, y, every):
    ...     return t[::every].copy(), y[::every].copy()
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _check_signature(fn)
        method_name = name or fn.__name__
        polymorphic = _make_polymorphic(fn, returns)
        functools.update_wrapper(polymorphic, fn)
        _REGISTRY.append((method_name, fn, returns))
        return polymorphic

    return decorator


def install_methods(traj_cls: type) -> None:
    """
    Install one method per registered op on ``traj_cls``.

    Also registers ``traj_cls`` as *the* Trajectory class so the
    polymorphic free-function form can always wrap ``(t, y)`` results as
    Trajectory objects regardless of how the user supplied the input.

    Called once from :mod:`tsdynamics.base.base` after :class:`Trajectory`
    is defined.  Idempotent per ``traj_cls``: calling it twice on the same
    class is a no-op (so module reloads don't blow up).
    """
    global _TRAJECTORY_CLS
    _TRAJECTORY_CLS = traj_cls
    if traj_cls in _INSTALLED_INTO:
        return
    for method_name, fn, returns in _REGISTRY:
        method = _make_method(fn, returns)
        method.__name__ = method_name
        method.__qualname__ = f"{traj_cls.__name__}.{method_name}"
        method.__doc__ = fn.__doc__
        _adjust_method_signature(method, fn)
        setattr(traj_cls, method_name, method)
    _INSTALLED_INTO.add(traj_cls)


def registered_ops() -> list[str]:
    """Return the names of all currently registered ops (for introspection)."""
    return [name for name, *_ in _REGISTRY]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _check_signature(fn: Callable[..., Any]) -> None:
    """Ensure ``fn``'s first two positional params are named ``t`` and ``y``."""
    sig = inspect.signature(fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(params) < 2 or params[0].name != "t" or params[1].name != "y":
        raise TypeError(
            f"@trajectory_op: {fn.__name__} must take (t, y) as its first two "
            f"positional arguments; got signature {sig}.  This is the universal "
            "convention so the decorator can extract them from a Trajectory."
        )


def _is_trajectory_like(obj: Any) -> bool:
    """Duck-type Trajectory: has ``.t``, ``.y``, ``.system`` attributes."""
    return hasattr(obj, "t") and hasattr(obj, "y") and hasattr(obj, "system")


def _wrap_result(
    result: Any,
    returns: ReturnShape,
    t_in: np.ndarray,
    sys_ref: Any,
) -> Any:
    """
    Wrap a function's return value uniformly across all input shapes.

    The ``passthrough`` case is the identity.  ``trajectory`` and
    ``ndarray_keep_t`` build a fresh Trajectory using the registered
    class — falling back to the raw value when no class is registered
    (e.g. in isolated unit tests that import :mod:`_trajectory_ops`
    without going through :mod:`tsdynamics.base`).
    """
    if returns == "passthrough":
        return result
    if _TRAJECTORY_CLS is None:
        return result
    if returns == "trajectory":
        t_new, y_new = result
        return _TRAJECTORY_CLS(t_new, y_new, sys_ref)
    if returns == "ndarray_keep_t":
        return _TRAJECTORY_CLS(t_in.copy(), result, sys_ref)
    raise ValueError(f"unknown returns={returns!r}")  # pragma: no cover


def _make_polymorphic(fn: Callable[..., Any], returns: ReturnShape) -> Callable[..., Any]:
    """Wrap ``fn(t, y, ...)`` to also accept Trajectory or (t, y) tuple."""

    def polymorphic(*args: Any, **kwargs: Any) -> Any:
        if not args:
            raise TypeError(f"{fn.__name__}: missing trajectory argument")

        first = args[0]
        if _is_trajectory_like(first):
            t = np.asarray(first.t)
            y = np.asarray(first.y)
            result = fn(t, y, *args[1:], **kwargs)
            return _wrap_result(result, returns, t, first.system)

        if isinstance(first, tuple) and len(first) == 2 and not isinstance(first[0], int):
            t = np.asarray(first[0])
            y = np.asarray(first[1])
            result = fn(t, y, *args[1:], **kwargs)
            return _wrap_result(result, returns, t, None)

        # Bare (t, y) two-argument form.
        if len(args) < 2:
            raise TypeError(f"{fn.__name__}: need a Trajectory, a (t, y) tuple, or two arrays")
        t = np.asarray(args[0])
        y = np.asarray(args[1])
        result = fn(t, y, *args[2:], **kwargs)
        return _wrap_result(result, returns, t, None)

    return polymorphic


def _make_method(fn: Callable[..., Any], returns: ReturnShape) -> Callable[..., Any]:
    """Build a method that forwards ``(self.t, self.y, *args, **kw)`` to ``fn``."""
    if returns == "trajectory":

        def method(self: Any, *args: Any, **kwargs: Any) -> Any:
            t_new, y_new = fn(self.t, self.y, *args, **kwargs)
            return type(self)(t_new, y_new, self.system)

    elif returns == "ndarray_keep_t":

        def method(self: Any, *args: Any, **kwargs: Any) -> Any:
            y_new = fn(self.t, self.y, *args, **kwargs)
            return type(self)(self.t.copy(), y_new, self.system)

    elif returns == "passthrough":

        def method(self: Any, *args: Any, **kwargs: Any) -> Any:
            return fn(self.t, self.y, *args, **kwargs)

    else:
        raise ValueError(f"unknown returns={returns!r}")  # pragma: no cover

    return method


def _adjust_method_signature(method: Callable[..., Any], fn: Callable[..., Any]) -> None:
    """
    Strip ``t`` and ``y`` from the original signature and prepend ``self``.

    So ``help(traj.decimate)`` shows ``decimate(self, every)`` rather than
    the underlying ``decimate(t, y, every)``.
    """
    sig = inspect.signature(fn)
    kept = [p for p in sig.parameters.values() if p.name not in ("t", "y")]
    self_param = inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    method.__signature__ = sig.replace(parameters=[self_param, *kept])  # type: ignore[attr-defined]


__all__ = ["ReturnShape", "install_methods", "registered_ops", "trajectory_op"]
