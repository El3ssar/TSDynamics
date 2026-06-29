"""Adapter base class shared by every Python benchmark adapter.

An adapter advertises its library identity / availability and, for each task it
supports, exposes a ``task_<key>(quick)`` method returning a zero-argument
callable that performs the task and returns the *comparable estimate* (a scalar
exponent / dimension / deviation / fixed-point coordinate), or ``None`` for a
speed-only task. A task the library cannot do is simply *not implemented* (no
``task_<key>`` method) and renders as a blank cell.

The callable is what the harness times (best-of-N); it must do the whole unit of
work on every call. The estimate it returns is captured once (outside timing) for
the precision table.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class BaseAdapter:
    """Common availability / dispatch machinery for a benchmark adapter."""

    name: str = "?"
    language: str = "python"

    def __init__(self, config: dict[str, Any]) -> None:
        self.cfg = config
        self._checked: tuple[bool, str, str] | None = None

    # -- availability ------------------------------------------------------- #

    def _probe(self) -> tuple[bool, str, str]:
        """Return ``(available, version, reason)``. Override per library."""
        raise NotImplementedError

    def _check(self) -> tuple[bool, str, str]:
        if self._checked is None:
            try:
                self._checked = self._probe()
            except Exception as exc:  # pragma: no cover - defensive
                self._checked = (False, "—", f"{type(exc).__name__}: {exc}")
        return self._checked

    @property
    def available(self) -> bool:
        return self._check()[0]

    @property
    def version(self) -> str:
        return self._check()[1]

    @property
    def reason(self) -> str:
        return self._check()[2]

    # -- dispatch ----------------------------------------------------------- #

    def build(self, task_key: str, *, quick: bool) -> Callable[[], Any] | None:
        """Return the zero-arg callable for ``task_key`` (``None`` ⇒ unsupported)."""
        method = getattr(self, f"task_{task_key}", None)
        if method is None:
            return None
        return method(quick)

    # -- shared config accessors -------------------------------------------- #

    @property
    def _ref(self) -> dict[str, Any]:
        return self.cfg["references"]

    @property
    def _intg(self) -> dict[str, Any]:
        return self.cfg["integration"]
