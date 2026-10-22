"""Versioned JSON (de)serialization of a :class:`~tsdynamics.viz.spec.Plot`.

A :class:`~tsdynamics.viz.spec.Plot` already round-trips losslessly through
its :meth:`~tsdynamics.viz.spec.Plot.to_dict` /
:meth:`~tsdynamics.viz.spec.Plot.from_dict` pair (NumPy arrays become nested
lists and back).  This module wraps that dict in a small **versioned envelope**
and serializes it to / from a JSON ``str`` with the standard-library
:mod:`json` — no third-party dependency, so a spec computed on one machine can
be cached, shipped to a web frontend, or replayed without re-running the
analysis, and without a plotting library installed.

The envelope is the seam through which the schema can evolve without breaking
old payloads.

Schema
------
The serialized text is a JSON object with two required top-level keys::

    {
        "schema_version": 4,
        "spec": { ... Plot.to_dict() ... }
    }

- ``"schema_version"`` — an integer (currently :data:`SCHEMA_VERSION`) stamping
  the envelope layout, so a future change to the spec serialization can be
  migrated on read.  :func:`from_json` tolerates a *missing* or *older*
  ``"schema_version"`` (back-compat): a bare ``Plot.to_dict()`` object — one
  that has no ``"schema_version"`` key but does carry the spec's own ``"kind"`` /
  ``"layers"`` keys — is read as an *unversioned* (legacy) payload.  A v1 payload
  (produced before the styling overhaul) loads cleanly: the new fields
  (``theme``, enriched ``Axis``/``Legend``/``Colorbar``) are absent → default
  values (``theme=None``, ``grid=None``, etc.).
- ``"spec"`` — the output of :meth:`Plot.to_dict`, with nonfinite numbers
  replaced by ``null``; rebuilt with :meth:`Plot.from_dict`.
- ``"nonfinite"`` — an optional list of paths and values (``"nan"``,
  ``"+inf"`` or ``"-inf"``) restoring those numbers. Paths are relative to
  ``"spec"``. Ordinary ``null`` values and literal metadata are unaffected.

:func:`from_json` (:func:`to_json` (spec)) reproduces the spec — every layer,
axis, annotation, color range, and ``meta`` field survives, and array data
returns as :class:`numpy.ndarray`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ._visibility import listing_dir
from .spec import Plot, PlotSpec

__all__ = [
    "SCHEMA_VERSION",
    "from_dict_envelope",
    "from_json",
    "to_dict_envelope",
    "to_json",
]

__dir__ = listing_dir(__all__)

#: The current JSON-envelope schema version stamped by :func:`to_json` /
#: :func:`to_dict_envelope`.  Bumped only when the *envelope* or the underlying
#: :meth:`PlotSpec.to_dict` layout changes in a way that needs migration on read.
#:
#: Revision history:
#:   1 — initial versioned envelope (``{"schema_version": 1, "spec": {...}}``).
#:   2 — styling/theming overhaul: ``PlotSpec.theme`` field added (a
#:       :class:`~tsdynamics.viz.style.Theme` dict or ``null``); enriched
#:       :class:`~tsdynamics.viz.spec.Axis` fields (``grid``, ``color``,
#:       ``label_size``, ``tick_size``, ``tick_rotation``); enriched
#:       :class:`~tsdynamics.viz.spec.Legend` fields (``font_size``, ``ncol``,
#:       ``frame``); enriched :class:`~tsdynamics.viz.spec.Colorbar` field
#:       (``label_size``).  All new fields are optional with
#:       :meth:`~tsdynamics.viz.spec.PlotSpec.from_dict`-tolerated defaults, so a
#:       v1 payload loads without error (old fields are absent → default values).
#:   3 — web-export overhaul (stream VIZ-WEB-EXPORT).  This bump was about the
#:       **threejs geometry payload**, which originally shared this version: it gained a
#:       vertex cap (``metadata.resample``) and shed three redundant blocks — a
#:       ``"line"`` geometry no longer carries an ``indices`` buffer (a polyline's
#:       vertex order is its draw order), a scalar ``"c"`` channel replaces the
#:       pre-expanded per-vertex ``"colors"`` RGB, and ``metadata.units`` (which
#:       carried axis *tickformats*, not units, and which no consumer read) is
#:       gone.  A v2 threejs payload still loads in the reference loader, which
#:       accepts both spellings.  The ``PlotSpec`` envelope itself is unchanged,
#:       so :func:`from_json` reads a v1 / v2 / v3 document identically.
#:   4 — standards-compliant nonfinite numbers: ``null`` placeholders plus an
#:       envelope sidecar restore NaN and signed infinities without colliding
#:       with user metadata. The threejs geometry schema remains independently 3.
#:
#: **Not a bump — the v6 vocabulary narrowing.**  The v6 surgery removed seven
#: never-produced semantic kinds from :class:`~tsdynamics.viz.spec.PlotKind`
#: (``power_spectrum`` / ``spectrogram`` / ``histogram_null`` / ``feature_bars`` /
#: ``complexity_curve`` / ``trajectory_animation`` / ``ensemble_animation``; see
#: ``tests/test_viz_vocab.py``).  The *envelope* is untouched and every field is
#: read exactly as before, so the version does not move.  The one visible
#: consequence: an old document whose ``"kind"`` names one of the seven now fails
#: :func:`from_dict_envelope` with ``ValueError: '<kind>' is not a valid PlotKind``
#: rather than loading.  That is the intended, loud behaviour for a sanctioned v6
#: break — nothing in any released version ever *wrote* one of those kinds, so a
#: real payload carrying one would have to be hand-authored.
SCHEMA_VERSION = 4

#: The envelope key carrying the integer schema version.
_VERSION_KEY = "schema_version"

#: The envelope key carrying the :meth:`PlotSpec.to_dict` payload.
_SPEC_KEY = "spec"

_NONFINITE_KEY = "nonfinite"
_NONFINITE_VALUES = {"nan": math.nan, "+inf": math.inf, "-inf": -math.inf}


def _encode_nonfinite(value: Any, path: list[str | int], records: list[dict[str, Any]]) -> Any:
    """Replace nonfinite numbers without introducing reserved tags into user data."""
    if isinstance(value, float) and not math.isfinite(value):
        word = "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"
        records.append({"path": path, "value": word})
        return None
    if isinstance(value, Mapping):
        return {key: _encode_nonfinite(item, [*path, key], records) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [
            _encode_nonfinite(item, [*path, index], records) for index, item in enumerate(value)
        ]
    return value


def _restore_nonfinite(spec: Mapping[str, Any], records: Any) -> dict[str, Any]:
    """Restore only validated null locations in a private copy of the spec."""
    if not isinstance(records, list):
        raise ValueError("Plot nonfinite sidecar must be a list of path/value records.")
    restored = deepcopy(dict(spec))
    seen: set[tuple[str | int, ...]] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"path", "value"}:
            raise ValueError("Each Plot nonfinite record must contain exactly 'path' and 'value'.")
        path, word = record["path"], record["value"]
        if (
            not isinstance(path, list)
            or not path
            or any(type(part) not in (str, int) for part in path)
            or not isinstance(word, str)
            or word not in _NONFINITE_VALUES
        ):
            raise ValueError(
                "Plot nonfinite records need a nonempty path and value nan, +inf or -inf."
            )
        if tuple(path) in seen:
            raise ValueError("Plot nonfinite sidecar contains a duplicate path.")
        seen.add(tuple(path))
        target: Any = restored
        for index, part in enumerate(path):
            container: Any = target
            if (
                isinstance(target, dict)
                and isinstance(part, str)
                and part in target
                or isinstance(target, list)
                and type(part) is int
                and 0 <= part < len(target)
            ):
                child = container[part]
            else:
                raise ValueError("Plot nonfinite sidecar path does not address an existing value.")
            if index == len(path) - 1:
                if child is not None:
                    raise ValueError("Plot nonfinite sidecar paths must address null placeholders.")
                container[part] = _NONFINITE_VALUES[word]
            else:
                target = child
    return restored


def to_dict_envelope(spec: PlotSpec) -> dict[str, Any]:
    """Wrap ``spec`` in the versioned envelope as a JSON-friendly mapping.

    Parameters
    ----------
    spec : Plot
        The spec to serialize.

    Returns
    -------
    dict
        A nested mapping ready for strict :func:`json.dumps`, with an optional
        ``nonfinite`` sidecar preserving NaN and signed infinities.
    """
    records: list[dict[str, Any]] = []
    envelope = {
        _VERSION_KEY: SCHEMA_VERSION,
        _SPEC_KEY: _encode_nonfinite(spec.to_dict(), [], records),
    }
    if records:
        envelope[_NONFINITE_KEY] = records
    return envelope


def from_dict_envelope(envelope: Mapping[str, Any]) -> Plot:
    """Rebuild a :class:`Plot` from a (possibly unversioned) envelope mapping.

    Tolerates a missing / older ``"schema_version"`` for back-compat: an envelope
    with a ``"spec"`` key uses it; a bare :meth:`Plot.to_dict` mapping (no
    envelope wrapper — it carries the spec's own ``"kind"`` key directly) is read
    as a legacy unversioned payload.

    Parameters
    ----------
    envelope : dict
        The mapping produced by :func:`to_dict_envelope`, or a bare
        :meth:`Plot.to_dict` mapping (legacy / unversioned).

    Returns
    -------
    Plot

    Raises
    ------
    ValueError
        If the mapping is neither a recognizable envelope nor a bare spec dict,
        or its schema version is invalid or newer than this library supports.
    """
    if not isinstance(envelope, Mapping):
        raise ValueError("not a Plot JSON payload: expected a mapping.")
    if _VERSION_KEY in envelope:
        version = envelope[_VERSION_KEY]
        if type(version) is not int or version < 1:
            raise ValueError("Plot schema_version must be a positive integer.")
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"Plot schema version {version} is newer than the supported version "
                f"{SCHEMA_VERSION}; upgrade tsdynamics to load this document."
            )
    if _SPEC_KEY in envelope:
        spec_dict = envelope[_SPEC_KEY]
    elif "kind" in envelope:
        # A bare PlotSpec.to_dict() mapping written before the envelope existed.
        spec_dict = envelope
    else:
        raise ValueError(
            "not a PlotSpec JSON payload: expected a 'spec' envelope key or a "
            "bare PlotSpec.to_dict() mapping carrying a 'kind' key."
        )
    if not isinstance(spec_dict, Mapping) or "kind" not in spec_dict:
        raise ValueError("not a Plot JSON payload: 'spec' must be a plot mapping carrying 'kind'.")
    if _SPEC_KEY in envelope and _NONFINITE_KEY in envelope:
        if envelope.get(_VERSION_KEY) != 4:
            raise ValueError("Plot nonfinite sidecar requires schema_version 4.")
        spec_dict = _restore_nonfinite(spec_dict, envelope[_NONFINITE_KEY])
    return PlotSpec.from_dict(spec_dict)


def to_json(spec: PlotSpec, *, indent: int | None = None) -> str:
    """Serialize ``spec`` to a versioned JSON ``str``.

    Parameters
    ----------
    spec : Plot
        The spec to serialize.
    indent : int, optional
        Passed to :func:`json.dumps`; ``None`` (default) emits compact JSON,
        a positive integer pretty-prints with that indent.

    Returns
    -------
    str
        A JSON document ``{"schema_version": ..., "spec": ...}`` that
        :func:`from_json` reads back into an equal spec.
    """
    return json.dumps(to_dict_envelope(spec), indent=indent, allow_nan=False)


def from_json(text: str) -> Plot:
    """Rebuild a :class:`Plot` from JSON produced by :func:`to_json`.

    Tolerates a missing / older ``"schema_version"`` (back-compat) by delegating
    to :func:`from_dict_envelope`: both a current envelope and a legacy bare
    :meth:`Plot.to_dict` document load.

    Parameters
    ----------
    text : str
        A JSON document — an envelope from :func:`to_json`, or a bare
        :meth:`Plot.to_dict` document.

    Returns
    -------
    Plot

    Raises
    ------
    ValueError
        If ``text`` is not valid JSON, or not a recognizable spec payload.
    """
    constants: list[str] = []

    def legacy_constant(word: str) -> float:
        constants.append(word)
        return {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}[word]

    loaded = json.loads(text, parse_constant=legacy_constant)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"not a PlotSpec JSON payload: expected a JSON object, got {type(loaded).__name__}."
        )
    if loaded.get(_VERSION_KEY) == 4 and constants:
        raise ValueError(
            "Plot schema 4 requires standard JSON; encode nonfinite values in its sidecar."
        )
    return from_dict_envelope(loaded)
