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
The serialized text is a JSON object with two top-level keys::

    {
        "schema_version": 2,
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
- ``"spec"`` — the exact output of :meth:`Plot.to_dict`; rebuilt with
  :meth:`Plot.from_dict`.

:func:`from_json` (:func:`to_json` (spec)) reproduces the spec — every layer,
axis, annotation, color range, and ``meta`` field survives, and array data
returns as :class:`numpy.ndarray`.
"""

from __future__ import annotations

import json
from typing import Any

from .spec import Plot, PlotSpec

__all__ = [
    "SCHEMA_VERSION",
    "from_dict_envelope",
    "from_json",
    "to_dict_envelope",
    "to_json",
]

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
#:   3 — web-export overhaul (stream VIZ-WEB-EXPORT).  This bump is about the
#:       **threejs geometry payload**, which stamps the same version: it gained a
#:       vertex cap (``metadata.resample``) and shed three redundant blocks — a
#:       ``"line"`` geometry no longer carries an ``indices`` buffer (a polyline's
#:       vertex order is its draw order), a scalar ``"c"`` channel replaces the
#:       pre-expanded per-vertex ``"colors"`` RGB, and ``metadata.units`` (which
#:       carried axis *tickformats*, not units, and which no consumer read) is
#:       gone.  A v2 threejs payload still loads in the reference loader, which
#:       accepts both spellings.  The ``PlotSpec`` envelope itself is unchanged,
#:       so :func:`from_json` reads a v1 / v2 / v3 document identically.
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
SCHEMA_VERSION = 3

#: The envelope key carrying the integer schema version.
_VERSION_KEY = "schema_version"

#: The envelope key carrying the :meth:`PlotSpec.to_dict` payload.
_SPEC_KEY = "spec"


def to_dict_envelope(spec: PlotSpec) -> dict[str, Any]:
    """Wrap ``spec`` in the versioned envelope as a JSON-friendly mapping.

    Parameters
    ----------
    spec : Plot
        The spec to serialize.

    Returns
    -------
    dict
        ``{"schema_version": SCHEMA_VERSION, "spec": spec.to_dict()}`` — a nested
        mapping of plain ``str`` / ``float`` / ``list`` values, ready for
        :func:`json.dumps`.
    """
    return {_VERSION_KEY: SCHEMA_VERSION, _SPEC_KEY: spec.to_dict()}


def from_dict_envelope(envelope: dict[str, Any]) -> Plot:
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
        If the mapping is neither a recognizable envelope nor a bare spec dict.
    """
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
    return json.dumps(to_dict_envelope(spec), indent=indent)


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
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"not a PlotSpec JSON payload: expected a JSON object, got {type(loaded).__name__}."
        )
    return from_dict_envelope(loaded)
