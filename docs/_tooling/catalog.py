"""
Catalogue merge layer for the TSDynamics docs site.

The **registry is the single source of truth** for every fact about a system —
name, family, category (module stem), dimension, parameters, component
``variables``, literature ``reference`` / ``doi``, ``known_lyapunov``, and the
``_field_shape`` of a spatially-extended field.  This module walks
:func:`tsdynamics.registry.all_systems`, instantiates each class to resolve its
*effective* dimension (variable-dim fields like Lorenz-96 / Gray–Scott report a
``None`` static dim but a concrete one once built), and merges the optional
:mod:`editorial.json` decoration *over* those facts.

Editorial decoration is purely additive and never authoritative: it supplies
human display strings (subcategory / type labels, prose blurbs), parameter role
notes, plot-dt overrides, a 3-component projection pick for 4-D-plus non-field
systems, and behaviour tags.  Editorial keys that do not match a registry system
are ignored; registry systems with no editorial entry render undecorated.  No
computed quantity (Lyapunov spectra, fractal dimensions, Kaplan–Yorke) ever
lives in editorial — those are computed at build time elsewhere.

Public API
----------
``load_catalog()``        -> ``Catalog`` (all records + grouping helpers)
``systems_by_type()``     -> ``{family: {category: [SystemRecord, ...]}}``
``type_label(family)``    -> display label for a family (e.g. ``"ODEs"``)
``subcategory_label(cat)``-> display label for a registry category
``counts()``              -> nested + total counts derived from the registry

Importable standalone::

    sys.path.insert(0, "docs/_tooling"); import catalog; catalog.counts()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Editorial loading
# ---------------------------------------------------------------------------

_EDITORIAL_PATH = Path(__file__).with_name("editorial.json")

# Order in which families appear on the Systems index (matches the new IA).
_FAMILY_ORDER: tuple[str, ...] = ("ode", "dde", "sde", "map")

# Fallback family -> (display label, url slug) when editorial is missing.
_FAMILY_DEFAULTS: dict[str, tuple[str, str]] = {
    "ode": ("ODEs", "ode"),
    "dde": ("DDEs", "dde"),
    "sde": ("SDEs", "sde"),
    "map": ("Maps", "maps"),
}


@lru_cache(maxsize=1)
def _editorial() -> dict[str, Any]:
    """Load and cache ``editorial.json`` (empty dict if absent/unreadable)."""
    try:
        with _EDITORIAL_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _ed_section(key: str) -> dict[str, Any]:
    section = _editorial().get(key, {})
    return section if isinstance(section, dict) else {}


# ---------------------------------------------------------------------------
# Display-label / type helpers (usable without building the whole catalogue)
# ---------------------------------------------------------------------------


def type_label(family: str) -> str:
    """Human display label for a family id (``"ode"`` -> ``"ODEs"``)."""
    labels = _ed_section("type_labels")
    if family in labels:
        return str(labels[family])
    return _FAMILY_DEFAULTS.get(family, (family.upper(), family))[0]


def type_slug(family: str) -> str:
    """URL slug for a family id (``"map"`` -> ``"maps"``)."""
    return _FAMILY_DEFAULTS.get(family, (family, family))[1]


def type_blurb(family: str) -> str:
    """Prose blurb for a family (empty string if none)."""
    return str(_ed_section("type_blurbs").get(family, ""))


def type_definition(family: str) -> str:
    """KaTeX-ready defining equation for a family (empty string if none)."""
    return str(_ed_section("type_definitions").get(family, ""))


def _humanize_category(category: str) -> str:
    """Last-resort label for an un-editorialised category (``snake_case``)."""
    return category.replace("_", " ").strip().capitalize()


def subcategory_label(category: str) -> str:
    """Human display label for a registry category (module stem)."""
    labels = _ed_section("subcategory_labels")
    if category in labels:
        return str(labels[category])
    return _humanize_category(category)


def subcategory_blurb(category: str) -> str:
    """Prose blurb for a registry category (empty string if none)."""
    return str(_ed_section("subcategory_blurbs").get(category, ""))


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemRecord:
    """A single rich, render-ready catalogue record.

    Registry facts (authoritative) live alongside the merged editorial
    decoration (all optional).  ``dim`` is the *effective* dimension — resolved
    by instantiating the class, so a variable-dim field reports its built size
    (e.g. Lorenz-96 -> 20) rather than the static ``None``.
    """

    # --- registry facts (source of truth) ---
    name: str
    family: str
    category: str
    dim: int | None
    params: dict[str, Any]
    variables: tuple[str, ...] | None
    reference: str | None
    doi: str | None
    known_lyapunov: dict[str, Any] | None
    field_shape: tuple[int, ...] | None
    field_labels: tuple[str, ...] | None
    module: str
    cls: Any

    # --- editorial decoration (optional) ---
    blurb: str = ""
    featured: bool = False
    param_roles: dict[str, str] = field(default_factory=dict)
    plot_dt: float | None = None
    projection: tuple[Any, ...] | None = None
    behavior: tuple[str, ...] = ()

    # --- derived display helpers ---
    @property
    def type_label(self) -> str:
        """Display label for this system's family (e.g. ``"ODEs"``)."""
        return type_label(self.family)

    @property
    def subcategory_label(self) -> str:
        """Display label for this system's registry category."""
        return subcategory_label(self.category)

    @property
    def is_field(self) -> bool:
        """A spatially-extended field (declares ``_field_shape``)."""
        return self.field_shape is not None

    @property
    def is_spatial(self) -> bool:
        """Render as a field/spacetime image rather than a comet.

        True for a declared field, or any non-field system with a state of
        four or more components (Lorenz-96 / KS / hyperchaotic / few-body) for
        which a 2-D/3-D comet would be misleading — unless an editorial
        ``projection`` deliberately picks three components to portray.
        """
        if self.is_field:
            return True
        return self.dim is not None and self.dim >= 4 and self.projection is None

    @property
    def tau(self) -> float | None:
        """The delay parameter of a DDE (its ``tau`` param), else ``None``."""
        if self.family != "dde":
            return None
        val = self.params.get("tau")
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view (drops the live class object)."""
        return {
            "name": self.name,
            "family": self.family,
            "type_label": self.type_label,
            "category": self.category,
            "subcategory_label": self.subcategory_label,
            "dim": self.dim,
            "params": dict(self.params),
            "variables": list(self.variables) if self.variables else None,
            "reference": self.reference,
            "doi": self.doi,
            "field_shape": list(self.field_shape) if self.field_shape else None,
            "field_labels": list(self.field_labels) if self.field_labels else None,
            "is_field": self.is_field,
            "is_spatial": self.is_spatial,
            "tau": self.tau,
            "blurb": self.blurb,
            "featured": self.featured,
            "param_roles": dict(self.param_roles),
            "plot_dt": self.plot_dt,
            "projection": list(self.projection) if self.projection else None,
            "behavior": list(self.behavior),
        }


@dataclass(frozen=True)
class Catalog:
    """The whole merged catalogue plus grouping/counting helpers."""

    records: tuple[SystemRecord, ...]

    # --- lookups ---
    def by_name(self, name: str) -> SystemRecord | None:
        """Return the record for a system name, or ``None`` if absent."""
        for rec in self.records:
            if rec.name == name:
                return rec
        return None

    def of_family(self, family: str) -> list[SystemRecord]:
        """All records belonging to a family id, in catalogue order."""
        return [r for r in self.records if r.family == family]

    # --- grouping ---
    def by_type(self) -> dict[str, dict[str, list[SystemRecord]]]:
        """``{family: {category: [records...]}}`` in IA order.

        Families follow :data:`_FAMILY_ORDER`; categories are sorted by their
        display label; records within a category by name.
        """
        out: dict[str, dict[str, list[SystemRecord]]] = {}
        families = list(_FAMILY_ORDER) + sorted(
            {r.family for r in self.records} - set(_FAMILY_ORDER)
        )
        for fam in families:
            fam_recs = self.of_family(fam)
            if not fam_recs:
                continue
            cats: dict[str, list[SystemRecord]] = {}
            for cat in sorted({r.category for r in fam_recs}, key=subcategory_label):
                cats[cat] = sorted((r for r in fam_recs if r.category == cat), key=lambda r: r.name)
            out[fam] = cats
        return out

    def featured(self) -> list[SystemRecord]:
        """Editorial-flagged headline systems, in family/name order."""
        order = {fam: i for i, fam in enumerate(_FAMILY_ORDER)}
        return sorted(
            (r for r in self.records if r.featured),
            key=lambda r: (order.get(r.family, 99), r.name),
        )

    # --- counts (all derived, never hardcoded) ---
    def counts(self) -> dict[str, Any]:
        """Nested + total counts straight from the records.

        ::

            {
              "total": 154,
              "by_family": {"ode": 120, "dde": 5, "sde": 3, "map": 26},
              "by_category": {"ode": {"chaotic_attractors": 47, ...}, ...},
            }
        """
        grouped = self.by_type()
        by_family = {fam: sum(len(v) for v in cats.values()) for fam, cats in grouped.items()}
        by_category = {
            fam: {cat: len(recs) for cat, recs in cats.items()} for fam, cats in grouped.items()
        }
        return {
            "total": len(self.records),
            "by_family": by_family,
            "by_category": by_category,
        }


# ---------------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------------


def _effective_dim(entry: Any) -> int | None:
    """Resolve the built (effective) dimension; fall back to the static one.

    Variable-dim systems (fields, Lorenz-96, MultiChua) report ``dim=None`` on
    the registry entry but a concrete dimension once instantiated.  Instantiation
    is cheap (no integration), so we always try it.
    """
    try:
        return entry.cls().dim  # build the default instance and ask
    except Exception:  # noqa: BLE001 — never let one odd class break the build
        return entry.dim


def _as_str_tuple(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    try:
        return tuple(str(v) for v in value)
    except TypeError:
        return None


def _merge_record(entry: Any, ed: dict[str, Any]) -> SystemRecord:
    cls = entry.cls
    variables = _as_str_tuple(getattr(cls, "variables", None))
    field_shape = getattr(cls, "_field_shape", None)
    field_shape = tuple(field_shape) if field_shape is not None else None
    field_labels = _as_str_tuple(getattr(cls, "field_labels", None))

    param_roles = ed.get("param_roles") if isinstance(ed.get("param_roles"), dict) else {}
    projection = ed.get("projection")
    projection = tuple(projection) if isinstance(projection, (list, tuple)) else None
    behavior = _as_str_tuple(ed.get("behavior")) or ()
    plot_dt = ed.get("plot_dt")
    try:
        plot_dt = float(plot_dt) if plot_dt is not None else None
    except (TypeError, ValueError):
        plot_dt = None

    return SystemRecord(
        name=entry.name,
        family=entry.family,
        category=entry.category,
        dim=_effective_dim(entry),
        params=dict(entry.params),
        variables=variables,
        reference=entry.reference,
        doi=getattr(entry, "doi", None),
        known_lyapunov=entry.known_lyapunov,
        field_shape=field_shape,
        field_labels=field_labels,
        module=getattr(entry, "module", ""),
        cls=cls,
        blurb=str(ed.get("blurb", "")),
        featured=bool(ed.get("featured", False)),
        param_roles={str(k): str(v) for k, v in param_roles.items()},
        plot_dt=plot_dt,
        projection=projection,
        behavior=behavior,
    )


@lru_cache(maxsize=1)
def load_catalog() -> Catalog:
    """Build (and cache) the full merged catalogue from the live registry."""
    from tsdynamics import registry

    ed_systems = _ed_section("systems")
    records: list[SystemRecord] = []
    for entry in registry.all_systems():
        ed = ed_systems.get(entry.name, {})
        ed = ed if isinstance(ed, dict) else {}
        records.append(_merge_record(entry, ed))
    records.sort(
        key=lambda r: (
            _FAMILY_ORDER.index(r.family) if r.family in _FAMILY_ORDER else 99,
            r.category,
            r.name,
        )
    )
    return Catalog(records=tuple(records))


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (the documented public surface)
# ---------------------------------------------------------------------------


def systems_by_type() -> dict[str, dict[str, list[SystemRecord]]]:
    """``{family: {category: [records...]}}`` for the whole catalogue."""
    return load_catalog().by_type()


def counts() -> dict[str, Any]:
    """Registry-derived counts (total + per family + per category)."""
    return load_catalog().counts()


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import json as _json

    cat = load_catalog()
    print(_json.dumps(cat.counts(), indent=2))
    print(f"featured: {[r.name for r in cat.featured()]}")
