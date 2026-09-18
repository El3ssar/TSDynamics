"""The compatibility-matrix governance gate (stream P1).

The declared matrix — one ``primitives`` row per registered plot transform — is a
*promise*: every pair in it draws a real figure, and every pair outside it
raises.  A promise nothing checks is how a plotting library ends up advertising
plots it cannot produce, which is precisely the failure the v6 vocabulary
surgery removed seven ``PlotKind`` members to avoid.

So this module renders the whole matrix.  It is modelled on
``tests/test_viz_honoring_contract.py``, which already proved the pattern works
here for ``StyleKey.honored_by``: **an overclaim cannot ship green.**

What it checks, in the order the plan lists them:

1. every **declared** cell builds, renders, and produces the artist class the
   primitive claims, carrying finite data (non-degenerate, not merely
   non-crashing);
2. every **undeclared** cell raises ``InvalidParameterError`` naming the valid
   set;
3. every **structural** claim holds — the primitive accepts the transform's
   coordinate space, and its required channels are present in the geometry;
4. the matrix's reading marks are real — ``*`` prints, and the vestigial ``!``
   (and the always-empty ``exclusive`` field behind it) stays deleted;
5. every registered transform's ``compute`` is callable and returns a
   ``Geometry`` — the check that stops the matrix advertising a plot the library
   cannot draw.

Plus the enabling invariant: every in-tree transform ships the small example
subject this gate drives it with, so **adding a transform is one registration
and no test edit**.

Cost note (the plan asks for it explicitly): the gate is O(cells) matplotlib
renders.  At today's 8 transforms / 21 declared cells that is fast; when the
matrix reaches the planned ~175 cells, split it — a per-transform sample in the
fast tier, the full matrix in the slow/nightly tier.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from tsdynamics.errors import InvalidParameterError  # noqa: E402
from tsdynamics.viz.render import register_builtin_renderers  # noqa: E402
from tsdynamics.viz.transforms import (  # noqa: E402
    PRIMITIVES,
    RESERVED_PRIMITIVES,
    Geometry,
    build_spec,
    compatibility,
    geometry,
    get,
    transforms,
)


@pytest.fixture(scope="module", autouse=True)
def _mpl_backend():
    from tsdynamics import registry

    register_builtin_renderers()
    if "matplotlib" not in registry.renderers:  # pragma: no cover - matplotlib present here
        pytest.skip("matplotlib backend did not register")
    yield


#: Which matplotlib artist container each primitive must fill.  This table is the
#: *test's* backend knowledge, deliberately not the primitive's: a primitive is
#: backend-neutral by construction, and pushing "which mpl container" onto the
#: record would be the first crack in that.
_MPL_CONTAINER: dict[str, str] = {
    "line": "lines",
    "line3d": "lines",
    "steps": "lines",
    "contour": "lines",
    "points": "collections",
    "points3d": "collections",
    "markers": "collections",
    "quiver": "collections",
    "surface3d": "collections",
    "band": "collections",
    "boundary": "collections",
    "image": "images",
    "density": "images",
    "bars": "patches",
    "histogram": "patches",
    "errorbars": "lines",
}


def _cells() -> list[tuple[str, str]]:
    """Every declared (transform, primitive) pair, sorted."""
    return sorted((t.name, p) for t in transforms() for p in t.primitives)


def _undeclared_cells() -> list[tuple[str, str]]:
    """Every (transform, primitive) pair the matrix does **not** declare."""
    return sorted((t.name, p) for t in transforms() for p in PRIMITIVES if p not in t.primitives)


def _artists(container: str, fig) -> list:
    """Collect the named artist container across every axes of ``fig``."""
    out: list = []
    for ax in fig.axes:
        out.extend(getattr(ax, container, []))
    return out


def _has_finite_data(artist) -> bool:
    """Whether an artist carries at least one finite number."""
    for accessor in ("get_xydata", "get_array", "get_offsets", "get_paths"):
        getter = getattr(artist, accessor, None)
        if getter is None:
            continue
        try:
            value = getter()
        except Exception:  # pragma: no cover - defensive
            continue
        if value is None:
            continue
        if accessor == "get_paths":
            return len(value) > 0
        data = np.asarray(getattr(value, "data", value), dtype=float)
        if data.size and np.isfinite(data).any():
            return True
    # A Rectangle (bar / histogram) has no array accessor; its geometry is enough.
    width = getattr(artist, "get_width", None)
    if width is not None:
        return np.isfinite(float(width()))
    return False  # pragma: no cover - defensive


# ---------------------------------------------------------------------------
# 1. Every DECLARED cell draws a real figure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "primitive"), _cells(), ids=lambda v: str(v))
def test_every_declared_cell_renders_non_degenerately(name, primitive):
    """A declared pair builds, renders, and puts finite data in the right artist.

    "It did not raise" is not the bar: a plot that draws nothing is the exact
    failure mode a spec-dict test cannot see.
    """
    record = get(name)
    assert record.example is not None, f"{name} declares no example (see the gate below)"
    subject, options = record.example(primitive)
    spec = build_spec(subject, name, primitive=primitive, **dict(options))

    container = _MPL_CONTAINER[primitive]
    fig = spec.render("matplotlib")
    drawn = _artists(container, fig)
    assert drawn, (
        f"transform {name!r} declares primitive {primitive!r}, but rendering it produced no "
        f"matplotlib {container}. A declared cell that cannot draw must not ship."
    )
    assert any(_has_finite_data(a) for a in drawn), (
        f"{name}.{primitive} drew {len(drawn)} {container} carrying no finite data."
    )


# ---------------------------------------------------------------------------
# 2. Every UNDECLARED cell raises, naming the valid set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "primitive"), _undeclared_cells(), ids=lambda v: str(v))
def test_every_undeclared_cell_raises_naming_the_valid_set(name, primitive):
    """An invalid pair raises — never a fallback, never a warning, never a wrong plot."""
    record = get(name)
    subject, options = record.example(record.default_primitive)
    with pytest.raises(InvalidParameterError) as excinfo:
        build_spec(subject, name, primitive=primitive, **dict(options))
    message = str(excinfo.value)
    assert primitive in message
    assert name in message
    for valid in record.primitives:
        assert valid in message, f"the error must name the valid set; {valid!r} is missing"


def test_the_invalid_pair_message_points_at_the_owning_transform():
    """When the requested primitive is another transform's default, say so.

    That is almost always what the caller actually wanted, and it turns a dead
    end into a redirection.
    """
    subject, options = get("time_series").example("line")
    with pytest.raises(InvalidParameterError, match="default primitive of transform"):
        build_spec(subject, "time_series", primitive="quiver", **dict(options))


def test_an_unregistered_primitive_says_so_rather_than_guessing():
    subject, options = get("time_series").example("line")
    with pytest.raises(InvalidParameterError, match="not a registered primitive at all"):
        build_spec(subject, "time_series", primitive="rainbow", **dict(options))


# ---------------------------------------------------------------------------
# 3. Structural claims
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "primitive"), _cells(), ids=lambda v: str(v))
def test_declared_cells_are_structurally_possible(name, primitive):
    """``transform.frame`` is drawable by the primitive, and its channels are present.

    Validity is *structural*, which is what keeps the rows short and checkable: a
    row cannot claim a pair the primitive itself refuses to draw.
    """
    record = get(name)
    prim = PRIMITIVES[primitive]
    for space in record.frame:
        assert prim.accepts_frame(space), (
            f"{name}.{primitive}: the primitive does not draw in {space.value!r}"
        )
    subject, options = record.example(primitive)
    geom = geometry(subject, name, **dict(options))
    for part in geom.parts:
        needed = PRIMITIVES[part.primitive].requires if part.primitive else prim.requires
        assert needed <= set(part.channels), (
            f"{name}.{primitive}: part is missing channel(s) {sorted(needed - set(part.channels))}"
        )


def test_every_geometry_frame_is_one_the_transform_declared():
    """A transform that computes a frame it did not declare is a contract break."""
    for record in transforms():
        for primitive in sorted(record.primitives):
            subject, options = record.example(primitive)
            geom = geometry(subject, record.name, **dict(options))
            assert geom.frame.space in record.frame
            assert geom.frame.ndim in record.ndim


# ---------------------------------------------------------------------------
# 4. The matrix's reading marks
# ---------------------------------------------------------------------------


def test_the_exclusive_field_and_its_marker_are_gone():
    """``exclusive`` was a field no row could set, driving a ``!`` that never printed.

    It was measured ``frozenset()`` on all 39 registered transforms, it was not a
    parameter of :func:`ts.viz.transforms.register`, and ``'!' in
    str(ts.viz.compatibility())`` was ``False`` — three independent ways of saying
    the same thing.  Deleted in v6 round 9; this pins it staying deleted, on the
    record, on the ``rows()`` projection, and in the printed table.
    """
    for record in transforms():
        assert not hasattr(record, "exclusive"), record.name
    assert "exclusive" not in compatibility().rows()[0]
    assert "!" not in repr(compatibility())


def test_the_matrix_reports_the_default_marker():
    """``compatibility()`` is readable *and* programmable: ``*`` marks the default."""
    matrix = compatibility()
    for record in transforms():
        row = matrix[record.name]
        assert f"{record.default_primitive}*" in row
        assert set(compatibility(record.name)) == set(row)
        assert record.name in repr(matrix)


# ---------------------------------------------------------------------------
# 5. Every registered transform is a real, exercisable transform
# ---------------------------------------------------------------------------


def test_every_registered_transform_computes_a_geometry():
    """``compute`` is callable and returns a ``Geometry`` — the anti-vapourware gate."""
    for record in transforms():
        assert callable(record.compute)
        assert record.doc, f"{record.name} has no one-line doc"
        assert record.source in ("data", "model")
        subject, options = record.example(record.default_primitive)
        result = record.compute(subject, **dict(options))
        assert isinstance(result, Geometry), f"{record.name} returned {type(result).__name__}"
        assert result.transform == record.name
        assert len(result.parts) >= 1
        assert result.channel_names(), f"{record.name} produced no channels"


def test_every_in_tree_transform_declares_a_gate_example():
    """The example lives on the record so that adding a transform edits no test.

    This is the invariant that makes "one registration and nothing else" true: if
    the example table lived in this file, every new transform would need a test
    edit, and the one that skipped it would silently escape the gate.
    """
    missing = [
        t.name
        for t in transforms()
        if t.example is None and t.compute.__module__.startswith("tsdynamics.")
    ]
    assert not missing, f"in-tree transforms with no example: {missing}"


def test_every_primitive_is_either_claimed_by_a_row_or_reserved_on_purpose():
    """No primitive sits in the library unaccounted for.

    A primitive claimed by a row is rendered by this gate; one that is not must be
    listed in ``RESERVED_PRIMITIVES`` with the reason it exists, and is
    smoke-tested directly in ``tests/test_viz_transforms.py``.  Nothing in
    ``RESERVED_PRIMITIVES`` appears in any row, so no caller is ever offered a
    plot they cannot get.
    """
    claimed = {p for t in transforms() for p in t.primitives}
    assert claimed | set(RESERVED_PRIMITIVES) == set(PRIMITIVES)
    assert not claimed & set(RESERVED_PRIMITIVES), (
        "a reserved primitive is claimed by a row — move it out of RESERVED_PRIMITIVES so "
        "the compatibility gate renders it"
    )
    for name, reason in RESERVED_PRIMITIVES.items():
        assert reason, f"reserved primitive {name!r} has no recorded reason"
