"""``help(subject.plot.<name>)`` prints THAT transform's options — the gate.

The owner drove the plotting layer by hand, liked it, and hit one wall::

    "on the plots made like that how can we have more control over them.  For
     example the orbit diagram, I would like to choose the parameter I want, the
     resolution of the sweep, etc..."

Every one of those knobs was real — ``param=`` ``values=`` ``points=``
``transient=`` ``bins=`` — and a *wrong* keyword already listed them back
(*"Keywords accepted here: [...]"*).  Nothing surfaced them on the successful
path.  Measured at HEAD before this file existed::

    >>> inspect.signature(traj.plot.psd)
    (*things: 'Any', layout: 'str' = 'overlay', rows: 'int | None' = None,
     cols: 'int | None' = None, share_x: 'bool | None' = None, ...)

— ``ts.plot``'s generic **composition** vocabulary, which is the one thing that
call is not asking about, because the namespace entry was a bare
:func:`functools.partial`.  And::

    >>> repr(ts.viz.transforms.get("orbit_diagram"))
    PlotTransform('orbit_diagram', source='model', primitives=['density', 'points*'])

These tests fail against both of those.
"""

from __future__ import annotations

import inspect

import pytest

import tsdynamics as ts


@pytest.fixture(scope="module")
def traj():
    """A short Lorenz orbit — every ``data`` transform admits it."""
    return ts.systems.Lorenz().run(final_time=4.0, dt=0.05, ic=[1.0, 1.0, 1.0])


class TestHelpShowsTheTransformsOwnOptions:
    """``help()`` / ``?`` / ``inspect.signature`` all answer about the transform."""

    def test_the_signature_is_the_transforms_own_not_ts_plots(self, traj):
        """``psd`` names ``components`` / ``method`` / ``nperseg``, not ``layout`` / ``rows``."""
        sig = inspect.signature(traj.plot.psd)
        assert {"components", "method", "nperseg"} <= set(sig.parameters)
        # The composition vocabulary belongs to ``ts.plot(*things, ...)``, and
        # printing it here answered a question nobody asked.
        assert "things" not in sig.parameters
        assert "layout" not in sig.parameters

    def test_the_orbit_diagram_shows_the_knobs_the_owner_went_looking_for(self):
        """The four words from the quote, on the object in hand."""
        sig = inspect.signature(ts.systems.Logistic().plot.orbit_diagram)
        assert {"param", "values", "points", "transient"} <= set(sig.parameters)
        assert sig.parameters["points"].default == 100  # the real default, not a guess

    def test_the_entry_is_named_and_documented(self, traj):
        """``__name__`` / ``__doc__`` — what ``help()`` and IPython's ``?`` read."""
        entry = traj.plot.psd
        assert entry.__name__ == "psd"
        doc = entry.__doc__ or ""
        assert "ts.plot(subject, 'psd'" in doc
        assert "Drawn by" in doc and "Accepts" in doc
        # The transform's own prose survives — the docstring is prepended, not
        # replaced by a generated stub.
        first = (ts.viz.transforms.get("psd").compute.__doc__ or "").strip().splitlines()[0]
        assert first in doc

    def test_every_transform_a_subject_offers_has_a_real_signature(self, traj):
        """Swept, not sampled: a generated namespace must be right for all of them."""
        for name in dir(traj.plot):
            entry = getattr(traj.plot, name)
            sig = inspect.signature(entry)
            assert "things" not in sig.parameters, name
            assert "primitive" in sig.parameters, name
            assert entry.__name__ == name

    def test_a_namespace_entry_still_draws(self, traj):
        """The signature is decoration; the call is the point."""
        import matplotlib.pyplot as plt

        try:
            plot = traj.plot.phase_portrait(components=("x", "z"))
            assert plot.layers
        finally:
            plt.close("all")


class TestTheRegistryRecordPrintsWhatYouCanSteer:
    """``ts.viz.transforms.get(name)`` is a door, not a label."""

    def test_the_repr_lists_the_options(self):
        text = repr(ts.viz.transforms.get("orbit_diagram"))
        assert "options:" in text
        for word in ("param", "values", "points", "transient", "bins"):
            assert word in text, word
        # ...and says where the full signature lives.
        assert "help(subject.plot.orbit_diagram)" in text

    def test_options_are_the_split_the_library_actually_performs(self):
        """Not a second description of the split — the same union it routes by."""
        from tsdynamics.viz.transforms._registry import row_option_names

        record = ts.viz.transforms.get("orbit_diagram")
        own = {p.name for p in record._compute_parameters()}
        assert set(record.options) == own | set(row_option_names(record))
        assert "bins" in record.primitive_options  # density's option, not compute's

    def test_the_dataframe_able_rows_carry_the_options(self):
        """The printed matrix stays a table; ``rows()`` is where width is free."""
        rows = {r["transform"]: r for r in ts.viz.compatibility().rows()}
        assert "param" in rows["orbit_diagram"]["options"]
        assert all("options" in r for r in rows.values())

    def test_the_matrix_says_where_the_options_are(self):
        text = repr(ts.viz.compatibility())
        assert "help(subject.plot.name)" in text
        assert "ts.viz.transforms.get(" in text
