"""Regression tests for ``MetaStore.__repr__`` (issue #117, stream WS-META117).

The repr must show the *latest value* per key (annotating overwritten keys with
``(xN)``) rather than the record *counts* the v3 repr printed — so that
``sys.meta`` reveals a computed spectrum instead of ``MetaStore({'...': 1})``.
"""

from __future__ import annotations

import numpy as np

from tsdynamics.families.base import MetaStore


class TestMetaStoreReprValues:
    def test_empty_store(self) -> None:
        assert repr(MetaStore()) == "MetaStore()"

    def test_single_value_unannotated(self) -> None:
        m = MetaStore()
        m["dt"] = 0.01
        assert repr(m) == "MetaStore(dt=0.01)"

    def test_string_value_is_quoted(self) -> None:
        m = MetaStore()
        m["system"] = "lorenz"
        assert repr(m) == "MetaStore(system='lorenz')"

    def test_overwritten_key_is_annotated_and_shows_latest(self) -> None:
        m = MetaStore()
        m["system"] = "lorenz"
        m["system"] = "rossler"
        r = repr(m)
        assert "system='rossler'" in r  # latest value, not the first
        assert "(x2)" in r
        assert "lorenz" not in r

    def test_multiple_keys_in_insertion_order(self) -> None:
        m = MetaStore()
        m["dt"] = 0.01
        m["system"] = "lorenz"
        m["system"] = "lorenz"  # overwrite -> (x2)
        assert repr(m) == "MetaStore(dt=0.01, system='lorenz' (x2))"

    def test_single_record_has_no_annotation(self) -> None:
        m = MetaStore()
        m["a"] = 1
        assert "(x" not in repr(m)


class TestMetaStoreReprIssue117:
    """The exact scenario from the bug report: a recorded spectrum must show."""

    def test_recorded_spectrum_value_appears(self) -> None:
        m = MetaStore()
        spectrum = np.array([9.17460497e-01, -5.40558461e-03, -1.45787182e01])
        m.record("lyapunov_spectrum", spectrum, dt=0.01, final_time=300.0)
        r = repr(m)
        # The leading exponent value is visible — not a bare record count.
        assert "9.17460497e-01" in r
        # The v3 count-dict form must be gone.
        assert "{'lyapunov_spectrum': 1}" not in r
        assert "lyapunov_spectrum=" in r


class TestMetaStoreReprFormatting:
    def test_long_value_is_truncated_single_line(self) -> None:
        m = MetaStore()
        m["big"] = np.arange(200.0)  # a long, multi-line array repr
        r = repr(m)
        assert "\n" not in r  # collapsed to one line
        assert "..." in r  # truncated
        # The per-value portion stays bounded (well under the full array repr).
        assert len(r) < 120

    def test_multiline_value_is_collapsed(self) -> None:
        m = MetaStore()
        m["mat"] = np.eye(3)  # numpy renders this across several lines
        r = repr(m)
        assert "\n" not in r

    def test_broken_repr_does_not_crash(self) -> None:
        class _Boom:
            def __repr__(self) -> str:
                raise RuntimeError("no repr for you")

        m = MetaStore()
        m["bad"] = _Boom()
        r = repr(m)  # must not raise
        assert "bad=<_Boom>" in r


class TestMetaStoreReprDoesNotChangeAccessors:
    """The repr change must leave __getitem__/latest()/history() untouched."""

    def test_getitem_and_latest_unchanged(self) -> None:
        m = MetaStore()
        m["a"] = 1
        m["a"] = 3
        m["b"] = 2
        assert m["a"] == 3
        assert m.latest() == {"a": 3, "b": 2}
        assert [h["value"] for h in m.history("a")] == [1, 3]
