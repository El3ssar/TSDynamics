"""A recurrence plot accepts the estimator's distance and density controls."""

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import InvalidParameterError


def test_recurrence_plot_fixed_threshold_matches_the_estimator():
    values = np.sin(np.arange(32.0))
    expected = ts.analysis.recurrence_matrix(values, threshold=0.2)
    geometry = ts.viz.geometry(values, "recurrence", threshold=0.2)
    np.testing.assert_array_equal(geometry.channels["z"].values, expected.matrix.toarray())
    assert geometry.meta["epsilon"] == 0.2


def test_recurrence_plot_keeps_the_default_density():
    values = np.sin(np.arange(32.0))
    default = ts.viz.geometry(values, "recurrence")
    explicit = ts.viz.geometry(values, "recurrence", recurrence_rate=0.05)
    np.testing.assert_array_equal(default.channels["z"].values, explicit.channels["z"].values)


def test_recurrence_plot_rejects_conflicting_scale_controls():
    with pytest.raises(InvalidParameterError, match="pass exactly one"):
        ts.viz.geometry(np.arange(16.0), "recurrence", threshold=0.1, recurrence_rate=0.05)
