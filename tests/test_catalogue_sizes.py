"""Structural dimensions must agree with the caller's declared sizes."""

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import InvalidParameterError

_SIZES = [
    (ts.systems.Lorenz96, "N", 8),
    (ts.systems.KuramotoSivashinsky, "N", 8),
    (ts.systems.MultiChua, "n_circuits", 2),
    (ts.systems.GrayScott, "N", 4),
    (ts.systems.SwiftHohenberg, "N", 4),
]


@pytest.mark.parametrize("cls,name,size", _SIZES)
@pytest.mark.parametrize("value", [4.5, -1, 0, True, np.nan, np.inf, "8"])
@pytest.mark.parametrize("channel", ["keyword", "params"])
def test_structural_sizes_refuse_invalid_values(cls, name, size, value, channel):
    kwargs = {name: value} if channel == "keyword" else {"params": {name: value}}
    with pytest.raises(InvalidParameterError, match=rf"{name} must be a positive integer"):
        cls(**kwargs)


@pytest.mark.parametrize("cls,name,size", _SIZES)
@pytest.mark.parametrize("channel", ["keyword", "params"])
def test_whole_numeric_sizes_remain_usable(cls, name, size, channel):
    kwargs = {name: float(size)} if channel == "keyword" else {"params": {name: np.int64(size)}}
    system = cls(**kwargs)
    assert system.params[name] == size
    assert system.copy().dim == system.dim
    assert system.with_params(**{name: size}).dim == system.dim


def test_small_gray_scott_grid_still_has_a_nucleation_patch():
    system = ts.systems.GrayScott(N=4)
    substrate = system.ic[:16].reshape(4, 4)
    # Small valid grids used to wrap a negative slice start and seed one corner.
    assert np.all(substrate < 0.6)
