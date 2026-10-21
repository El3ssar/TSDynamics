"""Shape-changing parameter rebuilds regenerate a compatible state and field."""

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import InvalidParameterError

_SIZES = [
    (ts.systems.Lorenz96, "N", 5, 4),
    (ts.systems.KuramotoSivashinsky, "N", 8, 7),
    (ts.systems.MultiChua, "n_circuits", 3, 2),
    (ts.systems.GrayScott, "N", 4, 3),
    (ts.systems.SwiftHohenberg, "N", 4, 3),
]


@pytest.mark.parametrize("cls,name,old_size,new_size", _SIZES)
@pytest.mark.parametrize("reverse", [False, True])
def test_with_params_resizes_default_initial_state_and_geometry(
    cls, name, old_size, new_size, reverse
):
    if reverse:
        old_size, new_size = new_size, old_size
    system = cls(**{name: old_size}, seed=12)
    # Exercise the retained-IC case even on systems whose IC is resolved lazily.
    before = system.run(0.01, dt=0.01, seed=12)
    old_ic = system.ic.copy()
    rebuilt = system.with_params(**{name: new_size})
    fresh = cls(**{name: new_size}, seed=12)
    assert rebuilt.dim == fresh.dim != system.dim
    assert rebuilt._field_shape == fresh._field_shape
    after = rebuilt.run(0.01, dt=0.01)
    assert after.y.shape == (2, fresh.dim)
    assert np.isfinite(after.y).all()
    assert len(rebuilt.variables) == fresh.dim
    assert rebuilt.ic.shape == (fresh.dim,)
    expected = fresh.run(0.01, dt=0.01)
    np.testing.assert_array_equal(after.y, expected.y)
    # A rebuild does not resize or mutate the source's arrays.
    np.testing.assert_array_equal(system.ic, old_ic)
    assert before.y.shape == (2, system.dim)


@pytest.mark.parametrize("cls,name,old_size,new_size", _SIZES)
@pytest.mark.parametrize("reverse", [False, True])
def test_direct_shape_mutation_refuses_with_an_executable_rebuild(
    cls, name, old_size, new_size, reverse
):
    if reverse:
        old_size, new_size = new_size, old_size
    system = cls(**{name: old_size})
    system.params[name] = new_size
    with pytest.raises(InvalidParameterError, match="with_params") as exc:
        system.run(0.01, dt=0.01)
    remedy = f"system = system.with_params({name}={new_size})"
    assert remedy in str(exc.value)
    namespace = {"system": system}
    exec(remedy, namespace)
    rebuilt = namespace["system"]
    result = rebuilt.run(0.01, dt=0.01)
    assert result.y.shape == (2, rebuilt.dim)
    assert np.isfinite(result.y).all()


def test_same_shape_parameter_rebuild_preserves_the_chosen_initial_state():
    system = ts.systems.Lorenz96(N=5, ic=np.arange(5.0), seed=12)
    rebuilt = system.with_params(f=9.0)
    np.testing.assert_array_equal(rebuilt.ic, system.ic)
    assert rebuilt._ic_explicit is True
    assert rebuilt.ic is not system.ic
