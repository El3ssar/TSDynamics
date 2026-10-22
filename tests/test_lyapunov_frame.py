"""Partial spectra must not be trapped in a contracting coordinate subspace."""

import json
import subprocess
import sys

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.derived.tangent import TangentSystem, _initial_frame


class ProductMap(ts.DiscreteMap):
    """A contracting x coordinate beside a chaotic logistic y coordinate."""

    dim = 2
    params = {}

    @staticmethod
    def _step(x):
        return np.array([0.5 * x[0], 4.0 * x[1] * (1.0 - x[1])])


class ReversedDiagonalFlow(ts.ContinuousSystem):
    """The first two coordinate directions contract fastest, not slowest."""

    dim = 4
    params = {}

    @staticmethod
    def _equations(y, t):
        return [-4.0 * y(0), -3.0 * y(1), -0.5 * y(2), -1.0 * y(3)]


@pytest.mark.parametrize("backend", ["interp", "jit", "reference"])
@pytest.mark.parametrize("k", [1, 2])
def test_product_map_finds_the_expanding_coordinate_and_sorts(backend, k):
    result = ts.analysis.lyapunov_spectrum(
        ProductMap(ic=[0.2, 0.4]), k=k, n=5000, transient=100, backend=backend
    )
    expected = [np.log(2.0), -np.log(2.0)][:k]
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.002)
    assert result.chaotic is True


@pytest.mark.parametrize("backend", ["interp", "jit", "reference"])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_diagonal_flow_finds_leading_directions_outside_first_coordinates(backend, k):
    result = ts.analysis.lyapunov_spectrum(
        ReversedDiagonalFlow(ic=[0.0] * 4),
        k=k,
        final_time=10.0,
        transient=40.0,
        dt=0.1,
        backend=backend,
    )
    np.testing.assert_allclose(result, [-0.5, -1.0, -3.0, -4.0][:k], rtol=0.0, atol=1e-7)


def test_initial_frames_are_orthonormal_reproducible_and_do_not_use_global_rng():
    before = np.random.get_state()
    q = _initial_frame(64, 64)
    np.testing.assert_array_equal(q, _initial_frame(64, 64))
    np.testing.assert_array_equal(q[:, :1], _initial_frame(64, 1))
    np.testing.assert_allclose(q.T @ q, np.eye(64), rtol=0.0, atol=2e-14)
    assert np.all(q[:, 0] != 0.0)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("backend", ["interp", "jit", "reference"])
def test_streaming_frame_and_growths_keep_the_same_qr_order(backend):
    tangent = TangentSystem(ProductMap(ic=[0.2, 0.4]), k=2, backend=backend)
    tangent.reinit([0.2, 0.4])
    initial = tangent.deviations()
    tangent.step()
    jac = np.diag([0.5, 4.0 * (1.0 - 2.0 * 0.4)])
    expected_q, expected_r = np.linalg.qr(jac @ initial)
    np.testing.assert_allclose(abs(tangent.deviations()), abs(expected_q), atol=1e-14)
    np.testing.assert_allclose(tangent.exponents(), np.log(abs(np.diag(expected_r))), atol=1e-14)


def test_raw_ffi_rejects_an_overflowing_gap_before_entering_the_solver():
    rust = pytest.importorskip("tsdynamics._rust")
    from tsdynamics._engine.compile import lower_ode

    system = ReversedDiagonalFlow(ic=[0.0] * 4)
    with pytest.raises(ValueError, match="gap.*not finite"):
        rust.integrate_dense(
            *lower_ode(system).to_arrays(),
            np.zeros(4),
            np.array([]),
            np.array([-1e308, 1e308]),
            "rk4",
            1e-7,
            1e-9,
            np.inf,
            False,
            False,
        )


class EscapeFlow(ts.ContinuousSystem):
    dim = 1
    params = {"rate": 20.0}

    @staticmethod
    def _equations(y, t, *, rate):
        return [rate * y(0)]


@pytest.mark.parametrize("backend", ["interp", "jit", "reference"])
def test_lyapunov_rejects_an_escaped_base_orbit_before_reporting_a_spectrum(backend):
    with pytest.raises(ts.errors.ConvergenceError, match="base orbit escaped at t=.*bounded orbit"):
        ts.analysis.lyapunov_spectrum(
            EscapeFlow(ic=[1.0]), final_time=3.0, transient=0.0, dt=0.05, backend=backend
        )


@pytest.mark.parametrize("backend", ["interp", "jit", "reference"])
@pytest.mark.parametrize("rate", [0.0, -0.2])
def test_large_unit_bounded_orbits_are_accepted_and_not_labelled_escaped(backend, rate):
    result = ts.analysis.lyapunov_spectrum(
        EscapeFlow(rate=rate, ic=[1e12]),
        final_time=2.0,
        transient=1.0,
        dt=0.1,
        backend=backend,
    )
    np.testing.assert_allclose(result, [rate], rtol=0.0, atol=1e-7)
    assert not result.unbounded


@pytest.mark.parametrize("backend", ["interp", "jit", "reference"])
def test_large_tangent_growth_does_not_count_as_a_base_orbit_escape(backend):
    result = ts.analysis.lyapunov_spectrum(
        EscapeFlow(rate=200.0, ic=[0.0]),
        final_time=0.2,
        transient=0.0,
        dt=0.1,
        backend=backend,
    )
    np.testing.assert_allclose(result, [200.0], rtol=0.0, atol=1e-5)
    assert not result.unbounded


def _run_spectrum_process(source, *args):
    try:
        process = subprocess.run(
            [sys.executable, "-c", source, *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(
            f"Lyapunov subprocess exceeded 30s. stdout={error.stdout!r}; stderr={error.stderr!r}",
            pytrace=False,
        )
    assert process.returncode == 0, (
        f"Lyapunov subprocess exited {process.returncode}. "
        f"stdout={process.stdout!r}; stderr={process.stderr!r}"
    )
    return json.loads(process.stdout)


@pytest.mark.parametrize("backend", ["interp", "jit", "reference"])
def test_deterministically_escaping_nonlinear_orbit_is_refused_promptly(backend):
    # x(t)=exp(20t), while y,z rotate at exp(5t) with constant radius.
    # Escape is guaranteed, unlike a particular chaotic orbit. Without an
    # escape check, reaching t=4 requires resolving an enormous rotation rate.
    # The guard should stop at the first .05 interval after log(1e8)/20.
    source = """
import json,sys,time
import tsdynamics as ts
class GrowingOscillator(ts.ContinuousSystem):
    dim = 3
    params = {}
    @staticmethod
    def _equations(y,t):
        omega = y(0)**.25
        return [20*y(0),-omega*y(2),omega*y(1)]
start = time.process_time()
try:
    ts.analysis.lyapunov_spectrum(GrowingOscillator(ic=[1.,1.,0.]), k=1,
                                 dt=.05, transient=0., final_time=4., backend=sys.argv[1])
except ts.errors.ConvergenceError as exc:
    print(json.dumps({'cpu':time.process_time()-start,'error':str(exc)}))
else:
    raise AssertionError('the escaped orbit produced a finite attractor spectrum')
"""
    result = _run_spectrum_process(source, backend)
    # The stop interval checks the algorithm; the subprocess fuse bounds a
    # runaway without treating a shared CI runner as calibrated benchmark hardware.
    assert "base orbit escaped at t=0.95" in result["error"]
    assert "bounded orbit" in result["error"]


def test_hyperlu_reference_start_returns_a_bounded_result_or_refuses_promptly():
    # Roundoff can send this chaotic orbit onto bounded or escaping branches on
    # different platforms. Either outcome must be prompt and report honestly;
    # the deterministic nonlinear case above independently requires refusal.
    source = """
import json,time,zlib
import numpy as np
import tsdynamics as ts
seed = zlib.crc32(b'HyperLu') & 0x7fffffff
ic = .1 * (2 * np.random.default_rng(seed).random(4) - 1)
start = time.process_time()
try:
    spectrum = ts.analysis.lyapunov_spectrum(ts.systems.HyperLu(), k=1, ic=ic,
                                            dt=.05, transient=50., final_time=400.)
except ts.errors.ConvergenceError as exc:
    result = {'status':'refused','error':str(exc)}
else:
    assert np.isfinite(spectrum).all(), spectrum
    assert not spectrum.unbounded, spectrum
    result = {'status':'bounded','spectrum':np.asarray(spectrum).tolist()}
result['cpu'] = time.process_time()-start
print(json.dumps(result))
"""
    result = _run_spectrum_process(source)
    assert result["status"] in {"bounded", "refused"}, result
    if result["status"] == "refused":
        assert "base orbit escaped at t=" in result["error"]
        assert "bounded orbit" in result["error"]
