"""The engine must not take the interpreter down with it (v6 WP2).

Four ways the compiled engine used to damage the *process* hosting it, rather
than just failing the call:

1. **An unchecked allocation aborted the interpreter.**  Every family sized an
   output buffer as ``vec![0.0; rows * cols]`` from numbers the caller chose.  A
   large ``steps`` either overflowed ``usize`` (a ``capacity overflow`` panic
   across the FFI) or asked for more memory than exists, whereupon Rust's
   allocation-error handler called ``abort()``.  ``SIGABRT`` is uncatchable:
   no traceback, no ``except``, and every unsaved notebook cell gone.
2. **Any parallel call poisoned ``fork()``.**  The ensembles fanned out over
   rayon's *global* pool, whose worker threads do not survive a fork; the first
   parallel call in a forked child then blocked forever on a queue nobody was
   draining.  One ``ensemble()`` anywhere in a session was enough to make every
   later ``multiprocessing`` child hang.
3. **No engine call was interruptible.**  Long calls release the GIL, so Ctrl-C
   set CPython's signal flag and nothing happened until the call returned — a
   mistyped ``final_time`` locked the session with no way out but ``kill``.
4. **A stalled run was misreported as a divergence.**  Exhausting the solver
   step budget with a perfectly finite state was reported as
   ``"integration diverged"``, sending a user whose model is merely stiff to
   hunt a blow-up that does not exist.

Each test that could take the *test session* down with it — the allocation and
fork ones — runs in a **subprocess**, so a regression fails the assertion
instead of killing the run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

# The whole module is about the compiled engine's process behaviour.
pytest.importorskip("tsdynamics._rust")

import tsdynamics as ts  # noqa: E402
from tsdynamics.errors import ConvergenceError, StepBudgetError  # noqa: E402


def _run_isolated(body: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    """Run `body` in a fresh interpreter; never let it take the session with it."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# 1. Unchecked allocation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("steps", "what"),
    [
        # `steps * dim` is representable but unservable: the allocator returns
        # null and `handle_alloc_error` used to abort the process (exit 134).
        (2**50, "unservable"),
        # `steps * dim` overflows usize: `RawVec` used to panic `capacity overflow`.
        (2**62, "overflowing"),
    ],
)
def test_absurd_map_step_count_raises_instead_of_killing_the_process(steps, what):
    """An impossible output buffer is a ``MemoryError``, not a ``SIGABRT``.

    Run in a subprocess precisely because the failure mode under test *is* the
    death of the interpreter: an in-process assertion could never report it.
    ``returncode == 0`` is therefore as much of the assertion as the exception
    type — a negative return code means the child was killed by a signal.
    """
    proc = _run_isolated(
        f"""
        import resource
        # Cap the address space so an unservable request fails fast rather than
        # sending the machine swapping.
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
        import tsdynamics as ts

        try:
            ts.systems.Henon().run(steps={steps}, ic=[0.1, 0.1])
        except MemoryError as e:
            assert "cannot allocate" in str(e), e
            print("RAISED")
        else:
            raise AssertionError("an impossible allocation must not succeed")
        """
    )
    assert proc.returncode == 0, (
        f"the {what} request killed the interpreter "
        f"(returncode {proc.returncode}): {proc.stderr[-2000:]}"
    )
    assert "RAISED" in proc.stdout, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# 2. fork() after a parallel call
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork() is POSIX-only")
def test_fork_after_a_parallel_call_does_not_deadlock_the_child():
    """A ``multiprocessing`` child must survive its parent's rayon pool.

    The child does the *whole* fragile sequence: it makes a parallel engine call
    of its own, which is what used to block forever on the parent's dead
    workers.  The subprocess timeout is the assertion — a regression hangs.
    """
    proc = _run_isolated(
        """
        import os
        import sys
        import warnings

        import numpy as np

        import tsdynamics as ts
        from tsdynamics._engine import run as engine_run

        warnings.simplefilter("ignore")  # CPython's fork-in-a-thread notice

        lor = ts.systems.Lorenz()
        ics = np.random.default_rng(0).normal(size=(64, 3))

        # Spin up the pool in the parent — the step that used to poison the fork.
        parent = engine_run.ensemble(lor, ics, final_time=1.0, dt=0.01)

        pid = os.fork()
        if pid == 0:
            try:
                child = engine_run.ensemble(lor, ics, final_time=1.0, dt=0.01)
                # The rebuilt pool must give the SAME answer: the parallel ==
                # serial determinism contract does not care which pool ran it.
                os._exit(0 if np.array_equal(child, parent) else 3)
            except BaseException:
                os._exit(2)

        _, status = os.waitpid(pid, 0)
        assert status == 0, f"child exited with status {status}"
        print("CHILD OK")
        """,
        timeout=180.0,
    )
    assert proc.returncode == 0, (
        f"the forked child did not complete (returncode {proc.returncode}); "
        f"a hang here is the deadlock this test exists for: {proc.stderr[-2000:]}"
    )
    assert "CHILD OK" in proc.stdout, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# 3. Ctrl-C during a long engine call
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not hasattr(os, "kill"), reason="needs os.kill to raise SIGINT")
def test_a_long_engine_call_is_interruptible():
    """SIGINT during a multi-minute integration surfaces promptly.

    The run is sized so it takes far longer than the interrupt deadline: if the
    call is not interruptible, the child hits the subprocess timeout rather than
    reporting a latency.  Asserting a *bound* (rather than a duration) keeps the
    test insensitive to machine speed while still failing if signals are only
    delivered at the end of the call.
    """
    proc = _run_isolated(
        """
        import os
        import signal
        import threading
        import time

        import tsdynamics as ts

        lor = ts.systems.Lorenz()
        threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()

        t0 = time.perf_counter()
        try:
            # ~4e7 rk4 steps: minutes of engine time.
            lor.run(final_time=200_000.0, dt=0.005, solver="rk4")
        except KeyboardInterrupt:
            print(f"INTERRUPTED {time.perf_counter() - t0:.3f}")
        else:
            raise AssertionError("the call ran to completion instead of stopping")
        """,
        timeout=120.0,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "INTERRUPTED" in proc.stdout, proc.stdout + proc.stderr
    elapsed = float(proc.stdout.split("INTERRUPTED")[1].split()[0])
    # The signal fires at 1.0 s; anything under a few seconds means the poll is
    # working. A deferred interrupt would take minutes (or time the child out).
    assert elapsed < 15.0, f"interrupt took {elapsed:.1f}s — signals are being deferred"


# The long-running public calls that are *not* a plain ``integrate``: each has
# its own step loop in the engine, so each needed its own poll site. Every one
# of them is sized here to run for minutes if left alone.
_LONG_CALLS: dict[str, str] = {
    # The map orbit-diagram sweep kernel (`param_sweep.rs`).
    "orbit_diagram": """
        import numpy as np
        ts.analysis.orbit_diagram(
            ts.systems.Logistic(), "r", np.linspace(3.5, 4.0, 4000),
            points_per_value=2000, transient=200_000,
        )
    """,
    # The extended-variational chunk loop (`lyapunov.rs`), which drives many
    # short integrations — the case a per-segment poller would have missed.
    "lyapunov_spectrum": """
        ts.analysis.lyapunov_spectrum(ts.systems.Lorenz(), final_time=2_000_000.0, dt=0.01)
    """,
    # The QR tangent-map iteration (`map_lyapunov.rs`).  Spelled through
    # `lyapunov_spectrum`, which absorbed `max_lyapunov` in v6 — `k=1` is the
    # leading exponent, and on a map that is the engine QR kernel.
    "map_lyapunov": """
        ts.analysis.lyapunov_spectrum(
            ts.systems.Henon(), k=1, ic=[0.1, 0.1], n=200_000_000
        )
    """,
    # The recurrence FSM (`basin.rs`), one `dt` segment per cell check.
    "basins_of_attraction": """
        import numpy as np
        grid = ts.data.Grid(
            np.array([-20.0, -20.0, 0.0]), np.array([20.0, 20.0, 40.0]), (80, 80, 80)
        )
        ts.analysis.basins(
            ts.systems.Lorenz(), grid, dt=0.01, max_steps=10_000_000
        )
    """,
    # The event march (`event.rs`): a plane the flow never reaches, so the
    # search runs the whole span.
    "poincare_section": """
        ts.analysis.poincare_section(
            ts.systems.Rossler(), plane=("y", 1e9, "up"), crossings=1, max_time=1e7, dt=0.001
        )
    """,
    # The three rayon fan-outs (`ensemble.rs` / `map.rs` / `sde.rs`). These are a
    # different failure mode from the loops above, which merely lacked a poll
    # site: the ensembles had one (each trajectory runs the polled per-trajectory
    # loop) and it could never fire, because `ThreadPool::install` parked the one
    # armed thread for the whole batch while every worker skipped the hook by
    # design.  See `test_an_ensemble_is_interruptible`.
    "ensemble_ode": """
        import numpy as np
        from tsdynamics._engine import run as engine_run
        ics = np.random.default_rng(0).normal(size=(64, 3))
        engine_run.ensemble(
            ts.systems.Lorenz(), ics, final_time=200_000.0, dt=0.005, method="rk4"
        )
    """,
    "ensemble_map": """
        import numpy as np
        from tsdynamics._engine import run as engine_run
        ics = np.random.default_rng(0).normal(size=(64, 2)) * 0.1
        engine_run.ensemble(ts.systems.Henon(), ics, final_time=2_000_000_000)
    """,
    "ensemble_sde": """
        import numpy as np
        ics = np.full((256, 1), 1.0)
        ts.systems.OrnsteinUhlenbeck().ensemble(
            ics).run(final_time=5_000.0, dt=1e-4, seed=0).final
    """,
}


@pytest.mark.slow
@pytest.mark.skipif(not hasattr(os, "kill"), reason="needs os.kill to raise SIGINT")
@pytest.mark.parametrize("name", sorted(_LONG_CALLS))
def test_every_long_engine_call_is_interruptible(name):
    """Ctrl-C escapes *every* long engine call, not just ``integrate``.

    Polling one loop is not enough: the sweep, the Lyapunov chunk loop, the QR
    tangent iteration, the basin FSM and the event march each own a step loop of
    their own, and an unpolled one is a session the user cannot get back.  Two
    of them additionally had to learn to *report* the interrupt: the Lyapunov
    chunk loop funnelled every integrate failure into "diverged", and the basin
    march's advance returned a bare ``bool``, so an interrupt would have been
    written into the basin image as a diverged initial condition.
    """
    proc = _run_isolated(
        f"""
        import os
        import signal
        import threading
        import time

        import tsdynamics as ts

        threading.Timer(2.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
        t0 = time.perf_counter()
        try:
{textwrap.indent(textwrap.dedent(_LONG_CALLS[name]).strip(), " " * 12)}
        except KeyboardInterrupt:
            print(f"INTERRUPTED {{time.perf_counter() - t0:.3f}}")
        else:
            raise AssertionError("the call ran to completion instead of stopping")
        """,
        timeout=180.0,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "INTERRUPTED" in proc.stdout, proc.stdout + proc.stderr
    elapsed = float(proc.stdout.split("INTERRUPTED")[1].split()[0])
    assert elapsed < 20.0, f"interrupt took {elapsed:.1f}s — signals are being deferred"


@pytest.mark.slow
@pytest.mark.skipif(not hasattr(os, "kill"), reason="needs os.kill to raise SIGINT")
def test_an_ensemble_is_interruptible():
    """Ctrl-C during a rayon fan-out must land, and land *promptly*.

    The ensembles were uninterruptible for a structural reason the per-loop poll
    sites above could not fix, and which reading the code made look covered:

    * ``ensemble_final`` entered the pool through ``ThreadPool::install``, which
      runs the closure on a pool worker and **parks the calling thread**;
    * the calling thread is the only *armed* one — ``PyErr_CheckSignals`` is a
      no-op off the main thread, and re-acquiring the GIL per worker would
      serialise the very loop the ensemble exists to parallelise, so workers are
      deliberately never armed;
    * so during a batch the one thread that could see a signal was asleep and
      every worker's ``Poller::tick`` skipped the hook. ``integrate_final`` *did*
      build a poller — it simply could never fire.

    The fix is a driver thread (which may park) plus a cancellation ``AtomicBool``
    the workers read on their normal stride, so no worker touches the GIL and the
    batch partition — the parallel == serial contract — is unchanged.

    Asserting a *tight* bound here, rather than the 20 s the sweep above allows:
    the batch is sized to run for well over a minute, so anything near the signal
    time proves cancellation propagated rather than the batch merely finishing.
    """
    proc = _run_isolated(
        """
        import os
        import signal
        import threading
        import time

        import numpy as np

        import tsdynamics as ts
        from tsdynamics._engine import run as engine_run

        lor = ts.systems.Lorenz()
        ics = np.random.default_rng(0).normal(size=(64, 3))

        threading.Timer(2.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
        t0 = time.perf_counter()
        try:
            # ~4e7 rk4 steps per trajectory, 64 of them: minutes of engine time.
            engine_run.ensemble(
                lor, ics, final_time=200_000.0, dt=0.005, method="rk4"
            )
        except KeyboardInterrupt:
            print(f"INTERRUPTED {time.perf_counter() - t0:.3f}")
        else:
            raise AssertionError("the batch ran to completion instead of stopping")
        """,
        timeout=180.0,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "INTERRUPTED" in proc.stdout, proc.stdout + proc.stderr
    elapsed = float(proc.stdout.split("INTERRUPTED")[1].split()[0])
    # The signal fires at 2.0 s. The driver polls every 5 ms and a worker notices
    # within one poll stride, so this lands in milliseconds; a couple of seconds
    # of slack keeps it insensitive to machine speed while still failing loudly
    # if the batch is once again only interruptible at its end.
    assert elapsed < 4.0, f"interrupt took {elapsed:.1f}s — the batch is not cancelling"


def test_an_absurd_orbit_diagram_raises_instead_of_killing_the_process():
    """The sweep kernel's point buffer is caller-sized too.

    ``orbit_diagram`` over a map runs the whole sweep in one engine call, whose
    output is ``n_values × n × n_components`` floats — a *triple* product of
    numbers the caller picks.  Written as ``vec![0.0; ...]`` it aborted the
    interpreter on any large ``n``, from a completely ordinary public call.
    """
    proc = _run_isolated(
        """
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))

        import numpy as np
        import tsdynamics as ts

        try:
            ts.analysis.orbit_diagram(
                ts.systems.Logistic(), "r", np.linspace(3.5, 4.0, 4),
                points_per_value=2**40, transient=0,
            )
        except MemoryError as e:
            assert "cannot allocate" in str(e), e
            print("RAISED")
        else:
            raise AssertionError("an impossible allocation must not succeed")
        """
    )
    assert proc.returncode == 0, (
        f"the sweep killed the interpreter (returncode {proc.returncode}): {proc.stderr[-2000:]}"
    )
    assert "RAISED" in proc.stdout, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# 4. Stalled vs diverged
# ---------------------------------------------------------------------------


class _VeryStiff(ts.ContinuousSystem):
    """``dx/dt = -1e10 (x - cos t)``: bounded solution, unreachable by rk45.

    The solution never leaves ``[-1, 1]``, but an explicit kernel needs
    ``h ~ 1e-10`` to stay stable, so the run exhausts its step budget with a
    perfectly healthy state — the exact case that used to be called a
    divergence.
    """

    params: dict[str, float] = {}
    dim = 1

    @staticmethod
    def _equations(y, t):
        import symengine as se

        return [-1e10 * (y(0) - se.cos(t))]


@pytest.mark.slow
def test_a_stalled_run_is_a_step_budget_error_not_a_divergence():
    """Exhausting the step budget with a finite state must say so.

    ``final_time == dt`` gives exactly one output segment, so the per-segment
    step cap is reached once rather than once per grid point.
    """
    with pytest.raises(StepBudgetError) as excinfo:
        _VeryStiff(ic=[1.0]).run(final_time=1.0, dt=1.0, solver="rk45")

    message = str(excinfo.value)
    assert "step limit" in message
    assert "diverged" not in message, f"a stalled run must not claim divergence: {message}"
    # The message has to be actionable: the remedy for a stall is a solver knob.
    assert "stalled" in message
    assert "rtol" in message and "bdf" in message
    # Additive by construction: every existing divergence handler still catches it.
    assert isinstance(excinfo.value, ConvergenceError)
    assert isinstance(excinfo.value, RuntimeError)


class _Blowup(ts.ContinuousSystem):
    """``dx/dt = x**2``: escapes to infinity in finite time (``t = 1`` from 1)."""

    params: dict[str, float] = {}
    dim = 1

    @staticmethod
    def _equations(y, t):
        return [y(0) ** 2]


@pytest.mark.parametrize("method", ["rk45", "dop853", "rosenbrock", "trbdf2", "sdirk2", "bdf"])
def test_a_diverging_ode_is_reported_promptly_by_every_kernel(method):
    """Blow-up must be caught by magnitude, not by waiting for ``inf``.

    Two defects met here.  The engine had no amplitude guard at all, so a
    diverging trajectory had to climb every decade to ``f64::MAX`` (with the
    controller shrinking the step all the way) before anything was reported; and
    ``bdf`` alone had no relative step floor, so it did not report at all — it
    ground to the 1e8-step cap, taking tens of seconds where the other implicit
    kernels took a fraction of one.

    Asserting *which* error and *what* it says (rather than a wall-clock bound,
    which would be machine-dependent and flaky) pins both: the magnitude guard
    fires, and it fires for every kernel including ``bdf``.
    """
    with pytest.raises(ConvergenceError) as excinfo:
        _Blowup(ic=[1.0]).run(final_time=10.0, dt=0.01, solver=method)

    message = str(excinfo.value)
    assert "diverged" in message
    # Either escape guard is correct; the magnitude one is what makes it prompt.
    assert "state magnitude reached" in message or "non-finite" in message
    # A stall must never be dressed up as this, and vice versa.
    assert not isinstance(excinfo.value, StepBudgetError)


def test_large_but_bounded_amplitudes_still_integrate():
    """The escape guard keys on an *overflow* scale, not a physical one.

    ``1e150`` is chosen so the next squaring overflows a float; a system that
    merely reaches ``1e6`` has to integrate normally.  Without this the guard
    would be a silent correctness regression for every large-amplitude model in
    the catalogue.
    """

    class Growth(ts.ContinuousSystem):
        params: dict[str, float] = {}
        dim = 1

        @staticmethod
        def _equations(y, t):
            return [y(0)]

    traj = Growth(ic=[1.0]).run(final_time=13.8, dt=0.1, solver="rk45")
    final = float(traj.y[-1, 0])
    assert 1e5 < final < 1e7, final
