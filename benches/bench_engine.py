"""Head-to-head engine benchmark: the Rust engine vs the v2 backends vs SciPy.

This is the decision-gate harness behind stream I-BENCH.  It answers one
question with real numbers: **is the shipping Rust engine (the SSA-tape
interpreter and the Cranelift JIT, reachable through
``integrate(backend="interp")`` / ``backend="jit"`` and ``iterate(backend=...)``)
fast enough to justify retiring the v2 backends (JiTCODE / JiTCDDE / Numba) at
migration milestone M3?**

It measures three regimes that have *opposite winners* — and reports all three
separately, never an average:

(a) **cold time-to-first-result** — a fresh process, the first integration,
    *including* every compile / JIT / warmup cost.  This is where the
    zero-warmup interpreter should win big against JiTCODE's C compile.  Measured
    by spawning a clean subprocess per sample (warmup state does not survive a
    process), with the JiTCODE on-disk compile cache controlled explicitly
    (a truly-cold empty cache vs a warm cache that only reloads the ``.so``).

(b) **warm steady-state throughput** — warmup excluded, one long hot run,
    best-of-N in a single process.  This is where Cranelift (no high opt level
    set) may trail gcc ``-O3``-compiled JiTCODE, and where the tape interpreter
    is at its worst on a large RHS (Lorenz-96 N=128).  Reported honestly.

(c) **ensemble / sweep throughput** — many trajectories: the engine's rayon
    fan-out vs the v2 per-trajectory Python loop vs a SciPy loop.

Backends
--------
``interp`` and ``jit`` are reported **separately** — they are different products
(zero-warmup interpreter vs native-code JIT).  ``jitcode`` / ``jitcdde`` /
``numba`` are the v2 defaults; ``reference`` is the pure-Python tape oracle (the
current SDE default); ``scipy`` is the dependency-light baseline.  Tolerances are
pinned equal across every backend for a given system.

Usage
-----
Build the engine extension first (it is not built by default)::

    cargo build --release --features extension-module \
        --manifest-path crates/tsdyn-core/Cargo.toml --locked
    cp crates/tsdyn-core/target/release/lib_rust.so src/tsdynamics/_rust.abi3.so

Then::

    uv run python benches/bench_engine.py            # full run → Markdown + JSON
    uv run python benches/bench_engine.py --quick     # CI-sized smoke (fast)
    uv run python benches/bench_engine.py --regime warm
    uv run python benches/bench_engine.py --out results.json

Numbers are machine-dependent — regenerate locally; the committed report
(``benches/REPORT.md``) records the machine it was produced on.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

# Heavy imports (numpy, tsdynamics) are deferred into functions so the cold
# subprocess worker can time `import tsdynamics` itself.

THIS = os.path.abspath(__file__)


# ---------------------------------------------------------------------------
# System catalogue for the benchmark
# ---------------------------------------------------------------------------
#
# Each entry is keyed by a stable name (the cold subprocess worker reconstructs
# the system from this name, so the builders must be importable here — no
# pickling).  ``kind`` selects the family path; ``v2`` names the v2 default
# backend the engine is measured against.  Run sizes are tuned so a warm run is
# tens to a few hundred milliseconds and a cold run is informative.


def _build_systems() -> dict[str, dict[str, Any]]:
    """Return the name -> spec table (imports tsdynamics lazily)."""
    import tsdynamics as ts

    return {
        # --- small / medium chaotic ODEs (the bread-and-butter case) ---
        "Lorenz": {
            "kind": "ode",
            "build": lambda: ts.Lorenz(),
            "ic": [1.0, 1.0, 1.0],
            "final_time": 200.0,
            "dt": 0.01,
            "method": "RK45",
            "v2_method": "RK45",
            "rtol": 1e-8,
            "atol": 1e-10,
            "v2": "jitcode",
            "scipy": True,
        },
        "Rossler": {
            "kind": "ode",
            "build": lambda: ts.Rossler(),
            "ic": [1.0, 0.0, 0.0],
            "final_time": 500.0,
            "dt": 0.02,
            "method": "RK45",
            "v2_method": "RK45",
            "rtol": 1e-8,
            "atol": 1e-10,
            "v2": "jitcode",
            "scipy": True,
        },
        # --- stiff ODEs (implicit kernels; Jacobian-carrying tape) ---
        "Oregonator": {
            "kind": "ode",
            "build": lambda: ts.Oregonator(),
            "ic": [1.0, 1.0, 1.0],
            "final_time": 40.0,
            "dt": 0.01,
            "method": "bdf",  # variable-order BDF (E-BDF) — the auto-stiffness default
            "v2_method": "LSODA",
            "scipy_method": "LSODA",
            "rtol": 1e-7,
            "atol": 1e-9,
            "stiff": True,
            "v2": "jitcode",
            "scipy": True,
        },
        "ForcedVanDerPol": {
            "kind": "ode",
            "build": lambda: ts.ForcedVanDerPol(),  # mu = 8.53 (moderately stiff)
            "ic": [0.1, 0.1, 0.0],
            "final_time": 200.0,
            "dt": 0.01,
            "method": "bdf",  # variable-order BDF (E-BDF) — the auto-stiffness default
            "v2_method": "LSODA",
            "scipy_method": "LSODA",
            "rtol": 1e-7,
            "atol": 1e-9,
            "stiff": True,
            "v2": "jitcode",
            "scipy": True,
        },
        # --- large, long-and-hot: the adversarial worst case for the tape
        #     interpreter and for Cranelift (128-dim RHS, many steps) ---
        "Lorenz96-N128": {
            "kind": "ode",
            "build": lambda: ts.Lorenz96(N=128),
            "ic": None,  # resolved per process (small perturbation of the fixed point)
            "final_time": 50.0,
            "dt": 0.01,
            "method": "RK45",
            "v2_method": "RK45",
            "rtol": 1e-8,
            "atol": 1e-10,
            "v2": "jitcode",
            "scipy": True,
            "big": True,
        },
        # --- DDE (no SciPy baseline; v2 is JiTCDDE) ---
        "MackeyGlass": {
            "kind": "dde",
            "build": lambda: ts.MackeyGlass(),
            "ic": [0.5],
            "final_time": 500.0,
            "dt": 0.5,
            "method": "RK45",
            "v2_method": "RK45",
            "rtol": 1e-6,
            "atol": 1e-6,
            "v2": "jitcdde",
            "history_const": 0.5,
        },
        # --- maps (v2 is Numba; no SciPy baseline) ---
        "Henon": {
            "kind": "map",
            "build": lambda: ts.Henon(),
            "ic": [0.1, 0.1],
            "steps": 200_000,
            "v2": "numba",
        },
        "Logistic": {
            "kind": "map",
            "build": lambda: ts.Logistic(),
            "ic": [0.4],
            "steps": 500_000,
            "v2": "numba",
        },
        # --- SDE (no v2 backend; the *current default* is the pure-Python
        #     reference, so "reference vs engine" is the migration question) ---
        "OU": {
            "kind": "sde",
            "build": _ou_builder,
            "ic": [2.0],
            "final_time": 50.0,
            "dt": 1e-3,
            "method": "euler_maruyama",
            "seed": 12345,
            "v2": "reference",
        },
        "GBM": {
            "kind": "sde",
            "build": _gbm_builder,
            "ic": [1.0],
            "final_time": 50.0,
            "dt": 1e-3,
            "method": "milstein",
            "seed": 12345,
            "v2": "reference",
        },
    }


def _ou_builder():
    """Ornstein–Uhlenbeck (additive noise) — a bench fixture, not a catalogue system."""
    from tsdynamics import StochasticSystem

    class OrnsteinUhlenbeck(StochasticSystem):
        params = {"theta": 1.0, "mu": 0.0, "sigma": 0.5}
        dim = 1
        variables = ("x",)

        @staticmethod
        def _drift(y, t, theta, mu, sigma):
            return [theta * (mu - y(0))]

        @staticmethod
        def _diffusion(y, t, theta, mu, sigma):
            return [sigma]

    return OrnsteinUhlenbeck()


def _gbm_builder():
    """Geometric Brownian motion (multiplicative noise) — a bench fixture."""
    from tsdynamics import StochasticSystem

    class GeometricBrownianMotion(StochasticSystem):
        params = {"mu": 0.15, "sigma": 0.3}
        dim = 1
        variables = ("x",)

        @staticmethod
        def _drift(y, t, mu, sigma):
            return [mu * y(0)]

        @staticmethod
        def _diffusion(y, t, mu, sigma):
            return [sigma * y(0)]

    return GeometricBrownianMotion()


def _resolve_ic(name: str, spec: dict, system) -> Any:
    """Resolve a deterministic initial condition for a spec (engine + v2 agree)."""
    import numpy as np

    if spec.get("ic") is not None:
        return list(spec["ic"])
    if name == "Lorenz96-N128":
        # A small reproducible perturbation off the f-valued fixed point.
        f = float(system.params["f"])
        u = np.full(system.dim, f, dtype=float)
        u[0] += 0.01
        return u.tolist()
    return [0.1] * system.dim


# ---------------------------------------------------------------------------
# The single dispatcher — used by both the cold worker and the warm path
# ---------------------------------------------------------------------------


def run_once(name: str, spec: dict, system, backend: str) -> Any:
    """Run one full integration of ``system`` on ``backend``; return its result.

    The one place that knows how each (kind, backend) pair is invoked, so the
    cold (subprocess) and warm (in-process) regimes call the same code.
    """
    kind = spec["kind"]
    if kind == "ode":
        return _run_ode(name, spec, system, backend)
    if kind == "dde":
        return _run_dde(spec, system, backend)
    if kind == "map":
        return _run_map(spec, system, backend)
    if kind == "sde":
        return _run_sde(spec, system, backend)
    raise ValueError(f"unknown kind {kind!r}")


def _run_ode(name: str, spec: dict, system, backend: str):
    import numpy as np

    ic = _resolve_ic(name, spec, system)
    ft, dt = spec["final_time"], spec["dt"]
    rtol, atol = spec["rtol"], spec["atol"]

    if backend == "scipy":
        from scipy.integrate import solve_ivp

        rhs = system._rhs_numeric()
        method = spec.get("scipy_method", spec["method"])
        sol = solve_ivp(
            lambda t, u: rhs(u, t),
            (0.0, ft),
            ic,
            t_eval=np.arange(0.0, ft, dt),
            method=method,
            rtol=rtol,
            atol=atol,
        )
        return sol.y.T

    if backend in ("interp", "jit"):
        method = spec["method"]
        if spec.get("stiff"):
            # The C-SOLV with_jacobian auto-merge is not wired into the family
            # seam yet; drive the engine seam directly so the implicit kernel
            # gets its Jacobian-carrying tape.
            from tsdynamics.engine import run

            return run.integrate(
                system,
                final_time=ft,
                dt=dt,
                ic=ic,
                method=method,
                rtol=rtol,
                atol=atol,
                backend=backend,
                with_jacobian=True,
            ).y
        return system.integrate(
            final_time=ft, dt=dt, ic=ic, method=method, rtol=rtol, atol=atol, backend=backend
        ).y

    if backend == "reference":
        method = spec.get("scipy_method", spec["method"]) if spec.get("stiff") else spec["method"]
        return system.integrate(
            final_time=ft, dt=dt, ic=ic, method=method, rtol=rtol, atol=atol, backend="reference"
        ).y

    # v2 backends (jitcode / diffsol)
    method = spec["v2_method"]
    return system.integrate(
        final_time=ft, dt=dt, ic=ic, method=method, rtol=rtol, atol=atol, backend=backend
    ).y


def _run_dde(spec: dict, system, backend: str):
    ft, dt = spec["final_time"], spec["dt"]
    rtol, atol = spec["rtol"], spec["atol"]
    c = spec["history_const"]
    history = lambda s: [c]  # noqa: E731 - constant past
    if backend in ("interp", "jit"):
        return system.integrate(
            final_time=ft,
            dt=dt,
            method=spec["method"],
            rtol=rtol,
            atol=atol,
            backend=backend,
            history=history,
        ).y
    # v2: jitcdde
    return system.integrate(
        final_time=ft, dt=dt, rtol=rtol, atol=atol, backend="jitcdde", history=history
    ).y


def _run_map(spec: dict, system, backend: str):
    return system.iterate(steps=spec["steps"], ic=list(spec["ic"]), backend=backend).y


def _run_sde(spec: dict, system, backend: str):
    return system.integrate(
        final_time=spec["final_time"],
        dt=spec["dt"],
        ic=list(spec["ic"]),
        method=spec["method"],
        seed=spec["seed"],
        backend=backend,
    ).y


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


def best_of(fn: Callable[[], Any], repeats: int) -> float:
    """Return the best (minimum) wall-clock time over ``repeats`` calls, seconds."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------------------
# Regime (b): warm steady-state throughput (in-process, best-of-N)
# ---------------------------------------------------------------------------


#: Pure-Python backends are orders of magnitude slower; cap their repeats and
#: skip them entirely on the large system so one warm sweep stays minutes, not
#: hours.  They are not the migration question (the engine vs the v2 defaults is).
_SLOW_BACKENDS = frozenset({"reference", "scipy"})


def warm_throughput(names: list[str], backends: list[str], repeats: int) -> dict:
    """Measure warm best-of-N wall time per (system, backend), warmup excluded."""
    systems = _build_systems()
    rows: dict[str, dict[str, float | None]] = {}
    for name in names:
        spec = systems[name]
        system = spec["build"]()
        rows[name] = {}
        for be in backends:
            if not _applies(spec, be):
                rows[name][be] = None
                continue
            if be in _SLOW_BACKENDS and spec.get("big"):
                rows[name][be] = None  # 128-dim pure-Python is impractically slow
                continue
            reps = min(repeats, 2) if be in _SLOW_BACKENDS else repeats
            try:
                # Warm up once (compile / JIT / Numba / cache) — untimed.
                run_once(name, spec, system, be)
                t = best_of(
                    lambda name=name, spec=spec, system=system, be=be: run_once(
                        name, spec, system, be
                    ),
                    reps,
                )
                rows[name][be] = t
            except Exception as exc:  # noqa: BLE001 - record and continue
                rows[name][be] = f"ERR: {type(exc).__name__}: {exc}"[:80]
        _log(f"  warm {name}: {rows[name]}")
    return rows


# ---------------------------------------------------------------------------
# Regime (c): ensemble / sweep throughput (in-process, best-of-N)
# ---------------------------------------------------------------------------


def ensemble_throughput(repeats: int, n_traj: int, quick: bool) -> dict:
    """Measure ensemble final-state throughput: engine rayon vs v2 loop vs SciPy loop."""
    import numpy as np

    import tsdynamics as ts
    from tsdynamics.engine import run

    rng = np.random.default_rng(0)
    rows: dict[str, dict[str, Any]] = {}

    # ODE ensemble: Lorenz (small) and Lorenz96-N128 (big) if not quick.
    ode_cases = [("Lorenz", lambda: ts.Lorenz(), 3, 5.0, 0.01, "RK45")]
    if not quick:
        ode_cases.append(("Lorenz96-N128", lambda: ts.Lorenz96(N=128), 128, 5.0, 0.01, "RK45"))

    for cname, ctor, dim, ft, dt, method in ode_cases:
        system = ctor()
        ics = rng.uniform(-1.0, 1.0, size=(n_traj, dim))
        if cname.startswith("Lorenz96"):
            ics = ics + float(system.params["f"])
        row: dict[str, Any] = {}

        def eng(be, system=system, ics=ics, ft=ft, dt=dt, method=method):
            return run.ensemble(
                system,
                ics,
                final_time=ft,
                dt=dt,
                method=method,
                rtol=1e-8,
                atol=1e-10,
                backend=be,
            )

        def v2_loop(system=system, ics=ics, ft=ft, dt=dt, method=method):
            # The realistic v2 ensemble: a Python loop over the compiled stepper.
            out = np.empty_like(ics)
            for i, ic in enumerate(ics):
                out[i] = system.integrate(
                    final_time=ft,
                    dt=dt,
                    ic=ic,
                    method=method,
                    rtol=1e-8,
                    atol=1e-10,
                    backend="jitcode",
                ).y[-1]
            return out

        def scipy_loop(system=system, ics=ics, ft=ft, dt=dt, method=method):
            from scipy.integrate import solve_ivp

            rhs = system._rhs_numeric()
            out = np.empty_like(ics)
            for i, ic in enumerate(ics):
                sol = solve_ivp(
                    lambda t, u: rhs(u, t),
                    (0.0, ft),
                    ic,
                    method="RK45",
                    rtol=1e-8,
                    atol=1e-10,
                )
                out[i] = sol.y[:, -1]
            return out

        for label, fn in (
            ("interp", lambda: eng("interp")),
            ("jit", lambda: eng("jit")),
            ("jitcode-loop", v2_loop),
            ("scipy-loop", scipy_loop),
        ):
            try:
                fn()  # warm
                row[label] = best_of(fn, repeats)
            except Exception as exc:  # noqa: BLE001
                row[label] = f"ERR: {type(exc).__name__}: {exc}"[:80]
        rows[f"{cname} (n={n_traj})"] = row
        _log(f"  ensemble {cname}: {row}")

    # SDE ensemble: engine rayon vs the pure-Python reference loop (the default).
    ou = _ou_builder()
    ics = np.full((n_traj, 1), 2.0)
    sde_ft, sde_dt = (5.0, 1e-3)
    row = {}
    for be in ("reference", "interp", "jit"):

        def fn(be=be):
            return ou.ensemble(
                ics, final_time=sde_ft, dt=sde_dt, method="euler_maruyama", seed=7, backend=be
            )

        try:
            fn()  # warm
            row[be] = best_of(fn, repeats)
        except Exception as exc:  # noqa: BLE001
            row[be] = f"ERR: {type(exc).__name__}: {exc}"[:80]
    rows[f"OU-SDE (n={n_traj})"] = row
    _log(f"  ensemble OU-SDE: {row}")
    return rows


# ---------------------------------------------------------------------------
# Regime (a): cold time-to-first-result (subprocess per sample)
# ---------------------------------------------------------------------------


#: The v2 backends that compile a system to a cached ``.so`` keyed in
#: ``TSDYNAMICS_CACHE`` — a fresh empty cache forces the true cold compile.  (The
#: Numba map backend has no cross-process cache, so a fresh process recompiles it
#: regardless; the engine/reference/scipy backends have no on-disk cache at all —
#: the Cranelift JIT recompiles per process, the interpreter has no warmup.)
_FRESH_CACHE_BACKENDS = frozenset({"jitcode", "jitcdde"})


def cold_first_result(names: list[str], backends: list[str], repeats: int) -> dict:
    """Spawn a fresh process per sample and time the first integration.

    Each family's v2 compile-to-cache backend (``jitcode`` for ODEs, ``jitcdde``
    for DDEs) is measured **truly cold** — a fresh empty ``TSDYNAMICS_CACHE`` per
    sample, so the C compiler actually runs (the honest first-ever cost).  The
    ``jitcode-warm`` column populates the cache once then reloads the cached
    ``.so`` — the typical repeat-use cost.  ``numba`` recompiles per fresh
    process inherently.  The engine (``interp``/``jit``), ``reference`` and
    ``scipy`` have no on-disk cache and are measured directly.
    """
    import tempfile

    systems = _build_systems()  # validates names; the worker rebuilds per process
    rows: dict[str, dict[str, Any]] = {}
    for name in names:
        spec = systems[name]
        rows[name] = {}
        for be in backends:
            base = "jitcode" if be == "jitcode-warm" else be
            if not _applies(spec, base):
                rows[name][be] = None
                continue
            if base in _SLOW_BACKENDS and spec.get("big"):
                rows[name][be] = None  # 128-dim pure-Python is impractically slow
                continue
            try:
                if be == "jitcode-warm":
                    cache = tempfile.mkdtemp(prefix="tsdbench-warm-")
                    _cold_sample(name, base, cache)  # populate the cache once
                    t = min(_cold_sample(name, base, cache)["first_s"] for _ in range(repeats))
                elif base in _FRESH_CACHE_BACKENDS:
                    # Truly cold: a fresh empty cache per sample → full compile.
                    samples = []
                    for _ in range(repeats):
                        cache = tempfile.mkdtemp(prefix="tsdbench-cold-")
                        samples.append(_cold_sample(name, base, cache)["first_s"])
                    t = min(samples)
                else:
                    t = min(_cold_sample(name, base, None)["first_s"] for _ in range(repeats))
                rows[name][be] = t
            except Exception as exc:  # noqa: BLE001
                rows[name][be] = f"ERR: {type(exc).__name__}: {exc}"[:80]
        _log(f"  cold {name}: {rows[name]}")
    return rows


def _cold_sample(name: str, backend: str, cache: str | None) -> dict:
    """Run one cold sample in a fresh subprocess; return its timing dict."""
    env = dict(os.environ)
    if cache is not None:
        env["TSDYNAMICS_CACHE"] = cache
    env.setdefault("OMP_NUM_THREADS", "1")  # cold first-result is single-trajectory
    out = subprocess.run(
        [sys.executable, THIS, "--worker", json.dumps({"name": name, "backend": backend})],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    try:
        data = json.loads(line)
    except (json.JSONDecodeError, IndexError) as err:
        raise RuntimeError(
            f"worker failed for {name}/{backend}: {out.stderr.strip()[-300:] or out.stdout[-300:]}"
        ) from err
    if not data.get("ok"):
        raise RuntimeError(f"worker error for {name}/{backend}: {data.get('error')}")
    return data


def _worker(payload: dict) -> None:
    """Subprocess entry: time `import tsdynamics` then one cold integration."""
    name, backend = payload["name"], payload["backend"]
    t0 = time.perf_counter()
    import tsdynamics  # noqa: F401  (timed)

    import_s = time.perf_counter() - t0
    try:
        systems = _build_systems()
        spec = systems[name]
        system = spec["build"]()
        t1 = time.perf_counter()
        run_once(name, spec, system, backend)
        first_s = time.perf_counter() - t1
        print(json.dumps({"ok": True, "import_s": import_s, "first_s": first_s}))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))


# ---------------------------------------------------------------------------
# Applicability + reporting
# ---------------------------------------------------------------------------


def _applies(spec: dict, backend: str) -> bool:
    """Whether ``backend`` is a valid measurement for this system's family."""
    kind = spec["kind"]
    if backend in ("interp", "jit"):
        return True
    if backend == "reference":
        return kind in ("ode", "sde")  # no DDE/map reference integrator path here
    if backend == "scipy":
        return bool(spec.get("scipy"))
    if backend in ("jitcode", "jitcode-warm"):
        return kind == "ode"
    if backend == "jitcdde":
        return kind == "dde"
    if backend == "numba":
        return kind == "map"
    return False


_VERBOSE = True


def _log(msg: str) -> None:
    if _VERBOSE:
        print(msg, file=sys.stderr, flush=True)


def machine_info() -> dict:
    """Collect a reproducibility block: CPU / OS / Python / dep versions."""
    import numpy as np
    import scipy

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    # CPU model name (Linux).
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import tsdynamics

        info["tsdynamics"] = tsdynamics.__version__
        import tsdynamics._rust as _rust

        info["engine"] = _rust._version()
        info["engine_solvers"] = sorted(_rust.solvers())
    except Exception as exc:  # noqa: BLE001
        info["engine"] = f"NOT BUILT ({type(exc).__name__})"
    return info


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if v >= 1.0:
        return f"{v:.3f}"
    if v >= 1e-3:
        return f"{v * 1e3:.2f} ms"
    return f"{v * 1e6:.1f} µs"


def _table(title: str, rows: dict, backends: list[str], note: str = "") -> str:
    out = [f"### {title}", ""]
    if note:
        out += [note, ""]
    out.append("| system | " + " | ".join(backends) + " |")
    out.append("|" + "---|" * (len(backends) + 1))
    for name, vals in rows.items():
        cells = [_fmt(vals.get(be)) for be in backends]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI args and run the requested regime(s) (or the cold-subprocess worker)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help="internal: JSON payload for a cold subprocess sample")
    parser.add_argument(
        "--regime",
        choices=["cold", "warm", "ensemble", "all"],
        default="all",
        help="which regime(s) to run",
    )
    parser.add_argument("--quick", action="store_true", help="CI-sized smoke (small + fewer reps)")
    parser.add_argument("--out", default=None, help="write the raw results as JSON to this path")
    args = parser.parse_args()

    if args.worker is not None:
        _worker(json.loads(args.worker))
        return

    info = machine_info()
    print("## Machine\n")
    for k, v in info.items():
        print(f"- **{k}**: {v}")
    print()
    if str(info.get("engine", "")).startswith("NOT BUILT"):
        print(
            "ERROR: the Rust engine extension (tsdynamics._rust) is not built — "
            "interp/jit cannot be measured. Build it first (see the module docstring).",
            file=sys.stderr,
        )
        sys.exit(2)

    quick = args.quick
    all_names = list(_build_systems().keys())
    if quick:
        names = ["Lorenz", "Oregonator", "MackeyGlass", "Henon", "OU"]
        warm_reps, cold_reps, ens_reps, n_traj = 2, 1, 1, 64
    else:
        names = all_names
        warm_reps, cold_reps, ens_reps, n_traj = 5, 2, 3, 512

    results: dict[str, Any] = {"machine": info, "quick": quick}

    if args.regime in ("warm", "all"):
        backs = ["jitcode", "interp", "jit", "reference", "scipy", "jitcdde", "numba"]
        print("\n## (b) Warm steady-state throughput — best wall time (lower is better)\n")
        rows = warm_throughput(names, backs, warm_reps)
        results["warm"] = rows
        print(_table("Warm throughput", rows, backs))

    if args.regime in ("ensemble", "all"):
        print("\n## (c) Ensemble / sweep throughput — best wall time (lower is better)\n")
        rows = ensemble_throughput(ens_reps, n_traj, quick)
        results["ensemble"] = rows
        # Ensemble columns vary per row family; print a wide table.
        cols = sorted({c for v in rows.values() for c in v})
        print(_table("Ensemble throughput", rows, cols))

    if args.regime in ("cold", "all"):
        backs = [
            "jitcode",
            "jitcode-warm",
            "jitcdde",
            "numba",
            "interp",
            "jit",
            "reference",
            "scipy",
        ]
        print("\n## (a) Cold time-to-first-result — fresh process, first integration\n")
        cold_names = (
            names
            if quick
            else ["Lorenz", "Rossler", "Oregonator", "Lorenz96-N128", "MackeyGlass", "Henon", "OU"]
        )
        # DDE/map/sde have no jitcode/scipy columns; the table prints — for those.
        rows = cold_first_result(cold_names, backs, cold_reps)
        results["cold"] = rows
        print(_table("Cold time-to-first-result", rows, backs))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"\nWrote raw results to {args.out}")


if __name__ == "__main__":
    main()
