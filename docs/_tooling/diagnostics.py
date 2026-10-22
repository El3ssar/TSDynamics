"""Reproducible numerical evidence for the system catalogue.

Run ``uv run python docs/_tooling/diagnostics.py --refresh --workers 2`` to
regenerate ``diagnostics_data.json``. Each system runs in an isolated process;
the timeout is a computation budget, not a claim about its dynamics. Ordinary
documentation builds reuse matching records instead of integrating the whole
catalogue. Changing a model, parameters, estimator, or this protocol invalidates
its record.

These are finite-time measurements at the shipped parameters, from a recorded
initial condition. They are not certified asymptotic exponents or proofs of
chaos. Published reference values are retained separately, including their
parameter overrides. A missing or inapplicable measurement never becomes zero.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
import warnings
import zlib
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(__file__).with_name("diagnostics_data.json")
SCHEMA_VERSION = 1
TIMEOUT_SECONDS = 180.0

# In-basin starts used by the independent catalogue dynamics gate. These are
# inputs, not numerical answers; retaining them here makes the published run
# reproducible without importing test code into documentation tooling.
INITIAL_CONDITIONS = {
    "TurchinHanski": [0.5, 0.5, 0.0],
    "CoevolvingPredatorPrey": [0.5, 0.3, 0.2],
    "HastingsPowell": [0.8, 0.2, 8.0],
    "SprottL": [0.1, 0.1, 0.1],
    "SprottJerk": [0.0, 0.0, 1.0],
    "SprottE": [0.1, 0.1, 0.1],
    "ItikBanksTumor": [0.1, 0.1, 0.1],
    "CellCycle": [0.01, 0.01, 0.01, 0.01, 0.9, 0.9],
    "Oregonator": [0.1, 0.1, 0.1],
}

# Non-equilibrium histories from the independently tested DDE catalogue inputs.
# A generic large history leaves Voss's small physical basin.
DDE_HISTORIES = {
    "MackeyGlass": (1.0, 0.1, 0.2, "sin"),
    "IkedaDelay": (0.1, 0.05, 0.3, "cos"),
    "SprottDelay": (0.5, 0.1, 0.2, "sin"),
    "ScrollDelay": (0.3, 0.05, 0.1, "cos"),
    "PiecewiseCircuit": (0.4, 0.05, 0.15, "sin"),
    "VossDelay": (0.15, 0.1, 0.2, "sin"),
}


def unavailable(reason: str, *, applicable: bool = True, **extra: Any) -> dict[str, Any]:
    """Describe an absent answer without inventing a numerical value."""
    return {
        "status": "unavailable" if applicable else "not_applicable",
        "reason": reason,
        **extra,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    # Catalogue parameters include exact SymEngine constants such as 2*pi/3.
    # Their evaluated value is what the numerical engine receives.
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def fingerprint(entry: Any) -> str:
    """Hash the model and numerical implementation, not only its class body."""
    import tsdynamics

    package = Path(tsdynamics.__file__).parent
    paths = sorted((package / "analysis").rglob("*.py"))
    paths += sorted((package / "families").rglob("*.py"))
    paths += sorted((package / "_engine").rglob("*.py"))
    paths += sorted((package / "_utils").rglob("*.py"))
    paths += [package / "derived/tangent.py", package / "derived/_variational.py"]
    paths += sorted((package / "systems").glob("_*.py"))
    for cls in entry.cls.__mro__:
        if cls.__module__.startswith("tsdynamics.systems"):
            filename = inspect.getsourcefile(cls)
            if filename is not None:
                paths.append(Path(filename))
    # Native QR/integration changes alter the numerical evidence just as much
    # as a Python estimator change; a Python-only key would silently retain old
    # results across an engine repair.
    for crate in (ROOT / "crates").iterdir():
        if crate.is_dir() and crate.name != "target":
            paths += sorted((crate / "src").rglob("*.rs"))
            paths += [p for p in (crate / "Cargo.toml", crate / "build.rs") if p.exists()]
    protocol = Path(__file__).resolve()
    paths += [ROOT / "crates/Cargo.toml", ROOT / "crates/Cargo.lock", protocol]

    def label(path: Path) -> str:
        if path == protocol:
            return "protocol/diagnostics.py"
        if path.is_relative_to(package):
            return "package/" + path.relative_to(package).as_posix()
        return "native/" + path.relative_to(ROOT / "crates").as_posix()

    paths = sorted(set(paths), key=label)
    signature = tuple((label(p), str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
    digest = hashlib.sha256(_implementation_digest(signature))
    for cls in entry.cls.__mro__:
        if cls is object or cls.__module__.startswith("tsdynamics.families"):
            continue  # framework sources are included in the shared digest
        digest.update(_class_digest(cls))
    digest.update(json.dumps(_jsonable(entry.params), sort_keys=True).encode())
    return digest.hexdigest()


@lru_cache(maxsize=512)
def _class_digest(cls: type) -> bytes:
    """Include inherited catalogue equations, not just a child's override body."""
    return _python_semantics(inspect.getsource(cls))


class _NumericalSource(ast.NodeTransformer):
    """Exclude prose and display-only hooks from numerical cache invalidation."""

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        display = {
            "_answer",
            "_interpretation",
            "_context",
            "_details",
            "_item_lines",
            "_as_item",
            "_derived",
            "__repr__",
            "__str__",
            "_repr_html_",
            "__plot_spec__",
            "__dir__",
        }
        metadata = {"reference", "doi", "_reference", "_doi", "_known_lyapunov", "known_lyapunov"}
        body = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in display:
                continue
            targets = (
                item.targets
                if isinstance(item, ast.Assign)
                else [item.target]
                if isinstance(item, ast.AnnAssign)
                else []
            )
            if any(isinstance(target, ast.Name) and target.id in metadata for target in targets):
                continue
            body.append(item)
        node.body = body
        return self.generic_visit(node)


def _canonical_ast(value: Any) -> Any:
    if isinstance(value, ast.AST):
        # Ignore optional empty fields added by newer Python AST versions (e.g.
        # type_params) so an unchanged 3.12/3.14 source has the same fingerprint.
        fields = {
            name: _canonical_ast(item)
            for name, item in ast.iter_fields(value)
            if item is not None and not (isinstance(item, list) and not item)
        }
        return [type(value).__name__, fields]
    if isinstance(value, list):
        return [_canonical_ast(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if value is Ellipsis:
        return {"ellipsis": True}
    return value


def _python_semantics(source: str) -> bytes:
    tree = _NumericalSource().visit(ast.parse(source))
    return json.dumps(_canonical_ast(tree), sort_keys=True, separators=(",", ":")).encode()


@lru_cache(maxsize=64)
def _implementation_digest(signature: tuple[tuple[str, str, int, int], ...]) -> bytes:
    digest = hashlib.sha256()
    for label, filename, modified, size in signature:
        digest.update(label.encode() + b"\0")
        digest.update(_file_digest(filename, modified, size))
    return digest.digest()


@lru_cache(maxsize=1024)
def _file_digest(filename: str, modified: int, size: int) -> bytes:
    path = Path(filename)
    source = path.read_text()  # universal-newline normalization (including Windows CRLF)
    semantic = _python_semantics(source) if path.suffix == ".py" else source.encode()
    return hashlib.sha256(semantic).digest()


def reference(entry: Any) -> dict[str, Any] | None:
    """Keep reference conditions beside the values rather than relabel defaults."""
    from tsdynamics import registry

    meta = registry._classvar(entry.cls, "known_lyapunov")
    if not meta:
        return None
    return {"status": "reference", **_jsonable(dict(meta)), "citation": entry.reference}


def _explicit_time(system: Any) -> bool:
    """Whether the concrete ODE, with its current parameters, uses time."""
    from tsdynamics.analysis._tangent import flow_has_explicit_time

    return flow_has_explicit_time(system)


def _initial_condition(entry: Any, system: Any, seed: int) -> np.ndarray:
    from tsdynamics import registry

    initial = INITIAL_CONDITIONS.get(entry.name)
    if initial is None:
        initial = registry._classvar(entry.cls, "default_ic")
    if initial is not None:
        return np.asarray(initial, dtype=float)
    return 0.1 * (2.0 * np.random.default_rng(seed).random(system.dim) - 1.0)


def _spectrum(
    entry: Any, system: Any, seed: int, forced: bool, *, leading: int | None = None
) -> dict[str, Any]:
    import tsdynamics as ts
    from tsdynamics._utils.tolerances import DEFAULT_ATOL, DEFAULT_RTOL

    if entry.family == "sde":
        return unavailable(
            "A stochastic Lyapunov exponent needs a shared-noise tangent estimator; "
            "the deterministic spectrum estimator does not implement that method."
        )
    dim = int(system.dim)
    if dim > 128:
        return unavailable(
            f"The default spatial discretisation has {dim} states. Its tangent spectrum "
            "is outside this catalogue computation budget; choose a grid and k explicitly."
        )
    settings: dict[str, Any] = {"backend": "jit"}
    if entry.family == "map":
        # Save both the original start and the post-burn-in start. A seeded
        # map's retry can legitimately change its initial condition.
        warmup = system.run(steps=500, seed=seed, max_retries=20)
        initial = np.asarray(warmup.y[-1], dtype=float)
        settings.update(k=dim, n=20_000, transient=0, ic=initial.tolist())
        preparation = {
            "steps": 500,
            "seed": seed,
            "ic": np.asarray(warmup.y[0], dtype=float).tolist(),
        }
        unit = "per iteration"
    elif entry.family == "dde":
        delay = max(system._delays())
        step = min(0.1, delay / 40.0)

        offset, amplitude, frequency, trig = DDE_HISTORIES[entry.name]

        def history(t: float) -> list[float]:
            fn = np.sin if trig == "sin" else np.cos
            return [offset + amplitude * fn(frequency * t)] * dim

        warmup = system.run(final_time=400.0, dt=step, history=history)
        initial = np.asarray(warmup.y[-1], dtype=float)
        count = {"IkedaDelay": 20, "VossDelay": 32}.get(entry.name, 6)
        settings.update(k=count, final_time=2000.0, transient=400.0, dt=step, ic=initial.tolist())
        preparation = {
            "final_time": 400.0,
            "dt": step,
            "history": f"each coordinate: {offset} + {amplitude}*{trig}({frequency}*t), t <= 0",
            "note": "The estimator starts with constant past at the warm-up endpoint; "
            "its own transient discards that history reset.",
        }
        unit = "per time unit"
    else:
        initial = _initial_condition(entry, system, seed)
        horizon, transient, step = (400.0, 100.0, 0.1)
        if entry.name == "BickleyJet":
            horizon, transient, step = (40_000.0, 10_000.0, 16.0)
        settings.update(
            k=dim if dim <= 64 else 1,
            final_time=horizon,
            transient=transient,
            dt=step,
            ic=initial.tolist(),
            rtol=DEFAULT_RTOL,
            atol=DEFAULT_ATOL,
            solver=system._default_method,
        )
        preparation = {"seed": seed, "ic_policy": "declared/in-basin start, else seeded small ball"}
        if entry.name == "Oregonator":
            settings["k"] = 1
            preparation["spectrum_scope"] = (
                "The stiff full variational frame fails at practical QR intervals "
                "with both native BDF and the independent SciPy-BDF reference. "
                "Only the feasible leading BDF rate is measured here."
            )
        unit = "per time unit"
    started = time.perf_counter()
    if leading is not None:
        settings["k"] = min(leading, dim) if entry.family != "dde" else leading
    try:
        result = ts.analysis.lyapunov_spectrum(system, **settings)
    except Exception as exc:
        return unavailable(
            f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            settings=settings,
            preparation=preparation,
            method="Benettin QR tangent dynamics",
        )
    values = np.asarray(result, dtype=float)
    if not values.size or not np.isfinite(values).all():
        return unavailable(
            "The finite-time estimator returned non-finite exponents.", settings=settings
        )
    if result.unbounded:
        return unavailable(
            "The recorded orbit escaped; its finite-time rates are not attractor exponents.",
            settings=settings,
            evidence={"orbit_peak": result.meta.get("orbit_peak")},
        )
    complete = entry.family != "dde" and values.size == dim
    return {
        "status": "measured",
        "origin": "computed",
        "value": values.tolist(),
        "complete": complete,
        "state_dimension": dim,
        "unit": unit,
        "settings": settings,
        "preparation": preparation,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "method": "Benettin QR tangent dynamics"
        if entry.family != "dde"
        else "history-space Benettin QR",
        "note": "Finite-time estimate, not a convergence certificate. "
        + (
            "Response directions only; the forcing phase is not an extra state."
            if forced
            else "Only the leading directions were computed."
            if not complete
            else "All state-space directions were computed."
        ),
    }


def kaplan_yorke(card: dict[str, Any], *, forced: bool = False) -> dict[str, Any]:
    """Derive a dimension only when enough leading exponents were measured."""
    import tsdynamics as ts

    if forced:
        return unavailable(
            "Explicit forcing: the response spectrum omits the forcing phase. "
            "An autonomous extension is needed for an attractor dimension.",
            applicable=False,
        )
    if card.get("status") not in ("measured", "reference") or "value" not in card:
        return unavailable("A numerical Lyapunov spectrum is required.")
    values = np.sort(np.asarray(card["value"], dtype=float))[::-1]
    if not card.get("complete") and np.sum(values) >= 0.0:
        return unavailable(
            "The leading spectrum does not reach a negative cumulative sum. "
            "More exponents are needed; its length is not a dimension."
        )
    return {
        "status": card["status"],
        "origin": card.get("origin"),
        "value": float(ts.analysis.kaplan_yorke_dimension(values)),
        "method": "Kaplan–Yorke interpolation of the recorded spectrum",
        "note": "Lyapunov dimension estimate; equality with a fractal or information "
        "dimension requires additional dynamical assumptions.",
        "settings": card.get("settings", {}),
    }


def _equilibria(entry: Any, system: Any, *, forced: bool) -> dict[str, Any]:
    import tsdynamics as ts

    if entry.family == "sde":
        return unavailable(
            "Sample paths have no deterministic equilibria; drift zeros describe a different object.",
            applicable=False,
        )
    if entry.family == "dde":
        return unavailable(
            "Delay-equation equilibria and their history-space stability are not implemented by fixed_points."
        )
    if forced:
        return unavailable(
            "The equations depend explicitly on time. A root of f(x, 0) is not an equilibrium of the forced system.",
            applicable=False,
        )
    if system.dim > 64:
        return unavailable(
            f"A {system.dim}-state multistart root search is outside this catalogue budget."
        )
    settings = {"method": "newton", "n_seeds": 80, "seed": 0}
    started = time.perf_counter()
    roots = ts.analysis.fixed_points(system, **settings)
    stable = unstable = marginal = 0
    for point in roots.details:
        eigenvalues = np.asarray(point.eigenvalues)
        tolerance = 1e-7 * max(1.0, float(np.max(np.abs(eigenvalues))))
        lead = (
            float(np.max(eigenvalues.real))
            if entry.family == "ode"
            else float(np.max(np.abs(eigenvalues))) - 1.0
        )
        if lead < -tolerance:
            stable += 1
        elif lead > tolerance:
            unstable += 1
        else:
            marginal += 1
    return {
        "status": "measured",
        "origin": "computed",
        "value": len(roots),
        "stable": stable,
        "unstable": unstable,
        "marginal": marginal,
        "stability_test": "max Re(eigenvalue) for flows; max |multiplier| - 1 for maps. "
        "Boundary band: 1e-7 * max(1, max |eigenvalue|).",
        "kind": "fixed points found" if entry.family == "map" else "equilibria found",
        "settings": settings,
        "points": np.asarray(roots).tolist(),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "note": "A seeded numerical search, not a completeness proof. Stability means "
        "linear asymptotic stability. Marginal/nonhyperbolic roots have no nonlinear "
        "stability verdict from this linearisation.",
    }


def _worker(entry: Any, *, checkpoint: bool = False, leading: int | None = None) -> dict[str, Any]:
    """Compute evidence independently and retain roots before an expensive spectrum."""
    import scipy

    import tsdynamics as ts

    system = entry.cls()
    seed = zlib.crc32(entry.name.encode()) & 0x7FFFFFFF
    forced = entry.family in ("ode", "dde") and _explicit_time(system)
    record = {
        "name": entry.name,
        "family": entry.family,
        "fingerprint": fingerprint(entry),
        "provenance": {
            "parameters": _jsonable(dict(system.params)),
            "parameter_expressions": {
                key: str(value)
                for key, value in system.params.items()
                if hasattr(value, "free_symbols")
            },
            "state_dimension": system.dim,
            "explicit_time": forced,
            "library_version": ts.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "protocol_version": SCHEMA_VERSION,
        },
        "reference": reference(entry),
        **{
            key: unavailable("Computation not finished.")
            for key in ("lyapunov_spectrum", "kaplan_yorke", "equilibria")
        },
    }
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        calculations = (
            ("equilibria", lambda: _equilibria(entry, system, forced=forced)),
            ("lyapunov_spectrum", lambda: _spectrum(entry, system, seed, forced, leading=leading)),
        )
        for name, run in calculations:
            if leading is not None and name == "equilibria":
                continue
            try:
                record[name] = run()
            except Exception as exc:
                record[name] = unavailable(f"{type(exc).__name__}: {str(exc).splitlines()[0]}")
            if checkpoint and name == "equilibria":
                print(json.dumps(record, allow_nan=False), flush=True)
    record["kaplan_yorke"] = kaplan_yorke(record["lyapunov_spectrum"], forced=forced)
    return record


def _last_record(output: str | bytes | None) -> dict[str, Any] | None:
    """Recover flushed evidence even if the later estimator is interrupted."""
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    for line in reversed((output or "").splitlines()):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and "equilibria" in record:
            return record
    return None


def _in_subprocess(
    entry: Any, timeout: float = TIMEOUT_SECONDS, *, leading: int | None = None
) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", entry.name]
    if leading is not None:
        command += ["--leading", str(leading)]
    partial = None
    try:
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        partial = _last_record(process.stdout)
        if process.returncode:
            raise RuntimeError(f"worker exited with code {process.returncode}")
        if partial is None:
            raise ValueError("worker returned no valid record")
        return (
            _leading_fallback(entry, partial, full_timeout=timeout) if leading is None else partial
        )
    except (subprocess.SubprocessError, ValueError, IndexError, RuntimeError) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            partial = _last_record(exc.stdout)
            reason = f"Computation exceeded {timeout:g}s (worker stopped)."
        else:
            reason = f"Numerical worker failed: {exc}"
        if partial is not None:
            partial["lyapunov_spectrum"] = unavailable(reason)
            partial["kaplan_yorke"] = unavailable(reason)
            return (
                _leading_fallback(entry, partial, full_timeout=timeout)
                if leading is None
                else partial
            )
        record = {
            "name": entry.name,
            "family": entry.family,
            "fingerprint": fingerprint(entry),
            "reference": reference(entry),
            **{
                key: unavailable(reason)
                for key in ("lyapunov_spectrum", "kaplan_yorke", "equilibria")
            },
        }
        return _leading_fallback(entry, record, full_timeout=timeout) if leading is None else record


def _leading_fallback(entry: Any, record: dict[str, Any], *, full_timeout: float) -> dict[str, Any]:
    """Keep feasible leading rates after one bounded full-frame attempt fails."""
    full = record["lyapunov_spectrum"]
    if entry.family != "ode" or full.get("status") != "unavailable":
        return record
    reason = full.get("reason", "")
    if "exceeded" not in reason and entry.name != "Oregonator":
        return record
    dim = entry.cls().dim
    count = min(6, dim) if dim > 6 else 1
    attempts = []
    choices = [(count, 60.0)] + ([(1, 30.0)] if count > 1 else [])
    for leading, budget in choices:
        fallback = _in_subprocess(entry, timeout=budget, leading=leading)
        card = fallback["lyapunov_spectrum"]
        attempts.append(
            {
                "k": leading,
                "timeout_seconds": budget,
                "status": card["status"],
                "reason": card.get("reason"),
            }
        )
        if card["status"] == "measured":
            if "provenance" not in record:
                record["provenance"] = fallback["provenance"]
            record["lyapunov_spectrum"] = card
            card["full_spectrum_attempt"] = {
                "reason": reason,
                "settings": full.get("settings", {}),
                "timeout_seconds": full_timeout,
            }
            if len(attempts) > 1:
                card["leading_attempt"] = {"attempts": attempts}
            record["kaplan_yorke"] = fallback["kaplan_yorke"]
            break
    else:
        record["lyapunov_spectrum"]["leading_attempt"] = {"attempts": attempts}

    return record


def load_records() -> dict[str, Any]:
    """Read the versioned, reproducible dataset; corrupt files fail visibly."""
    if not DATA_PATH.exists():
        return {}
    payload = json.loads(DATA_PATH.read_text())
    return payload.get("systems", {}) if payload.get("schema_version") == SCHEMA_VERSION else {}


def compute_diagnostics(entry: Any) -> dict[str, Any]:
    """Return matching recorded evidence, or compute the missing system once."""
    saved = load_records().get(entry.name)
    if saved is not None and saved.get("fingerprint") == fingerprint(entry):
        # Reference conditions may be corrected without changing equations or
        # numerical settings. Refresh that separate metadata, never relabel its
        # values as measurements or discard the original numerical provenance.
        return {**saved, "reference": reference(entry)}
    if os.environ.get("TSD_DOCS_PROPS", "1") == "0":
        return {
            "name": entry.name,
            "family": entry.family,
            "reference": reference(entry),
            **{
                key: unavailable(
                    "No current recorded result; recomputation is disabled (TSD_DOCS_PROPS=0)."
                )
                for key in ("lyapunov_spectrum", "kaplan_yorke", "equilibria")
            },
        }
    return _in_subprocess(entry)


def main() -> None:
    """Regenerate catalogue evidence through bounded isolated workers."""
    from tsdynamics import registry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Recompute even matching records.")
    parser.add_argument("--only", nargs="+", help="System names to compute.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--leading", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    entries = registry.all_systems()
    if args.worker:
        entry = next(e for e in entries if e.name == args.worker)
        print(
            json.dumps(_worker(entry, checkpoint=True, leading=args.leading), allow_nan=False),
            flush=True,
        )
        return
    if args.workers < 1 or args.timeout <= 0 or not np.isfinite(args.timeout):
        parser.error("workers and timeout must be positive (timeout must be finite)")
    if args.only:
        unknown = set(args.only) - {e.name for e in entries}
        if unknown:
            parser.error(f"Unknown systems: {', '.join(sorted(unknown))}")
        entries = [e for e in entries if e.name in args.only]
    records = load_records()
    work = [
        e
        for e in entries
        if args.refresh or records.get(e.name, {}).get("fingerprint") != fingerprint(e)
    ]
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    try:
        pending = {executor.submit(_in_subprocess, e, args.timeout): e for e in work}
        for future in concurrent.futures.as_completed(pending):
            entry = pending[future]
            record = future.result()
            records[entry.name] = record
            payload = {"schema_version": SCHEMA_VERSION, "systems": dict(sorted(records.items()))}
            temporary = DATA_PATH.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            temporary.replace(DATA_PATH)
            print(
                entry.name,
                {
                    k: record[k]["status"]
                    for k in ("lyapunov_spectrum", "kaplan_yorke", "equilibria")
                },
                flush=True,
            )

    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
