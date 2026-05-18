"""Regression: Lyapunov spectra vs a **small** set of ``*.lyap.npz`` goldens (**N3**).

Full committed goldens (~100+ files) remain for manual or dedicated regression
runs; CI only smoke-tests a representative slice so ``pytest -m slow`` stays
affordable.  Regenerate all files with:

``uv run python scripts/generate_ode_lyap_goldens.py`` (or ``--only-missing``).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_ode_systems import ALL_ODE_SYSTEMS  # noqa: E402

_REG = Path(__file__).resolve().parent / "native" / "regression" / "ode"

_ID_BY_CLASS = {name: (mod, name) for mod, name in ALL_ODE_SYSTEMS}

# Representative chaos + dim-3 / dim-4 / Sprott family — must have matching
# ``<name>.lyap.npz`` on disk.
_LYAP_GOLDEN_SMOKE: frozenset[str] = frozenset(
    {
        "Lorenz",
        "Rossler",
        "HenonHeiles",
        "HyperRossler",
        "Halvorsen",
        "SprottA",
    }
)


def _smoke_golden_paths() -> list[Path]:
    missing = sorted(
        _LYAP_GOLDEN_SMOKE - {p.name.removesuffix(".lyap.npz") for p in _REG.glob("*.lyap.npz")}
    )
    if missing:
        pytest.fail(
            "Lyapunov smoke goldens missing on disk: " + ", ".join(missing) + f" (under {_REG})"
        )
    return sorted(_REG / f"{name}.lyap.npz" for name in sorted(_LYAP_GOLDEN_SMOKE))


@pytest.mark.slow
@pytest.mark.parametrize(
    "golden_path", _smoke_golden_paths(), ids=lambda p: p.stem.replace(".lyap", "")
)
def test_lyapunov_matches_golden_smoke(golden_path: Path) -> None:
    stem = golden_path.name.removesuffix(".lyap.npz")
    pair = _ID_BY_CLASS.get(stem)
    if pair is None:
        pytest.fail(f"no ALL_ODE_SYSTEMS entry for golden {golden_path.name}")

    g = np.load(golden_path, allow_pickle=True)
    ref = np.asarray(g["lyapunov"], dtype=float)
    ic = np.asarray(g["ic"], dtype=float)
    burn_in = float(g["burn_in"])
    final_time = float(g["final_time"])
    dt = float(g["dt"])
    method = str(np.asarray(g["method"]).item())
    rtol = float(g["rtol"])
    atol = float(g["atol"])
    n_exp = int(g["n_exp"])

    mod = importlib.import_module(pair[0])
    cls = getattr(mod, pair[1])
    sys = cls()
    exps = sys.lyapunov_spectrum(
        final_time=final_time,
        dt=dt,
        ic=ic.copy(),
        n_exp=n_exp,
        burn_in=burn_in,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    assert exps.shape == ref.shape, (exps.shape, ref.shape)
    assert np.allclose(exps, ref, rtol=1e-3, atol=1e-3), (
        f"{stem} max abs diff {np.max(np.abs(exps - ref))}: got {exps}, ref {ref}"
    )
