"""Generate golden trajectory + Lyapunov spectrum files for every built-in map.

Runs the *current* (Numba) ``DiscreteMap.iterate`` and ``lyapunov_spectrum``
implementations and writes one ``.npz`` per map under
``tests/native/regression/``. These files become the reference that the
Rust-backed N1 pipeline must reproduce to within numerical tolerance.

Run once, commit the npz files, do NOT regenerate after the N1 rewrite lands.
If you ever need to regenerate (e.g. after a deliberate semantic change),
note the regeneration in CHANGELOG.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

# (module, class_name, dim, ic) — explicit ICs make goldens reproducible across
# Python sessions. Where the class defines `default_ic`, use that; otherwise
# pick a stable interior point.
_CHAOTIC = "tsdynamics.systems.discrete.chaotic_maps"
_EXOTIC = "tsdynamics.systems.discrete.exotic_maps"
_GEOMETRIC = "tsdynamics.systems.discrete.geometric_maps"
_POLY = "tsdynamics.systems.discrete.polynomial_maps"
_POPN = "tsdynamics.systems.discrete.population_maps"

MAPS: list[tuple[str, str, np.ndarray]] = [
    # Chaotic
    (_CHAOTIC, "Henon", np.array([0.1, 0.1])),
    (_CHAOTIC, "Ulam", np.array([0.3])),
    (_CHAOTIC, "Ikeda", np.array([0.1, 0.1])),
    (_CHAOTIC, "Tinkerbell", np.array([-0.72, -0.64])),  # default_ic
    (_CHAOTIC, "Gingerbreadman", np.array([0.5, 0.6])),
    (_CHAOTIC, "Zaslavskii", np.array([0.1, 0.1])),
    (_CHAOTIC, "Chirikov", np.array([0.1, 0.1])),
    (_CHAOTIC, "FoldedTowel", np.array([0.085, -0.121, 0.075])),
    (_CHAOTIC, "GeneralizedHenon", np.array([0.0, 0.0, 0.0])),
    # Exotic
    (_EXOTIC, "Bogdanov", np.array([0.1, 0.1])),
    (_EXOTIC, "Svensson", np.array([0.1, 0.1])),
    (_EXOTIC, "Bedhead", np.array([0.1, 0.1])),
    (_EXOTIC, "ZeraouliaSprott", np.array([0.1, 0.1])),
    (_EXOTIC, "GumowskiMira", np.array([0.1, 0.1])),
    (_EXOTIC, "Hopalong", np.array([0.1, 0.1])),
    (_EXOTIC, "Pickover", np.array([0.1, 0.1])),
    # Geometric
    (_GEOMETRIC, "Tent", np.array([0.3])),
    (_GEOMETRIC, "Baker", np.array([0.2, 0.3])),
    (_GEOMETRIC, "Circle", np.array([0.3])),
    (_GEOMETRIC, "Chebyshev", np.array([0.3])),
    # Polynomial
    (_POLY, "Gauss", np.array([0.3])),
    (_POLY, "DeJong", np.array([0.1, 0.1])),
    (_POLY, "KaplanYorke", np.array([0.1, 0.1])),
    # Population
    (_POPN, "Logistic", np.array([0.3])),
    (_POPN, "Ricker", np.array([0.3])),
    (_POPN, "MaynardSmith", np.array([0.1, 0.1])),
]

STEPS_ITERATE = 10_000
STEPS_LYAP = 10_000
SEED = 0


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "tests" / "native" / "regression"
    out_dir.mkdir(parents=True, exist_ok=True)

    for module_name, class_name, ic in MAPS:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        sys = cls()
        params_tuple = sys.params.as_tuple()

        np.random.seed(SEED)
        traj = sys.iterate(steps=STEPS_ITERATE, ic=ic.copy())

        # Reset internal IC (iterate's retry path overwrites self.ic on failure)
        object.__setattr__(sys, "ic", None)

        np.random.seed(SEED)
        lyap = sys.lyapunov_spectrum(steps=STEPS_LYAP, ic=ic.copy())

        out_path = out_dir / f"{class_name}.npz"
        np.savez(
            out_path,
            trajectory=traj.y,
            lyapunov=lyap,
            ic=ic,
            params=np.array(params_tuple, dtype=float),
            steps_iterate=STEPS_ITERATE,
            steps_lyap=STEPS_LYAP,
            seed=SEED,
        )
        print(f"  {class_name:<20} trajectory={traj.y.shape} lyap={lyap}")

    print(f"\nWrote {len(MAPS)} golden files to {out_dir}")


if __name__ == "__main__":
    main()
