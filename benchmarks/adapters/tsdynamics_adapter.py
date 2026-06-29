"""TSDynamics adapter — the library under test.

Constructed with a ``backend`` (``"interp"`` the zero-warmup default, or
``"jit"`` the Cranelift hot path); the orchestrator registers it twice so both
appear as columns. Backend only affects the engine-driven tasks (the three
integrations, the system Lyapunov family, basins, Poincaré); the from-data
analyses (correlation dimension, Lyapunov-from-data) are pure NumPy/SciPy and so
read identically across the two columns.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import series

from adapters._base import BaseAdapter


def _make_newton_map() -> Any:
    """Define the Newton root-finding map on f(z)=z³−1 as a TSDynamics map.

    Iterating ``z ← z − (z³−1)/(3z²)`` on the complex plane converges to one of
    the three cube roots of unity; the (x,y)=Re/Im real form below is what the
    engine lowers. Defined lazily so importing this module never requires the
    compiled engine.
    """
    import symengine as se

    import tsdynamics as ts

    def step_exprs(x: Any, y: Any) -> tuple[Any, Any]:
        a = x**3 - 3 * x * y**2 - 1  # Re(z³ − 1)
        b = 3 * x**2 * y - y**3  # Im(z³ − 1)
        c = 3 * x**2 - 3 * y**2  # Re(3z²)
        d = 6 * x * y  # Im(3z²)
        den = c * c + d * d
        return x - (a * c + b * d) / den, y - (b * c - a * d) / den

    class NewtonZ3(ts.DiscreteMap):
        """Newton's method for z³ − 1 = 0, in real (x, y) coordinates."""

        variables = ("x", "y")
        dim = 2
        params: dict[str, float] = {}
        _jacobian_fd_check = False  # the map has a pole at the origin

        @staticmethod
        def _step(X):  # type: ignore[no-untyped-def]
            return list(step_exprs(X[0], X[1]))

        @staticmethod
        def _jacobian(X):  # type: ignore[no-untyped-def]
            x, y = X[0], X[1]
            f0, f1 = step_exprs(x, y)
            return [[se.diff(f0, x), se.diff(f0, y)], [se.diff(f1, x), se.diff(f1, y)]]

    return NewtonZ3


class TSDynamicsAdapter(BaseAdapter):
    language = "python"

    def __init__(self, config_dict: dict[str, Any], *, backend: str = "interp") -> None:
        super().__init__(config_dict)
        self.backend = backend
        self.name = f"TSDynamics ({backend})"

    def _probe(self) -> tuple[bool, str, str]:
        import tsdynamics as ts

        return True, getattr(ts, "__version__", "?"), ""

    # -- helpers ------------------------------------------------------------ #

    def _lorenz(self) -> Any:
        import tsdynamics as ts

        return ts.systems.Lorenz().with_params(**self.cfg["lorenz"]["params"])

    # -- integration -------------------------------------------------------- #

    def task_integrate_short(self, quick: bool) -> Callable[[], Any]:
        lor = self._lorenz()
        ic = self.cfg["lorenz"]["ic"]
        T = 50.0 if quick else self._intg["t_short"]
        dt = self._intg["dt"]

        def run() -> None:
            lor.integrate(
                ic=ic,
                final_time=T,
                dt=dt,
                method="dop853",
                rtol=1e-9,
                atol=1e-9,
                backend=self.backend,
            )

        return run

    def task_integrate_long(self, quick: bool) -> Callable[[], Any]:
        lor = self._lorenz()
        ic = self.cfg["lorenz"]["ic"]
        T = 1000.0 if quick else self._intg["t_long"]
        dt = self._intg["dt"]

        def run() -> None:
            lor.integrate(
                ic=ic,
                final_time=T,
                dt=dt,
                method="dop853",
                rtol=1e-9,
                atol=1e-9,
                backend=self.backend,
            )

        return run

    def task_integrate_accuracy(self, quick: bool) -> Callable[[], float]:
        lor = self._lorenz()
        ic = self.cfg["lorenz"]["ic"]
        T = self._intg["t_acc"]
        ref = np.asarray(self._ref["lorenz_acc_final"], dtype=float)

        def run() -> float:
            tr = lor.integrate(
                ic=ic,
                final_time=T,
                dt=T,
                method="dop853",
                rtol=self._intg["acc_rtol"],
                atol=self._intg["acc_atol"],
                backend=self.backend,
            )
            return float(np.max(np.abs(np.asarray(tr.y[-1], dtype=float) - ref)))

        return run

    # -- Lyapunov ----------------------------------------------------------- #

    def task_lyapunov_spectrum(self, quick: bool) -> Callable[[], float]:
        import tsdynamics as ts

        lor = self._lorenz()
        ic = self.cfg["lorenz"]["ic"]
        T = 200.0 if quick else 500.0

        def run() -> float:
            # dt=0.05 is the renormalisation interval (comparable to the Julia
            # column's Δt); it gives the best λ_max accuracy here and is ~5×
            # faster than dt=0.01 (no accuracy gain below 0.05).
            ls = ts.lyapunov_spectrum(lor, final_time=T, dt=0.05, ic=ic, transient=20.0)
            return float(np.max(np.asarray(ls.exponents)))

        return run

    def task_max_lyapunov(self, quick: bool) -> Callable[[], float]:
        import tsdynamics as ts

        h = ts.systems.Henon().with_params(**self.cfg["henon"]["params"])
        ic = self.cfg["henon"]["ic"]
        n = 2000 if quick else 5000

        def run() -> float:
            return float(ts.max_lyapunov(h, ic=ic, n=n, seed=0))

        return run

    def task_lyapunov_from_data(self, quick: bool) -> Callable[[], float]:
        import tsdynamics as ts

        s = self.cfg["series"]
        x = series.lorenz_series()
        xs = np.ascontiguousarray(x[:: s["lyap_stride"]][: s["lyap_n"]])
        dt_eff = s["dt"] * s["lyap_stride"]

        def run() -> float:
            # Rosenstein, matching nolds.lyap_r / nolitsa.mle for an
            # apples-to-apples algorithm comparison on the identical series.
            r = ts.lyapunov_from_data(
                xs,
                dt=dt_eff,
                dimension=s["embed_dim"],
                delay=s["lyap_delay"],
                theiler=s["lyap_theiler"],
                k_max=20,
                method="rosenstein",
                fit=(1, 8),
            )
            return float(r.lyapunov)

        return run

    # -- dimension ---------------------------------------------------------- #

    def task_correlation_dimension(self, quick: bool) -> Callable[[], float]:
        import tsdynamics as ts

        s = self.cfg["series"]
        x = series.lorenz_series()[: s["corr_n"]]
        emb = series.delay_embed(x, dim=s["embed_dim"], delay=s["embed_delay"])
        # Radii spanning the scaling region (a small fraction of the attractor
        # extent up to ~its size); nolitsa is likewise given an explicit range,
        # so the scaling-region question is identical across libraries.
        sd = float(np.std(emb))
        radii = np.logspace(np.log10(sd * 0.05), np.log10(sd * 1.5), 30)

        def run() -> float:
            return float(ts.correlation_dimension(emb, theiler=s["theiler"], radii=radii))

        return run

    # -- bifurcation -------------------------------------------------------- #

    def task_bifurcation_diagram(self, quick: bool) -> Callable[[], Any]:
        import tsdynamics as ts

        lg = self.cfg["logistic"]
        n_rates = 200 if quick else lg["n_rates"]
        values = np.linspace(lg["r_min"], lg["r_max"], n_rates)
        log = ts.systems.Logistic()

        def run() -> None:
            ts.orbit_diagram(
                log,
                "r",
                values,
                n=lg["n_gens"],
                transient=lg["n_discard"],
                ic=[lg["ic"]],
            )

        return run

    # -- basins ------------------------------------------------------------- #

    def task_basins_of_attraction(self, quick: bool) -> Callable[[], Any]:
        import tsdynamics as ts

        nw = self.cfg["newton"]
        res = 80 if quick else nw["grid_res"]
        NewtonZ3 = _make_newton_map()
        grid = ts.data.Grid(
            lo=np.array([nw["grid_min"], nw["grid_min"]]),
            hi=np.array([nw["grid_max"], nw["grid_max"]]),
            counts=(res, res),
        )

        def run() -> None:
            ts.basins_of_attraction(
                NewtonZ3(), region=grid, dt=1.0, max_steps=nw["max_steps"], seed=0
            )

        return run

    # -- fixed points ------------------------------------------------------- #

    def task_fixed_points(self, quick: bool) -> Callable[[], float]:
        import tsdynamics as ts

        h = ts.systems.Henon().with_params(**self.cfg["henon"]["params"])

        def run() -> float:
            fps = ts.fixed_points(h, seed=0)
            return float(max(float(np.asarray(fp.x)[0]) for fp in fps))

        return run

    # -- Poincaré ----------------------------------------------------------- #

    def task_poincare_section(self, quick: bool) -> Callable[[], Any]:
        import tsdynamics as ts

        ros = ts.systems.Rossler().with_params(**self.cfg["rossler"]["params"])
        n = 200 if quick else 1000

        def run() -> None:
            ts.poincare_section(ros, plane=("y", 0.0, "up"), n=n, dt=0.01, seed=0)

        return run

    # -- from-data complexity / scaling / recurrence ------------------------ #

    def task_sample_entropy(self, quick: bool) -> Callable[[], float]:
        import tsdynamics as ts

        s = self.cfg["series"]
        x = np.ascontiguousarray(series.lorenz_series()[: s["entropy_n"]])

        def run() -> float:
            return float(ts.sample_entropy(x, dimension=s["entropy_m"], delay=1).value)

        return run

    def task_permutation_entropy(self, quick: bool) -> Callable[[], float]:
        import tsdynamics as ts

        s = self.cfg["series"]
        x = np.ascontiguousarray(series.lorenz_series()[: s["entropy_n"]])

        def run() -> float:
            return float(
                ts.permutation_entropy(
                    x, dimension=s["entropy_m"] + 1, delay=1, normalize=True
                ).value
            )

        return run

    def task_multiscale_entropy(self, quick: bool) -> Callable[[], float]:
        import tsdynamics as ts

        s = self.cfg["series"]
        x = np.ascontiguousarray(series.lorenz_series()[: s["entropy_n"]])

        def run() -> float:
            return float(np.mean(np.asarray(ts.multiscale_entropy(x, scales=5)[:])))

        return run

    def task_rqa_determinism(self, quick: bool) -> Callable[[], float]:
        import tsdynamics as ts

        s = self.cfg["series"]
        x = series.lorenz_series()[: s["rqa_n"]]
        # ts.rqa takes no embedding args — embed externally to the SAME (dim, τ) the
        # pyunicorn/neurokit RQA use, so all three see the same reconstructed set.
        emb = series.delay_embed(x, dim=s["rqa_embed_dim"], delay=s["rqa_embed_delay"])

        def run() -> float:
            return float(ts.rqa(emb, recurrence_rate=s["rqa_recurrence_rate"]).determinism)

        return run

    def task_embedding_dimension(self, quick: bool) -> Callable[[], float]:
        from tsdynamics.analysis.embedding import embedding_dimension

        s = self.cfg["series"]
        x = np.ascontiguousarray(series.lorenz_series()[: s["entropy_n"]])

        def run() -> float:
            return float(
                embedding_dimension(
                    x, method="cao", delay=s["embed_target_delay"], max_dim=s["embed_max_dim"]
                ).dimension
            )

        return run

    def task_surrogate_generation(self, quick: bool) -> Callable[[], None]:
        import tsdynamics as ts

        x = np.ascontiguousarray(series.lorenz_series()[: self.cfg["series"]["entropy_n"]])

        def run() -> None:  # speed-only: generate (timed), no estimate
            ts.iaaft_surrogate(x, n=1, seed=42)

        return run
