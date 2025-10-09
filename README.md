
# tsdynamics

Adaptive, compiled integration for **ODEs** (JiTCODE) and **DDEs** (JiTCDDE) with a small, clean API. You write the math; we handle compilation, tolerances, trajectories, and Lyapunov spectra.

* ODE base: `DynSys` → **JiTCODE** (`jitcode`)
* DDE base: `DynSysDelay` → **JiTCDDE** (`jitcdde`)
* Lyapunov spectra: `jitcode_lyap` / `jitcdde_lyap`
* Parameters are simple dicts exposed as attributes
* Deterministic output grids via `generate_timesteps`


## Contents

* [Install](#install)
* [Quickstart](#quickstart)
* [Define your own system](#define-your-own-system)
* [Lyapunov spectra](#lyapunov-spectra)
* [Notes & tips](#notes--tips)
* [Contributing](#contributing)
* [License](#license)

---

## Install

You’ll need a C/C++ toolchain for JiTCODE/JiTCDDE.

* **Linux:** `sudo apt-get install build-essential python3-dev` (Debian/Ubuntu)
* **macOS:** `xcode-select --install`
* **Windows:** Install “Microsoft C++ Build Tools” (VS Build Tools)

### Option A — uv (recommended)

```bash
# create & activate venv
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# library only
uv pip install .

# or editable + dev tools (tests, lint, etc.)
uv pip install -e ".[dev]"
# docs extras:
uv pip install -e ".[docs]"
```

### Option B — pip (plain)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install .
# or
pip install -e ".[dev]" ".[docs]"
```

### Option C — conda (env + pip for the package)

```bash
conda create -n tsdynamics python=3.12 -y
conda activate tsdynamics

pip install .
# or
pip install -e ".[dev]" ".[docs]"
```

> Dependencies (from `pyproject.toml`):
> `numpy>=2,<3`, 
> `scipy>=1.14,<2`, 
> `matplotlib>=3.10.6`,
> `numba==0.62.1`, 
> `jitcdde==1.8.3`, 
> `jitcode==1.7.3`,
> `symengine==0.14.1`, `sympy==1.14.0`.

---

## Quickstart

### ODE — Lorenz

```python
import numpy as np
from tsdynamics.base.ode_base import DynSys

class Lorenz(DynSys):
    params = {"sigma": 10.0, "rho": 28.0, "beta": 8.0/3.0}
    n_dim = 3

    @staticmethod
    def _rhs(y, t, beta, rho, sigma):
        x, yv, z = y(0), y(1), y(2)
        return (sigma*(yv - x), rho*x - x*z - yv, x*yv - beta*z)

lor = Lorenz()
t, X = lor.integrate(dt=0.01, final_time=50.0, method="dop853", rtol=1e-8, atol=1e-10)
exps = lor.lyapunov_spectrum(dt=0.1, burn_in=50.0, final_time=300.0,
                             method="dop853", rtol=1e-8, atol=1e-10)
print(exps)  # ~[0.91, ~0, -14.57]
```

### DDE — Mackey–Glass

```python
import numpy as np
from tsdynamics.base.dde_base import DynSysDelay

class MackeyGlass(DynSysDelay):
    params = {"beta": 0.2, "gamma": 0.1, "tau": 17.0, "n": 10}
    n_dim = 1

    @staticmethod
    def _rhs(y, t, beta, gamma, tau, n):
        y_tau = y(0, t - tau)
        y_now = y(0, t)
        return [beta * y_tau / (1 + y_tau**n) - gamma * y_now]

mg = MackeyGlass()
history = lambda s: [1.0 + 0.1*np.sin(0.2*s)]  # avoid trivial equilibrium history

t, y = mg.integrate(dt=0.05, final_time=200.0, history=history, rtol=1e-8, atol=1e-10)
exps = mg.lyapunov_spectrum(n_lyap=1, dt=0.2, burn_in=100.0, final_time=600.0,
                            history=history, rtol=1e-8, atol=1e-10)
print(exps)  # small positive (~1e-3)
```

### ODE — Lorenz–96 (periodic)

```python
class Lorenz96(DynSys):
    params = {"f": 8.0, "N": 20}
    n_dim = 20

    @staticmethod
    def _rhs(y, t, f, N):
        return [ (y((i+1)%N) - y((i-2)%N)) * y((i-1)%N) - y(i) + f
                 for i in range(N) ]

l96 = Lorenz96()
t, X = l96.integrate(dt=0.05, final_time=50.0, method="dop853", rtol=1e-8, atol=1e-10)
```

### ODE — Kuramoto–Sivashinsky (FD, periodic)

```python
class KuramotoSivashinsky(DynSys):
    def __init__(self, N, L, initial_conds=None):
        if N < 5: raise ValueError("N >= 5 required")
        super().__init__(n_dim=N, params={"N": int(N), "L": float(L)}, initial_conds=initial_conds)

    @staticmethod
    def _rhs(y, t, N, L):
        dx = L / N
        inv_dx, inv_dx2, inv_dx4 = 1.0/dx, 1.0/dx**2, 1.0/dx**4
        rhs = []
        for j in range(N):
            jm2, jm1, jp1, jp2 = (j-2)%N, (j-1)%N, (j+1)%N, (j+2)%N
            u = y(j)
            nonlinear = - (y(jp1)**2 - y(jm1)**2) * (0.25*inv_dx)   # -0.5*(u^2)_x
            uxx      = (y(jp1) - 2*u + y(jm1)) * inv_dx2
            uxxxx    = (y(jm2) - 4*y(jm1) + 6*u - 4*y(jp1) + y(jp2)) * inv_dx4
            rhs.append(nonlinear - uxx - uxxxx)
        return rhs

ks = KuramotoSivashinsky(N=64, L=32.0, initial_conds=1e-2*np.random.randn(64))
t, U = ks.integrate(dt=0.1, final_time=300.0, method="dop853", rtol=1e-8, atol=1e-10)
```

---

## Define your own system

### ODE (`DynSys`)

* Set `params` (dict) and `n_dim` (int).
* Implement **static** `_rhs(y, t, **params)`; return `n_dim` expressions. Use `y(i)` to access state components.

```python
class MyODE(DynSys):
    params = {"a": 2.0, "b": 0.5}
    n_dim = 2
    @staticmethod
    def _rhs(y, t, a, b):
        x, z = y(0), y(1)
        return (a*x - b*z, x + z - x*z)
```

### DDE (`DynSysDelay`)

* Same pattern but delayed access is `y(i, t - tau)`.

```python
class MyDDE(DynSysDelay):
    params = {"tau": 1.5, "k": 2.0}
    n_dim = 1
    @staticmethod
    def _rhs(y, t, tau, k):
        x_tau = y(0, t - tau)
        x_now = y(0, t)
        return [k * x_tau - x_now]
```

### Map (`DynMap`)

* Same pattern but `iterate` is used instead of `integrate`.

```python
class MyMap(DynMap):
    params = {"a": 2.0, "b": 0.5}
    n_dim = 2
    @staticmethod
    def _rhs(y, a, b):
        x, z = y(0), y(1)
        return (a*x - b*z, x + z - x*z)
```
---

## Lyapunov spectra

```python
# ODE
exps = obj.lyapunov_spectrum(
    dt=0.1, burn_in=50.0, final_time=300.0,
    n_lyap=None, method="dop853", rtol=1e-8, atol=1e-10
)

# DDE
exps = obj.lyapunov_spectrum(
    n_lyap=2, dt=0.2, burn_in=100.0, final_time=600.0,
    history=history, rtol=1e-8, atol=1e-10
)
```

* ODE: time-weight the local LEs; constant `dt` ≈ simple mean.
* DDE: **use the weight returned by `jitcdde_lyap`** at each step.

---

## Contributing

We welcome sharp, clean contributions. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for setup (uv/pip/conda), style, tests, and PR workflow.

---

## License

MIT.
