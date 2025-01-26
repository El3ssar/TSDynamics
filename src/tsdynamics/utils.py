from typing import Callable

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
from diffrax import PIDController
from scipy.spatial.distance import pdist

try:
    from numba import njit
except ImportError:

    def njit(func):
        return func


def staticjit(func: Callable) -> Callable:
    """Decorator to apply numba's njit decorator to a static method"""
    return staticmethod(njit(func))


def calculate_typical_scale(data, sample_size=1000):
    """
    Calculates the typical scale of an attractor from time series data.

    Parameters:
    - data (np.ndarray): The time series data array of shape (T, D).
    - sample_size (int): Number of samples to use for estimation.

    Returns:
    - float: The typical scale of the attractor.
    """
    T, D = data.shape

    # Step 1: Sampling
    if T > sample_size:
        indices = np.random.choice(T, size=sample_size, replace=False)
        sampled_data = data[indices]
    else:
        sampled_data = data

    # Step 2: Compute Pairwise Distances
    pairwise_distances = pdist(sampled_data, metric="euclidean")

    # Step 3: Calculate the Average Distance
    typical_scale = np.mean(pairwise_distances)

    return typical_scale


def jax_solve_ivp(fun, t_eval, y0, method="Tsit5", rtol=1e-3, atol=1e-3):
    """
    Solves an initial value problem using JAX. This function is to mimic the functionality of the `solve_ivp` function in `scipy.integrate`.
    
    Args:
        - fun (callable): The right-hand side of the system of equations.
        - t_eval (np.ndarray): The time points to evaluate the solution.
        - y0 (np.ndarray): The initial condition.
        - method (str): The method to use for solving the ODE.
        - rtol (float): Relative tolerance.
        - atol (float): Absolute tolerance.
    
    """
    
    if method in dfx.__dict__:
        solver = dfx.__dict__[method]()
    y0 = jnp.array(y0)

    rhs = dfx.ODETerm(fun)

    sol = dfx.diffeqsolve(
        terms=rhs,
        solver=solver,
        t0=t_eval[0],
        t1=t_eval[-1],
        dt0=0.1,
        y0=y0,
        saveat=dfx.SaveAt(ts=t_eval),
        stepsize_controller=PIDController(rtol=rtol, atol=atol),
    )

    return sol.ts, sol.ys.T


__all__ = ["staticjit"]
