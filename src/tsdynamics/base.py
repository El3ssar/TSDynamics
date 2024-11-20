from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from ddeint import ddeint
from scipy.integrate import solve_ivp

class BaseDyn(ABC):
    """Abstract base class for all dynamical systems."""

    def __init__(
        self,
        n_dim=None,
        params=None
    ) -> None:
        self.n_dim = n_dim if n_dim is not None else getattr(self, "n_dim", None)
        self.params = params if params is not None else getattr(self, "params", None)
        self.initial_conds = None

        # Make the parameters available as attributes
        if self.params:
            for key, value in self.params.items():
                setattr(self, key, value)


    def __setattr__(
        self,
        name: str,
        value: Any
    ) -> None:
        """Set an attribute and add it to the parameters dictionary."""
        if "params" in self.__dict__ and name in self.__dict__.get("params", {}):
            self.__dict__["params"][name] = value
        super().__setattr__(name, value)

    @abstractmethod
    def rhs(
        self,
        X,
        t
    ):
        """Right-hand side of the dynamical system."""
        pass

    def generate_timesteps(
        self,
        dt=0.02,
        steps=None,
        final_t=1.0
    ):
        """
        Generate a set of timesteps for a given time series

        Args:
            dt (float): the time step
            steps (int): the number of steps
            final_t (float): the final time

        Returns:
            timesteps (ndarray): the time steps

        Notes:
            final_t takes precedence over steps, being the physical time more relevant. So if both are given, the final_t is used.
        """
        if final_t is None:
            if steps is None:
                raise ValueError("Either steps or final_t must be provided")
            final_t = steps * dt
        else:
            if steps is not None:
                print("Both steps and final_t are given. final_t will be used.")

        timesteps = np.arange(0, final_t + dt, dt)
        return timesteps

class DynSys(BaseDyn):
    """Class for continuous dynamical systems."""

    def rhs(self, X, t):
        # the * operator unpacks the tuple X into the arguments of self._rhs before t and **self.params
        return self._rhs(X=X, t=t, **self.params)

    @abstractmethod
    def _rhs(self, X, t, **params):
        """Right-hand side function to be implemented by subclasses."""
        pass

    def integrate(
        self,
        dt=0.02,
        steps=None,
        final_time=100,
        initial_conds=None,
        **kwargs
    ):
        """Integrate the ODE system using scipy's solve_ivp."""

        if initial_conds is None:
            if self.n_dim is None:
                raise ValueError("Initial conditions must be provided, else n_dim must be set")
            initial_conds = np.random.rand(self.n_dim)
            self.initial_conds = initial_conds


        t_eval = self.generate_timesteps(dt=dt, steps=steps, final_t=final_time)
        t_span = (t_eval[0], t_eval[-1])

        # Integrate the system
        sol = solve_ivp(
            fun=lambda t, y: self.rhs(X=y, t=t),
            t_span=t_span,
            y0=initial_conds,
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-6,
            **kwargs
        )

        return sol.y.T

class DynMap(BaseDyn):
    """Class for discrete maps."""

    def rhs(self, y):
        y = np.atleast_1d(y)
        return self._rhs(*y, **self.params)

    @abstractmethod
    def _rhs(self, y, **params):
        """Right-hand side function to be implemented by subclasses."""
        pass

    def iterate(
        self,
        y0=None,
        steps=1000,
        max_retries=10
    ):
        """Iterate the map for n_steps starting from y0."""
        retries = 0

        while retries < max_retries:
            if y0 is None:
                if self.n_dim is None:
                    raise ValueError("Initial conditions must be provided, else n_dim must be set")
                y0 = np.random.rand(self.n_dim) if self.n_dim > 1 else np.random.rand()

            y = np.array(y0)
            trajectory = np.empty((steps, self.n_dim))
            try:
                for i in range(steps):
                    y = self.rhs(y)
                    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                        raise ValueError(f"The trajectory diverged at step {i}: y = {y}")
                    trajectory[i] = y
                return trajectory
            except ValueError as e:
                print(f"Warning: {e}. Retrying with a new random initial condition.")
                y0 = None
                retries += 1

        raise ValueError(f"Failed to iterate the map after {max_retries} retries")

class DynSysDelay(BaseDyn):
    """Class for dynamical systems with time delay."""

    def __init__(
        self,
        delays=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.delays = delays if delays is not None else getattr(self, "delays", None)

    @abstractmethod
    def _rhs(self, X_current, X_delayed, t, **kwargs):
        """Right-hand side function to be implemented by subclasses."""
        # This is where the banana goes, X_current and X_delayed might be arrays or scalars, depending on the system.
        # You might want to unpack them.
        pass

    def rhs(self, X, t):
        """Wrapper to match the signature expected by ddeint."""
        # Current state
        X_current = X(t)

        # Delayed states
        X_delayed = [X(t - delay) for delay in self.delays]

        ans = self._rhs(X_current, X_delayed, t, **self.params)

        return ans

    def integrate(self,
                  dt=0.02,
                  steps=None,
                  final_time=100,
                  initial_conds=None, # Can be a callable function or an array
                ):
        """Integrate the delay differential equation."""
        t_eval = self.generate_timesteps(dt=dt, steps=steps, final_t=final_time)

        if initial_conds is None:
            if self.n_dim is None:
                raise ValueError("Initial history function 'history_f' must be provided, else n_dim must be set")
            else:
                # Define a constant history function with random values. Needs to be a single scalar if n_dim=1
                y0_values = np.random.rand(self.n_dim) if self.n_dim > 1 else np.random.rand()
                def history_f(t):
                    return y0_values
        elif isinstance(initial_conds, np.ndarray):
            def history_f(t):
                return initial_conds
        # Raise error if history_f is not callable
        elif not callable(initial_conds):
            raise ValueError("history_f must be a callable function or an array")


        # # Use ddeint to integrate
        sol = ddeint(self.rhs, history_f, t_eval)

        return sol


__all__ = ["BaseDyn", "DynSys", "DynMap", "DynSysDelay"]