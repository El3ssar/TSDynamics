from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from numba import njit


class BaseDyn:
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

        timesteps = np.arange(0, final_t + dt, dt) # the final_t + dt is to include final time
        return timesteps


class DynSys(BaseDyn):
    """Class for continuous dynamical systems."""

    def rhs(self, X, t):
        # the * operator unpacks the tuple X into the arguments of self._rhs before t and **self.params
        return self._rhs(X=X, t=t, **self.params)

    def _rhs(self, X, t, **params):
        """Right-hand side function to be implemented by subclasses."""
        raise NotImplementedError

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
            **kwargs
        )

        return sol.y.T


    def lyapunov_spectrum(
        self,
        dt=0.01,
        final_time=1000.0,
        initial_conds=None,
        num_exponents=None,
        perturbation_scale=1e-8,
        reorthonormalize_interval=1,
    ):
        # Ensure initial conditions are set
        if initial_conds is None:
            if self.n_dim is None:
                raise ValueError("Initial conditions must be provided or n_dim must be set")
            initial_conds = np.random.rand(self.n_dim)
        else:
            initial_conds = np.asarray(initial_conds)
        self.initial_conds = initial_conds

        if num_exponents is None:
            num_exponents = self.n_dim

        n_dim = self.n_dim

        # Initialize augmented state vector
        total_dim = n_dim + n_dim * num_exponents
        y0_aug = np.zeros(total_dim)
        y0_aug[:n_dim] = initial_conds

        # Initialize perturbation vectors (scaled identity matrix)
        perturbations = np.identity(n_dim) * perturbation_scale
        y0_aug[n_dim:] = perturbations[:num_exponents].flatten()

        # Time settings
        t_span = (0.0, final_time)
        t_eval = np.arange(0.0, final_time, dt)

        # Lyapunov sums
        lyapunov_sums = np.zeros(num_exponents)
        total_intervals = 0

        def augmented_rhs(t, y_aug):
            n_dim = self.n_dim

            # Original state variables
            X = y_aug[:n_dim]

            # Compute the RHS of the original system
            dXdt = np.asarray(self._rhs(X, t, **self.params))

            # Compute the Jacobian at the current state
            J = np.asarray(self._jac(X, t, **self.params))

            # Perturbation vectors
            Q = y_aug[n_dim:].reshape(num_exponents, n_dim)

            # Variational equations
            dQdt = np.dot(Q, J.T)

            # Flatten dQdt
            dQdt_flat = dQdt.flatten()

            # Concatenate the derivatives
            dydt_aug = np.concatenate((dXdt, dQdt_flat))

            return dydt_aug

        # Integration loop with event handling for reorthonormalization
        def reorthonormalize(t, y_aug):
            nonlocal lyapunov_sums, total_intervals

            n_dim = self.n_dim

            # Extract perturbation vectors
            Q = y_aug[n_dim:].reshape(num_exponents, n_dim)

            # QR decomposition
            Q, R = np.linalg.qr(Q.T)
            Q = Q.T  # Transpose back to original shape

            # Update the augmented state vector
            y_aug[n_dim:] = Q.flatten()

            # Accumulate the logarithms of the diagonal elements of R
            lyapunov_sums += np.log(np.abs(np.diag(R)))
            total_intervals += 1

            return y_aug

        # Integrate using solve_ivp with a custom step function
        y_aug = y0_aug.copy()
        t_current = t_span[0]

        while t_current < t_span[1]:
            t_next = t_current + dt * reorthonormalize_interval

            if t_next > t_span[1]:
                t_next = t_span[1]

            sol = solve_ivp(
                augmented_rhs,
                (t_current, t_next),
                y_aug,
                method='RK45',
                t_eval=None,
                vectorized=False,
                rtol=1e-9,
                atol=1e-9,
            )

            if not sol.success:
                raise RuntimeError("Integration failed.")

            # Update current time and state
            t_current = sol.t[-1]
            y_aug = sol.y[:, -1]

            # Reorthonormalization
            y_aug = reorthonormalize(t_current, y_aug)

        # Compute Lyapunov exponents
        total_time = total_intervals * dt * reorthonormalize_interval
        exponents = lyapunov_sums / total_time

        return exponents


    def _augmented_rhs(self, t, y_aug, num_exponents):
        """
        Computes the derivatives for the augmented state vector.

        Args:
            t (float): Current time.
            y_aug (array): Augmented state vector.
            num_exponents (int): Number of Lyapunov exponents.

        Returns:
            dydt_aug (array): Derivatives of the augmented state vector.
        """
        n_dim = self.n_dim

        # Original state variables
        X = y_aug[:n_dim]

        # Compute the RHS of the original system
        dXdt = np.asarray(self._rhs(X, t, **self.params))

        # Compute the Jacobian at the current state
        J = np.asarray(self._jac(X, t, **self.params))

        # Perturbation vectors
        Q = y_aug[n_dim:].reshape(num_exponents, n_dim)

        # Variational equations
        dQdt = np.dot(J, Q.T).T

        # Flatten dQdt
        dQdt_flat = dQdt.flatten()

        # Concatenate the derivatives
        dydt_aug = np.concatenate((dXdt, dQdt_flat))

        return dydt_aug


class DynMap(BaseDyn):
    """Class for discrete maps."""

    def rhs(self, X):
        X = np.atleast_1d(X)
        return self._rhs(X=X, **self.params)

    @abstractmethod
    def _rhs(self, X, **params):
        """Right-hand side function to be implemented by subclasses."""
        pass

    def jac(self, X):
        X = np.atleast_1d(X)
        return self._jac(X=X, **self.params)

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

    def lyapunov_spectrum(
        self,
        y0=None,
        steps=1000,
        num_exponents=None,
        perturbation_scale=1e-8,
        reorthonormalize_interval=1,
    ):
        """
        Compute the Lyapunov exponents of the map.

        Args:
            y0 (array): Initial condition for the state variables.
            steps (int): Number of iterations.
            num_exponents (int): Number of Lyapunov exponents to compute.
            perturbation_scale (float): Initial scale of perturbation vectors.
            reorthonormalize_interval (int): Steps between reorthonormalizations.
            max_retries (int): Maximum number of retries with new initial conditions in case of divergence.

        Returns:
            exponents (array): Array of Lyapunov exponents.
        """
        if y0 is None:
            if self.n_dim is None:
                raise ValueError("Initial conditions must be provided, else n_dim must be set")
            y0 = np.random.rand(self.n_dim) #if self.n_dim > 1 else np.random.rand()
        else:
            y0 = np.asarray(y0)

        if num_exponents is None:
            num_exponents = self.n_dim

        n_dim = self.n_dim

        # Initialize the state and perturbation vectors
        state = y0.copy()
        perturbations = np.eye(n_dim)[:num_exponents] * perturbation_scale

        # Accumulate logarithms of stretching factors
        lyapunov_sums = np.zeros(num_exponents)
        total_intervals = 0

        states = self.iterate(state, steps) # Compute the trajectory with integrate to avoid the nans or infs

        for step in range(steps):
            # Map the state
            state = states[step]

            # Compute the Jacobian at the current state
            J = np.array(self.jac(state))

            # Update perturbations
            perturbations = np.dot(J, perturbations.T).T

            # Reorthonormalization
            if (step + 1) % reorthonormalize_interval == 0:
                Q, R = np.linalg.qr(perturbations.T)
                perturbations = Q.T

                # Accumulate the logarithms of the absolute values of the diagonal elements of R
                lyapunov_sums += np.log(np.abs(np.diag(R)))
                total_intervals += 1

        # Compute the Lyapunov exponents
        exponents = lyapunov_sums / (total_intervals * reorthonormalize_interval)

        return exponents


class DynSysDelay(BaseDyn):
    """Class for dynamical systems with time delay."""

    def __init__(self, delays=None, **kwargs):
        super().__init__(**kwargs)
        self.delays = np.array(delays if delays is not None else getattr(self, "delays", []))
        self.max_delay = np.max(self.delays) if self.delays.size > 0 else 0.0

    @abstractmethod
    def _rhs(self, X_current, X_delayed, t, params):
        """Right-hand side function to be implemented by subclasses."""
        pass

    def integrate(
        self,
        dt=0.02,
        steps=None,
        final_time=100,
        initial_conds=None,  # Should be an array
    ):
        """
        Integrate the delay differential equation using the 4th-order Runge-Kutta method.
        """
        t_eval = self.generate_timesteps(dt=dt, steps=steps, final_t=final_time)

        if initial_conds is None:
            if self.n_dim is None:
                raise ValueError(
                    "Initial conditions must be provided, or n_dim must be set"
                )
            else:
                y0 = np.random.rand(self.n_dim)
        elif isinstance(initial_conds, np.ndarray):
            y0 = initial_conds
        else:
            raise ValueError("initial_conds must be an array")

        n_steps = len(t_eval)
        delay_steps = (self.delays / dt).astype(np.int32)
        max_delay_steps = np.max(delay_steps)

        # Initialize solution array with initial conditions
        sol = np.zeros((n_steps + max_delay_steps, self.n_dim))
        sol[: max_delay_steps + 1] = y0  # Fill initial history

        # Precompute parameters for Numba
        params = self.get_params_array()

        # JIT-compile the integration loop
        sol = self.numba_integration_loop(
            sol, n_steps, delay_steps, dt, params, max_delay_steps, t_eval
        )

        # Return only the computed part of the solution
        return sol[max_delay_steps:]

    def get_params_array(self):
        """Convert parameters to a NumPy array for Numba compatibility."""
        return np.array(list(self.params.values()), dtype=np.float64)

    def numba_integration_loop(
        self, sol, n_steps, delay_steps, dt, params, max_delay_steps, t_eval
    ):
        """Numba-accelerated integration loop using RK4."""
        rhs_func = self._rhs  # The RHS function should be Numba-compatible

        @njit
        def integration_loop(sol, n_steps, delay_steps, dt, params, max_delay_steps):
            for i in range(max_delay_steps, n_steps + max_delay_steps - 1):
                X_current = sol[i]
                X_delayed = np.zeros_like(X_current)
                for j in range(len(delay_steps)):
                    delay_idx = i - delay_steps[j]
                    X_delayed += sol[delay_idx]  # Adjust as needed for multiple delays
                t = dt * (i - max_delay_steps)

                # Runge-Kutta 4th order method
                k1 = rhs_func(X_current, X_delayed, t, params)
                k2 = rhs_func(X_current + 0.5 * dt * k1, X_delayed, t + 0.5 * dt, params)
                k3 = rhs_func(X_current + 0.5 * dt * k2, X_delayed, t + 0.5 * dt, params)
                k4 = rhs_func(X_current + dt * k3, X_delayed, t + dt, params)

                xdot = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                sol[i + 1] = X_current + dt * xdot
            return sol

        sol = integration_loop(sol, n_steps, delay_steps, dt, params, max_delay_steps)
        return sol

__all__ = ["BaseDyn", "DynSys", "DynMap", "DynSysDelay"]