from abc import abstractmethod

import numpy as np

from .base import BaseDyn


class DynMap(BaseDyn):
    """Class for discrete maps."""

    def rhs(self, X):
        """Evaluate the map at state X, passing params positionally."""
        X = np.asarray(X, dtype=np.float64)
        params = tuple(float(v) for v in self.params.values())
        # Call with positional args only
        out = self._rhs(X, *params)
        return np.asarray(out, dtype=np.float64)

    def jac(self, X):
        """Evaluate the Jacobian at state X, passing params positionally."""
        X = np.asarray(X, dtype=np.float64)
        params = tuple(float(v) for v in self.params.values())
        out = self._jac(X, *params)
        return np.asarray(out, dtype=np.float64)

    @abstractmethod
    def _rhs(self, X, **params):
        """Right-hand side function to be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _jac(self, X, **params):
        """Jacobian function to be implemented by subclasses."""
        raise NotImplementedError

    def iterate(self, initial_conds=None, steps=1000, max_retries=10):
        """Iterate the map for n_steps starting from initial_conds."""
        # Resolve the starting IC once; on retry always try a new random IC.
        if initial_conds is not None:
            ic = np.asarray(initial_conds, float).reshape(self.n_dim)
            self.initial_conds = np.array(ic, copy=True)
        elif self.initial_conds is not None:
            ic = np.atleast_1d(np.asarray(self.initial_conds, float))
        else:
            ic = None  # will be generated randomly below

        for _attempt in range(max_retries):
            if ic is None:
                ic = np.random.rand(self.n_dim).reshape(self.n_dim).astype(float)
                self.initial_conds = np.array(ic, copy=True)

            y = np.atleast_1d(ic)
            trajectory = np.empty((steps, y.size))

            try:
                for i in range(steps):
                    y = self.rhs(y)
                    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                        raise ValueError(f"The trajectory diverged at step {i}: y = {y}")
                    trajectory[i] = np.atleast_1d(y)
                return np.arange(steps), trajectory

            except ValueError as e:
                print(f"Warning: {e}. Retrying with a new random initial condition.")
                ic = None  # next iteration generates a fresh random IC

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

        Returns
        -------
            exponents (array): Array of Lyapunov exponents.
        """
        if y0 is None:
            if self.initial_conds is not None:
                y0 = np.atleast_1d(np.asarray(self.initial_conds, float))
            elif self.n_dim is not None:
                y0 = np.random.rand(self.n_dim)
            else:
                raise ValueError("Initial conditions must be provided, else n_dim must be set")
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

        _, states = self.iterate(
            state, steps
        )  # Compute the trajectory with integrate to avoid the nans or infs

        for step in range(steps):
            # Map the state
            state = states[step]

            # Compute the Jacobian at the current state
            J = np.atleast_2d(np.array(self.jac(state), dtype=float))

            # Update perturbations
            perturbations = np.dot(J, perturbations.T).T

            # Reorthonormalization
            if (step + 1) % reorthonormalize_interval == 0:
                Q, R = np.linalg.qr(perturbations.T)
                perturbations = Q.T

                # Accumulate the logarithms of the absolute values of the diagonal elements of R
                # Guard against exact zeros on the diagonal (e.g. stable fixed-point trajectories)
                diag_abs = np.abs(np.diag(R))
                diag_abs = np.where(diag_abs == 0.0, np.finfo(float).tiny, diag_abs)
                lyapunov_sums += np.log(diag_abs)
                total_intervals += 1

        # Compute the Lyapunov exponents
        exponents = lyapunov_sums / (total_intervals * reorthonormalize_interval)

        return exponents
