from abc import abstractmethod

import numpy as np

from .base import BaseDyn


class DynMap(BaseDyn):
    """Class for discrete maps."""

    def rhs(self, X):
        X = np.atleast_1d(X)
        out = self._rhs(X=X, **self.params)
        return np.atleast_1d(out)

    @abstractmethod
    def _rhs(self, X, **params):
        """Right-hand side function to be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _jac(self, X, **params):
        """Jacobian function to be implemented by subclasses."""
        raise NotImplementedError

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

            y = np.atleast_1d(y0)
            trajectory = np.empty((steps, y.size))
            try:
                for i in range(steps):
                    y = self.rhs(y)
                    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                        raise ValueError(f"The trajectory diverged at step {i}: y = {y}")
                    trajectory[i] = np.atleast_1d(y)
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

