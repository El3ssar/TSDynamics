from abc import abstractmethod

import numpy as np
from scipy.integrate import solve_ivp

from .base import BaseDyn


class DynSys(BaseDyn):
    """Class for continuous dynamical systems."""

    def rhs(self, X, t):
        """
        Right-hand side function wrapper to pass the parameters to the right-hand side function.

        Args:
            X (array): State vector.
            t (float): Time.

        Returns:
            array: Right-hand side of the dynamical system.
        """
        return self._rhs(X=X, t=t, **self.params)

    def jac(self, X, t):
        """
        Jacobian function wrapper to pass the parameters to the Jacobian function.

        Args:
            X (array): State vector.
            t (float): Time.

        Returns:
            array: Jacobian of the dynamical system.
        """
        return self._jac(X=X, t=t, **self.params)

    @abstractmethod
    def _rhs(self, X, t, **params) -> None:
        """Right-hand side function to be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _jac(self, X, t, **params) -> None:
        """Jacobian function to be implemented by subclasses."""
        raise NotImplementedError

    def integrate(
        self,
        dt=0.02,
        steps=None,
        final_time=100,
        initial_conds=None,
        method="RK45",
        rtol=1e-3,
        atol=1e-3,
        **kwargs
    ):
        """Integrate the ODE system using scipy's solve_ivp."""

        if self.initial_conds is not None:
            initial_conds = self.initial_conds
        elif initial_conds is None:
            if self.n_dim is None:
                raise ValueError("Initial conditions must be provided, else n_dim must be set")
            initial_conds = np.random.rand(self.n_dim)
            self.initial_conds = initial_conds


        t_eval = self.generate_timesteps(dt=dt, steps=steps, final_time=final_time)
        t_span = (t_eval[0], t_eval[-1])

        # Integrate the system
        sol = solve_ivp(
            fun=lambda t, y: self.rhs(X=y, t=t),
            t_span=t_span,
            y0=initial_conds,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            **kwargs
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        self.initial_conds = np.array(initial_conds, copy=True)
        return sol.t, sol.y.T

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

