from typing import Any, Optional

import numpy as np


class BaseDyn:
    """Abstract base class for all dynamical systems."""

    def __init__(
        self,
        n_dim=None,
        params=None,
        initial_conds=None
    ) -> None:
        self.n_dim = n_dim if n_dim is not None else getattr(self, "n_dim", None)
        self.params = (
            dict(params) if params is not None
            else dict(getattr(self, "params", {}) or {})
        )
        if initial_conds is not None:
            self.initial_conds = initial_conds
        else:
            self.initial_conds = getattr(self, "initial_conds", None)

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
        params = self.__dict__.get("params")
        if isinstance(params, dict) and name in params:
            params[name] = value
        object.__setattr__(self, name, value)

    def generate_timesteps(
        self,
        dt: float = 0.02,
        steps: Optional[int] = None,
        final_time: Optional[float] = 1.0
    ) -> np.ndarray:
        """
        Generate a sequence of time steps for a given simulation or time series.

        Parameters
        ----------
        dt : float, optional
            The time step size between consecutive time points. Default is 0.02.
        steps : int, optional
            The number of steps (i.e., points) to generate. If provided, this
            takes precedence over ``final_time``.
        final_time : float, optional
            The final simulation time. Used only if ``steps`` is not provided.
            Default is 1.0.

        Returns
        -------
        timesteps : ndarray of float64
            A 1D NumPy array containing the generated time steps.

        Raises
        ------
        ValueError
            If neither ``steps`` nor ``final_time`` is provided.

        Notes
        -----
        ``steps`` takes precedence over ``final_time``. If both are given, the
        ``steps`` value is used, and a warning message is printed. The returned
        array always includes the final point (either ``steps * dt`` or
        ``final_time``).

        Examples
        --------
        >>> obj.generate_timesteps(dt=0.1, steps=5)
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5])

        >>> obj.generate_timesteps(dt=0.1, final_time=0.4)
        array([0. , 0.1, 0.2, 0.3, 0.4])
        """
        if steps is None:
            if final_time is None:
                raise ValueError("Either 'steps' or 'final_time' must be provided.")
            ts = np.arange(0.0, final_time, dt)
            if ts.size == 0 or ts[-1] < final_time - 1e-12:
                ts = np.append(ts, final_time)
        else:
            if final_time is not None:
                print("Both 'steps' and 'final_time' are given. Using 'steps' instead.")
            ts = np.arange(0.0, steps * dt, dt)
            ts = np.append(ts, steps * dt)
        return ts

