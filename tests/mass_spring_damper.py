"""Mass-spring-damper model for unit tests."""

import numpy as np


class MassSpringDamper():
    """Mass-spring-damper model."""

    def __init__(self, mass: float, stiffness: float, damping: float) -> None:
        """Instantiate :class:`MassSpringDamper`.

        Parameters
        ----------
        mass : float
            Mass (kg).
        stiffness : float
            Stiffness (N/m).
        damping : float
            Viscous damping (N.s/m).
        """
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping

    @property
    def _A(self):
        """Compute ``A`` matrix."""
        A = np.array([
            [0, 1],
            [-self.stiffness / self.mass, -self.damping / self.mass],
        ])
        return A

    @property
    def _B(self):
        """Compute ``B`` matrix."""
        B = np.array([
            [0],
            [1 / self.mass],
        ])
        return B

    def f(self, t: float, x: np.ndarray, u: np.ndarray):
        """Implement differential equation.

        Parameters
        ----------
        t : float
            Time (s).
        x : np.ndarray
            State [position (m); velocity (m/s)].
        u : np.ndarray
            Input (N).

        Returns
        -------
        np.ndarray
            Time derivative of state.
        """
        x_dot = (self._A @ np.reshape(x, (-1, 1))
                 + self._B @ np.reshape(u, (-1, 1)))
        return np.ravel(x_dot)
