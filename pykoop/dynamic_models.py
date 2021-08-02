"""Example dynamic models."""

import abc
from typing import Callable

import numpy as np
from scipy import integrate


class ContinuousDynamicModel(metaclass=abc.ABCMeta):
    """Continueous-time dynamic model."""

    @abc.abstractmethod
    def f(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Implement differential equation.

        Parameters
        ----------
        t : float
            Time (s).
        x : np.ndarray
            State.
        u : np.ndarray
            Input.

        Returns
        -------
        np.ndarray
            Time derivative of state.
        """
        raise NotImplementedError()

    def g(self, t: float, x: np.ndarray) -> np.ndarray:
        """Implement output equation.

        Parameters
        ----------
        t : float
            Time (s).
        x : np.ndarray
            State.

        Returns
        -------
        np.ndarray
            Measurement of state.
        """
        return x

    def x0(self, x: np.ndarray) -> np.ndarray:
        """Generate initial conditions consistent with system constraints.

        For example, if the state of a pendulum is ``[x, y]``, and its length
        is ``l``, then ``x^2 + y^2 = l^2`` must hold. Using :func:`x0`, the
        initial condition may be specified as::

            initial_state = pendulum.x0(initial_angle)

        where ``initial_state`` will be calculated using the initial angle
        specified.

        Parameters
        ----------
        x : np.ndarray
            Reduced initial state.

        Returns
        -------
        np.ndarray
            Full initial state.
        """
        return x

    def simulate(
        self,
        t_range: tuple[float, float],
        t_step: float,
        x0: np.ndarray,
        u: Callable[[float, np.ndarray], np.ndarray],
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the model using numerical integration.

        Parameters
        ----------
        t_range : tuple[float, float]
            Start and stop times in a tuple.
        t_step : float
            Timestep of output data.
        x0 : np.ndarray
            Initial condition, shape (n, ).
        u : Callable[[float, np.ndarray], np.ndarray]
            Input function. Function of ``t`` and ``x``. Returns an input.
        **kwargs : dict
            Keyword arguments for :func:`integrate.solve_ivp`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Time and state at every timestep. Each timestep is one row.
        """
        sol = integrate.solve_ivp(
            lambda t, x: self.f(t, x, u(t, x)),
            t_range,
            x0,
            t_eval=np.arange(*t_range, t_step),
            **kwargs,
        )
        return (sol.t, sol.y.T)


class DiscreteDynamicModel(metaclass=abc.ABCMeta):
    """Discrete-time dynamic model."""

    @abc.abstractmethod
    def f(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Implement next-state equation.

        Parameters
        ----------
        t : float
            Time (s).
        x : np.ndarray
            State.
        u : np.ndarray
            Input.

        Returns
        -------
        np.ndarray
            Next state.
        """
        raise NotImplementedError()

    def g(self, t, x):
        """Implement output equation.

        Parameters
        ----------
        t : float
            Time (s).
        x : np.ndarray
            State.

        Returns
        -------
        np.ndarray
            Measurement of state.
        """
        return x

    def x0(self, x):
        """Generate initial conditions consistent with system constraints.

        For example, if the state of a pendulum is ``[x, y]``, and its length
        is ``l``, then ``x^2 + y^2 = l^2`` must hold. Using :func:`x0`, the
        initial condition may be specified as::

            initial_state = pendulum.x0(initial_angle)

        where ``initial_state`` will be calculated using the initial angle
        specified.

        Parameters
        ----------
        x : np.ndarray
            Reduced initial state.

        Returns
        -------
        np.ndarray
            Full initial state.
        """
        return x


class MassSpringDamper(ContinuousDynamicModel):
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
    def A(self):
        """Compute ``A`` matrix."""
        A = np.array([
            [0, 1],
            [-self.stiffness / self.mass, -self.damping / self.mass],
        ])
        return A

    @property
    def B(self):
        """Compute ``B`` matrix."""
        B = np.array([
            [0],
            [1 / self.mass],
        ])
        return B

    def f(self, t: float, x: np.ndarray, u: np.ndarray):
        # noqa: D102
        x_dot = (self.A @ np.reshape(x, (-1, 1))
                 + self.B @ np.reshape(u, (-1, 1)))
        return np.ravel(x_dot)
