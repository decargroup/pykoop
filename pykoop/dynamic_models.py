"""Example dynamic models."""

import abc
from typing import Callable, Tuple

import numpy as np
from scipy import constants, integrate


class ContinuousDynamicModel(metaclass=abc.ABCMeta):
    """Continuous-time dynamic model."""

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

    def simulate(
        self,
        t_range: Tuple[float, float],
        t_step: float,
        x0: np.ndarray,
        u: Callable[[float], np.ndarray],
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the model using numerical integration.

        Parameters
        ----------
        t_range : Tuple[float, float]
            Start and stop times in a tuple.
        t_step : float
            Timestep of output data.
        x0 : np.ndarray
            Initial condition, shape (n, ).
        u : Callable[[float], np.ndarray]
            Input function of time.
        **kwargs : dict
            Keyword arguments for :func:`integrate.solve_ivp`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Time and state at every timestep. Each timestep is one row.
        """
        sol = integrate.solve_ivp(
            lambda t, x: self.f(t, x, u(t)),
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

    def simulate(
        self,
        t_range: Tuple[float, float],
        t_step: float,
        x0: np.ndarray,
        u: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the model.

        Parameters
        ----------
        t_range : Tuple[float, float]
            Start and stop times in a tuple.
        t_step : float
            Timestep of output data.
        x0 : np.ndarray
            Initial condition, shape (n, ).
        u : np.ndarray
            Input array.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Time and state at every timestep. Each timestep is one row.
        """
        t = np.arange(*t_range, t_step)
        x = np.empty((t.shape[0], x0.shape[0]))
        x[0, :] = x0
        for k in range(1, t.shape[0]):
            x[k, :] = self.f(t[k - 1], x[k - 1, :], u[k - 1])
        return (t, x)


class MassSpringDamper(ContinuousDynamicModel):
    """Mass-spring-damper model.

    State is ``[position, velocity]``.

    Examples
    --------
    Simulate a mass-spring-damper

    >>> msd = pykoop.dynamic_models.MassSpringDamper(0.5, 0.7, 0.6)
    >>> x0 = np.array([1, 0])
    >>> t, x = msd.simulate((0, 1), 1e-3, x0, lambda t: 0)
    """

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


class Pendulum(ContinuousDynamicModel):
    """Point-mass pendulum with optional damping.

    State is ``[angle, angular_velocity]``.

    Examples
    --------
    Simulate a pendulum

    >>> pend = pykoop.dynamic_models.Pendulum(0.5, 1, 0.6)
    >>> x0 = np.array([np.pi / 2, 0])
    >>> t, x = pend.simulate((0, 1), 1e-3, x0, lambda t: 0)
    """

    def __init__(self, mass, length, damping=0):
        """Instantiate :class:`Pendulum`.

        Parameters
        ----------
        mass : float
            Mass (kg).
        length : float
            Length (m).
        damping : float
            Viscous damping (N.m.s/rad).
        """
        self.mass = mass
        self.length = length
        self.damping = damping

    def f(self, t, x, u):
        # noqa: D102
        theta, theta_dot = x
        x_dot = np.array([
            theta_dot,
            (-self.damping / self.mass / self.length**2 * theta_dot
             - constants.g / self.length * np.sin(theta)),
        ]) + np.array([
            0,
            1 / (self.mass * self.length**2),
        ]) * u
        return x_dot


class DuffingOscillator(ContinuousDynamicModel):
    r"""Duffing oscillator model.

    Equation is ``\ddot{x} + \delta \dot{x} + \beta x + \alpha x^3 = u(t)``
    where usually ``u(t) = a \cos(\omega t)``.
    """

    def __init__(
        self,
        alpha: float = 1,
        beta: float = -1,
        delta: float = 0.1,
    ) -> None:
        """Instantiate :class:`DuffingOscillator`.

        Parameters
        ----------
        alpha : float
            Coefficient of cubic term.
        beta : float
            Coefficient of linear term.
        delta : float
            Coefficient of first derivative.
        """
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def f(self, t: float, x: np.ndarray, u: np.ndarray):
        # noqa: D102
        x_dot = np.array([
            x[1],
            u - self.delta * x[1] - self.beta * x[0] - self.alpha * x[0]**3
        ])
        return x_dot


class DiscreteVanDerPol(DiscreteDynamicModel):
    """Van der Pol oscillator.

    Examples
    --------
    Simulate Van der Pol oscillator

    >>> t_step = 0.1
    >>> vdp = pykoop.dynamic_models.DiscreteVanDerPol(t_step, 2)
    >>> x0 = np.array([1, 0])
    >>> t_range = (0, 10)
    >>> u = 0.01 * np.cos(np.arange(*t_range, t_step))
    >>> t, x = vdp.simulate(t_range, t_step, x0, u)
    """

    def __init__(self, t_step: float, mu: float) -> None:
        """Instantiate :class:`DiscreteVanDerPol`.

        Parameters
        ----------
        t_step : float
            Timestep (s)
        mu : float
            Strength of nonlinearity.
        """
        self.t_step = t_step
        self.mu = mu

    def f(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # noqa: D102
        x_next = x + self.t_step * np.array(
            [x[1], self.mu * (1 - x[0]**2) * x[1] - x[0] + u])
        return x_next
