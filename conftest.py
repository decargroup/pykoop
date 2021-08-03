"""Pytest fixtures for doctests."""
import numpy as np
import pytest
import sklearn

import pykoop


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    """Add numpy to namespace."""
    doctest_namespace['np'] = np


@pytest.fixture(autouse=True)
def add_pykoop(doctest_namespace):
    """Add pykoop to namespace."""
    doctest_namespace['pykoop'] = pykoop


@pytest.fixture(autouse=True)
def add_sklearn(doctest_namespace):
    """Add sklearn to namespace."""
    doctest_namespace['sklearn'] = sklearn


@pytest.fixture(autouse=True)
def add_X_msd_no_input(doctest_namespace):
    """Add inputless mass-spring-damper data to namespace.

    Has an episode feature and no input.
    """
    # Specify duration
    t_range = (0, 10)
    t_step = 0.1
    # Create object
    msd = pykoop.dynamic_models.MassSpringDamper(
        mass=0.5,
        stiffness=0.7,
        damping=0.6,
    )
    # Specify initial conditions
    x0 = msd.x0(np.array([1, -1]))
    # Simulate ODE
    t, x = msd.simulate(
        t_range,
        t_step,
        x0,
        lambda t: 0,
        rtol=1e-8,
        atol=1e-8,
    )
    # Format the data
    X = np.hstack((
        np.zeros((t.shape[0], 1)),  # episode feature
        x,
    ))
    doctest_namespace['X_msd_no_input'] = X


@pytest.fixture(autouse=True)
def add_X_msd(doctest_namespace):
    """Add mass-spring-damper data to namespace.

    Has an episode feature and one input.
    """
    # Specify duration
    t_range = (0, 10)
    t_step = 0.1
    # Create object
    msd = pykoop.dynamic_models.MassSpringDamper(
        mass=0.5,
        stiffness=0.7,
        damping=0.6,
    )

    # Specify input
    def u(t):
        """Sinusoidal input."""
        return 0.1 * np.sin(t)

    # Specify initial conditions
    x0 = msd.x0(np.array([0, 0]))
    # Simulate ODE
    t, x = msd.simulate(t_range, t_step, x0, u, rtol=1e-8, atol=1e-8)
    # Format the data
    X = np.hstack((
        np.zeros((t.shape[0], 1)),  # episode feature
        x,
        np.reshape(u(t), (-1, 1)),
    ))
    doctest_namespace['X_msd'] = X


@pytest.fixture(autouse=True)
def add_X_pend(doctest_namespace):
    """Add pendulum data to namespace.

    Has an episode feature and one input.
    """
    # Specify duration
    t_range = (0, 10)
    t_step = 0.1
    # Create object
    pend = pykoop.dynamic_models.Pendulum(
        mass=0.5,
        length=1,
        damping=0.6,
    )

    # Specify input
    def u(t):
        """Sinusoidal input."""
        return 0.1 * np.sin(t)

    # Specify initial conditions
    x0 = pend.x0(np.array([0, 0]))
    # Simulate ODE
    t, x = pend.simulate(t_range, t_step, x0, u, rtol=1e-8, atol=1e-8)
    # Format the data
    X = np.hstack((
        np.zeros((t.shape[0], 1)),  # episode feature
        x,
        np.reshape(u(t), (-1, 1)),
    ))
    doctest_namespace['X_pend'] = X


@pytest.fixture(autouse=True)
def add_msd(doctest_namespace):
    """Add mass-spring-damper object to namespace."""
    # Create object
    msd = pykoop.dynamic_models.MassSpringDamper(
        mass=0.5,
        stiffness=0.7,
        damping=0.6,
    )
    doctest_namespace['msd'] = msd
