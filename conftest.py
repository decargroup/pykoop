"""Pytest fixtures for doctests."""
from typing import Any, Dict

import numpy as np
import pytest
import sklearn
from scipy import linalg, integrate

import pykoop

# Add remote option for MOSEK


def pytest_addoption(parser) -> None:
    """Add ``--remote`` command line flag to run regressions remotely."""
    parser.addoption(
        '--remote',
        action='store_true',
        help='run regressions on MOSEK OptServer instead of locally',
    )


@pytest.fixture
def remote(request) -> bool:
    """Get ``--remote`` flag value."""
    return request.config.getoption('--remote')


@pytest.fixture
def remote_url() -> str:
    """Return URL for MOSEK OptServer service.

    See http://solve.mosek.com/web/index.html
    """
    return 'http://solve.mosek.com:30080'


# Add mass-spring-damper reference trajectory for all tests


@pytest.fixture
def mass_spring_damper_sine_input() -> Dict[str, Any]:
    """Compute mass-spring-damper response with sine input."""
    # Set up problem
    t_range = (0, 10)
    t_step = 0.1
    msd = pykoop.dynamic_models.MassSpringDamper(0.5, 0.7, 0.6)

    def u(t):
        return 0.1 * np.sin(t)

    # Initial condition
    x0 = np.array([0, 0])
    # Solve ODE for training data
    t, x = msd.simulate(
        t_range,
        t_step,
        x0,
        u,
        rtol=1e-8,
        atol=1e-8,
    )
    # Compute discrete-time A and B matrices
    Ad = linalg.expm(msd.A * t_step)

    def integrand(s):
        return linalg.expm(msd.A * (t_step - s)).ravel()

    Bd = integrate.quad_vec(integrand, 0, t_step)[0].reshape((2, 2)) @ msd.B
    U_valid = np.hstack((Ad, Bd)).T
    # Split the data
    y_train, y_valid = np.split(x, 2, axis=0)
    u_train, u_valid = np.split(np.reshape(u(t), (-1, 1)), 2, axis=0)
    X_train = np.hstack((y_train[:-1, :], u_train[:-1, :]))
    Xp_train = y_train[1:, :]
    X_valid = np.hstack((y_valid[:-1, :], u_valid[:-1, :]))
    Xp_valid = y_valid[1:, :]
    return {
        'X_train': X_train,
        'Xp_train': Xp_train,
        'X_valid': X_valid,
        'Xp_valid': Xp_valid,
        'n_inputs': 1,
        'episode_feature': False,
        'U_valid': U_valid,
    }


@pytest.fixture
def mass_spring_damper_no_input() -> Dict[str, Any]:
    """Compute mass-spring-damper response with no input."""
    # Set up problem
    t_range = (0, 10)
    t_step = 0.1
    msd = pykoop.dynamic_models.MassSpringDamper(0.5, 0.7, 0.6)
    # Initial condition
    x0 = np.array([1, 0])
    # Solve ODE for training data
    t, x = msd.simulate(
        t_range,
        t_step,
        x0,
        lambda t: np.zeros((1, )),
        rtol=1e-8,
        atol=1e-8,
    )
    U_valid = linalg.expm(msd.A * t_step).T
    # Split the data
    y_train, y_valid = np.split(x.T, 2, axis=1)
    X_train = y_train[:, :-1]
    Xp_train = y_train[:, 1:]
    X_valid = y_valid[:, :-1]
    Xp_valid = y_valid[:, 1:]
    return {
        'X_train': X_train,
        'Xp_train': Xp_train,
        'X_valid': X_valid,
        'Xp_valid': Xp_valid,
        'n_inputs': 0,
        'episode_feature': False,
        'U_valid': U_valid,
    }


# Add common packages and data to doctest namespace


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
