import pytest
from pykoop import lifting_functions
import numpy as np

delay_test_cases = [(
    # Tests with no input
    0, 0, 0,
    np.array([
        [ 1,  2,  3,  4],  # noqa: E201
        [-1, -2, -3, -4],
    ]).T,
    np.array([
        [ 1,  2,  3,  4],  # noqa: E201
        [-1, -2, -3, -4],
    ]).T,
), (
    1, 0, 0,
    np.array([
        [ 1,  2,  3,  4],  # noqa: E201
        [-1, -2, -3, -4],
    ]).T,
    np.array([
        [ 2,  3,  4],  # noqa: E201
        [-2, -3, -4],
        [ 1,  2,  3],  # noqa: E201
        [-1, -2, -3],
    ]).T,
), (
    2, 0, 0,
    np.array([
        [ 1,  2,  3,  4],  # noqa: E201
        [-1, -2, -3, -4],
    ]).T,
    np.array([
        [ 3,  4],  # noqa: E201
        [-3, -4],
        [ 2,  3],  # noqa: E201
        [-2, -3],
        [ 1,  2],  # noqa: E201
        [-1, -2],
    ]).T,
), (
    3, 0, 0,
    np.array([
        [ 1,  2,  3,  4],  # noqa: E201
        [-1, -2, -3, -4],
    ]).T,
    np.array([
        [ 4],  # noqa: E201
        [-4],
        [ 3],  # noqa: E201
        [-3],
        [ 2],  # noqa: E201
        [-2],
        [ 1],  # noqa: E201
        [-1],
    ]).T,
), (
    # Tests with input, same delays for x and u
    0, 0, 2,
    np.array([
        # State
        [ 1,  2,  3,  4],  # noqa: E201
        [-1, -2, -3, -4],
        # Input
        [ 2,  3,  4,  5],  # noqa: E201
        [ 0, -1, -2, -3],  # noqa: E201
    ]).T,
    np.array([
        # State
        [ 1,  2,  3,  4],  # noqa: E201
        [-1, -2, -3, -4],
        # Input
        [ 2,  3,  4,  5],  # noqa: E201
        [ 0, -1, -2, -3],  # noqa: E201
    ]).T,
), (
    1, 1, 2,
    np.array([
        # State
        [ 1,  2,  3,  4],  # noqa: E201
        [-1, -2, -3, -4],
        # Input
        [ 2,  3,  4,  5],  # noqa: E201
        [ 0, -1, -2, -3],  # noqa: E201
    ]).T,
    np.array([
        # State
        [ 2,  3,  4],  # noqa: E201
        [-2, -3, -4],
        [ 1,  2,  3],  # noqa: E201
        [-1, -2, -3],
        # Input
        [ 3,  4,  5],  # noqa: E201
        [-1, -2, -3],
        [ 2,  3,  4],  # noqa: E201
        [ 0, -1, -2],  # noqa: E201
    ]).T,
), (
    2, 2, 2,
    np.array([
        # State
        [ 1,  2,  3,  4],  # noqa: E201
        [-1, -2, -3, -4],
        # Input
        [ 2,  3,  4,  5],  # noqa: E201
        [ 0, -1, -2, -3],  # noqa: E201
    ]).T,
    np.array([
        # State
        [ 3,  4],  # noqa: E201
        [-3, -4],
        [ 2,  3],  # noqa: E201
        [-2, -3],
        [ 1,  2],  # noqa: E201
        [-1, -2],
        # Input
        [ 4,  5],  # noqa: E201
        [-2, -3],
        [ 3,  4],  # noqa: E201
        [-1, -2],
        [ 2,  3],  # noqa: E201
        [ 0, -1],  # noqa: E201
    ]).T,
), (
    3, 3, 2,
    np.array([
        # State
        [ 1,  2,  3,  4],  # noqa: E201
        [-1, -2, -3, -4],
        # Input
        [ 2,  3,  4,  5],  # noqa: E201
        [ 0, -1, -2, -3],  # noqa: E201
    ]).T,
    np.array([
        # State
        [ 4],  # noqa: E201
        [-4],
        [ 3],  # noqa: E201
        [-3],
        [ 2],  # noqa: E201
        [-2],
        [ 1],  # noqa: E201
        [-1],
        # Input
        [ 5],  # noqa: E201
        [-3],
        [ 4],  # noqa: E201
        [-2],
        [ 3],  # noqa: E201
        [-1],
        [ 2],  # noqa: E201
        [ 0],  # noqa: E201
    ]).T,
), (
    # Tests with input, different delays for x and u
    0, 1, 2,
    np.array([
        # State
        [ 1,  2,  3,  4],  # noqa: E201
        [-1, -2, -3, -4],
        # Input
        [ 2,  3,  4,  5],  # noqa: E201
        [ 0, -1, -2, -3],  # noqa: E201
    ]).T,
    np.array([
        # State
        [ 2,  3,  4],  # noqa: E201
        [-2, -3, -4],  # noqa: E201
        # Input
        [ 3,  4,  5],  # noqa: E201
        [-1, -2, -3],
        [ 2,  3,  4],  # noqa: E201
        [ 0, -1, -2],  # noqa: E201
    ]).T,
), (
    1, 0, 2,
    np.array([
        # State
        [ 1,  2,  3,  4],  # noqa: E201
        [-1, -2, -3, -4],
        # Input
        [ 2,  3,  4,  5],  # noqa: E201
        [ 0, -1, -2, -3],  # noqa: E201
    ]).T,
    np.array([
        # State
        [ 2,  3,  4],  # noqa: E201
        [-2, -3, -4],  # noqa: E201
        [ 1,  2,  3],  # noqa: E201
        [-1, -2, -3],  # noqa: E201
        # Input
        [ 3,  4,  5],  # noqa: E201
        [-1, -2, -3],
    ]).T,
)]


@pytest.mark.parametrize('n_delay_x, n_delay_u, n_u, X, Xd_exp',
                         delay_test_cases)
def test_delay_forward(n_delay_x, n_delay_u, n_u, X, Xd_exp):
    lf = lifting_functions.Delay(n_delay_x=n_delay_x, n_delay_u=n_delay_u)
    # Check forward transform
    Xd_fit = lf.fit_transform(X, n_u=n_u)
    np.testing.assert_allclose(Xd_exp, Xd_fit)


@pytest.mark.parametrize('n_delay_x, n_delay_u, n_u, X, Xd_exp',
                         delay_test_cases)
def test_delay_inverse(n_delay_x, n_delay_u, n_u, X, Xd_exp):
    lf = lifting_functions.Delay(n_delay_x=n_delay_x, n_delay_u=n_delay_u)
    Xd_fit = lf.fit_transform(X, n_u=n_u)
    # Check inverse transform
    Xd_inv = lf.inverse_transform(Xd_fit)
    # If the number of delays for x and u are different, only the last samples
    # will be the same.
    np.testing.assert_allclose(X[-Xd_inv.shape[0]:, :], Xd_inv)
