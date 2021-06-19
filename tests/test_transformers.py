import pytest
from pykoop import lifting_functions
import numpy as np


def test_preprocess_noeps():
    ang = np.array([0, 1, 0], dtype=bool)
    pp = lifting_functions.AnglePreprocessor(angles=ang)
    X = np.array([
        [ 0,     1,  2,        3],  # noqa: E201
        [ 0, np.pi,  0, -np.pi/2],  # noqa: E201
        [-1,    -2, -1,       -2]
    ]).T
    Xt_exp = np.array([
        [ 0,  1,  2,  3],  # noqa: E201
        [ 1, -1,  1,  0],  # noqa: E201
        [ 0,  0,  0, -1],  # noqa: E201
        [-1, -2, -1, -2]
    ]).T
    pp.fit(X, episode_feature=False)
    Xt = pp.transform(X)
    np.testing.assert_allclose(Xt_exp, Xt, atol=1e-15)
    Xi = pp.inverse_transform(Xt)
    np.testing.assert_allclose(X, Xi)


def test_preprocess_eps():
    ang = np.array([0, 1, 0], dtype=bool)
    pp = lifting_functions.AnglePreprocessor(angles=ang)
    X = np.array([
        # Episodes
        [ 0,     0,  1,        1],
        # Data
        [ 0,     1,  2,        3],  # noqa: E201
        [ 0, np.pi,  0, -np.pi/2],  # noqa: E201
        [-1,    -2, -1,       -2]
    ]).T
    Xt_exp = np.array([
        # Episodes
        [ 0, 0,   1,  1],
        # Data
        [ 0,  1,  2,  3],  # noqa: E201
        [ 1, -1,  1,  0],  # noqa: E201
        [ 0,  0,  0, -1],  # noqa: E201
        [-1, -2, -1, -2]
    ]).T
    pp.fit(X, episode_feature=True)
    Xt = pp.transform(X)
    np.testing.assert_allclose(Xt_exp, Xt, atol=1e-15)
    Xi = pp.inverse_transform(Xt)
    np.testing.assert_allclose(X, Xi)


def test_preprocess_angle_wrap():
    ang = np.array([0, 1, 0], dtype=bool)
    pp = lifting_functions.AnglePreprocessor(angles=ang)
    X = np.array([
        [      0,     1,  2,        3],  # noqa: E201
        [2*np.pi, np.pi,  0, -np.pi/2],
        [     -1,    -2, -1,       -2]  # noqa: E201 E221
    ]).T
    Xt_exp = np.array([
        [ 0,  1,  2,  3],  # noqa: E201
        [ 1, -1,  1,  0],  # noqa: E201
        [ 0,  0,  0, -1],  # noqa: E201
        [-1, -2, -1, -2]
    ]).T
    Xi_exp = np.array([
        [ 0,     1,  2,        3],  # noqa: E201
        [ 0, np.pi,  0, -np.pi/2],  # noqa: E201
        [-1,    -2, -1,       -2]
    ]).T
    pp.fit(X, episode_feature=False)
    Xt = pp.transform(X)
    np.testing.assert_allclose(Xt_exp, Xt, atol=1e-15)
    Xi = pp.inverse_transform(Xt)
    np.testing.assert_allclose(Xi_exp, Xi, atol=1e-15)


def test_preprocess_fit_features():
    X = np.zeros((2, 5))
    # Mix of linear and angles
    ang = np.array([0, 0, 1, 1, 0], dtype=bool)
    pp = lifting_functions.AnglePreprocessor(angles=ang)
    pp.fit(X, episode_feature=False)
    lin = np.array([1, 1, 0, 0, 0, 0, 1], dtype=bool)
    cos = np.array([0, 0, 1, 0, 1, 0, 0], dtype=bool)
    sin = np.array([0, 0, 0, 1, 0, 1, 0], dtype=bool)
    np.testing.assert_allclose(lin, pp.lin_)
    np.testing.assert_allclose(cos, pp.cos_)
    np.testing.assert_allclose(sin, pp.sin_)
    # All linear
    ang = np.array([0, 0, 0, 0, 0], dtype=bool)
    pp = lifting_functions.AnglePreprocessor(angles=ang)
    pp.fit(X, episode_feature=False)
    lin = np.array([1, 1, 1, 1, 1], dtype=bool)
    cos = np.array([0, 0, 0, 0, 0], dtype=bool)
    sin = np.array([0, 0, 0, 0, 0], dtype=bool)
    np.testing.assert_allclose(lin, pp.lin_)
    np.testing.assert_allclose(cos, pp.cos_)
    np.testing.assert_allclose(sin, pp.sin_)
    # All angles
    ang = np.array([1, 1, 1, 1, 1], dtype=bool)
    pp = lifting_functions.AnglePreprocessor(angles=ang)
    pp.fit(X, episode_feature=False)
    lin = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    cos = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
    sin = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
    np.testing.assert_allclose(lin, pp.lin_)
    np.testing.assert_allclose(cos, pp.cos_)
    np.testing.assert_allclose(sin, pp.sin_)


poly_test_cases = [
    # Order 1, no inputs
    (
        np.array([
            [0,  1,  2,  3,  4,  5],
            [0, -1, -2, -3, -4, -5],
            [0,  2,  4,  5,  6, 10],
        ]).T,
        np.array([
            [0,  1,  2,  3,  4,  5],
            [0, -1, -2, -3, -4, -5],
            [0,  2,  4,  5,  6, 10],
        ]).T,
        lifting_functions.PolynomialLiftingFn(order=1),
        0,
    ),
    # Order 2, no inputs
    (
        np.array([
            [0,  1,  2,  3,  4,  5],
            [0,  2,  4,  5,  6, 10],
        ]).T,
        np.array([
            [0, 1, 2,   3,  4,   5],
            [0, 2, 4,   5,  6,  10],
            [0, 1, 4,   9, 16,  25],
            [0, 2, 8,  15, 24,  50],
            [0, 4, 16, 25, 36, 100],
        ]).T,
        lifting_functions.PolynomialLiftingFn(order=2),
        0,
    ),
    # Order 1, 1 input
    (
        np.array([
            [0,  1,  2,  3,  4,  5],
            [0, -1, -2, -3, -4, -5],
            [0,  2,  4,  5,  6, 10],
        ]).T,
        np.array([
            [0,  1,  2,  3,  4,  5],
            [0, -1, -2, -3, -4, -5],
            [0,  2,  4,  5,  6, 10],
        ]).T,
        lifting_functions.PolynomialLiftingFn(order=1),
        1,
    ),
    # Order 2, 1 input
    (
        np.array([
            [0,  1,  2,  3,  4,  5],
            [0,  2,  4,  5,  6, 10],
        ]).T,
        np.array([
            [0, 1, 2,   3,  4,   5],
            [0, 1, 4,   9, 16,  25],
            [0, 2, 4,   5,  6,  10],
            [0, 2, 8,  15, 24,  50],
            [0, 4, 16, 25, 36, 100],
        ]).T,
        lifting_functions.PolynomialLiftingFn(order=2),
        1,
    ),
    # Order 2, 0 input
    (
        np.array([
            [0,  1,  2,  3,  4,  5],
            [0,  2,  4,  6,  8, 10],
            [1,  3,  5,  7,  9, 11],
        ]).T,
        np.array([
            [0, 1,  2,  3,  4,   5],
            [0, 2,  4,  6,  8,  10],
            [1, 3,  5,  7,  9,  11],
            [0, 1,  4,  9, 16,  25],
            [0, 2,  8, 18, 32,  50],
            [0, 3, 10, 21, 36,  55],
            [0, 4, 16, 36, 64, 100],
            [0, 6, 20, 42, 72, 110],
            [1, 9, 25, 49, 81, 121],
        ]).T,
        lifting_functions.PolynomialLiftingFn(order=2),
        0,
    ),
    # Order 2, 1 input
    (
        np.array([
            [0,  1,  2,  3,  4,  5],
            [0,  2,  4,  6,  8, 10],
            [1,  3,  5,  7,  9, 11],
        ]).T,
        np.array([
            # State
            [0, 1,  2,  3,  4,   5],
            [0, 2,  4,  6,  8,  10],
            [0, 1,  4,  9, 16,  25],
            [0, 2,  8, 18, 32,  50],
            [0, 4, 16, 36, 64, 100],
            # Input
            [1, 3,  5,  7,  9,  11],
            [0, 3, 10, 21, 36,  55],
            [0, 6, 20, 42, 72, 110],
            [1, 9, 25, 49, 81, 121],
        ]).T,
        lifting_functions.PolynomialLiftingFn(order=2),
        1,
    ),
    # Order 2, 2 input
    (
        np.array([
            [0,  1,  2,  3,  4,  5],
            [0,  2,  4,  6,  8, 10],
            [1,  3,  5,  7,  9, 11],
        ]).T,
        np.array([
            # State
            [0, 1,  2,  3,  4,   5],
            [0, 1,  4,  9, 16,  25],
            # Input
            [0, 2,  4,  6,  8,  10],
            [1, 3,  5,  7,  9,  11],
            [0, 2,  8, 18, 32,  50],
            [0, 3, 10, 21, 36,  55],
            [0, 4, 16, 36, 64, 100],
            [0, 6, 20, 42, 72, 110],
            [1, 9, 25, 49, 81, 121],
        ]).T,
        lifting_functions.PolynomialLiftingFn(order=2),
        2,
    ),
]


@pytest.mark.parametrize('X, Xt_exp, poly, n_inputs', poly_test_cases)
def test_polynomial_forward_noeps(X, Xt_exp, poly, n_inputs):
    poly.fit(X, n_inputs=n_inputs, episode_feature=False)
    Xt = poly.transform(X)
    np.testing.assert_allclose(Xt_exp, Xt)


@pytest.mark.parametrize('X, Xt_exp, poly, n_inputs', poly_test_cases)
def test_polynomial_inverse_noeps(X, Xt_exp, poly, n_inputs):
    poly.fit(X, n_inputs=n_inputs, episode_feature=False)
    Xt = poly.transform(X)
    Xt_inv = poly.inverse_transform(Xt)
    np.testing.assert_allclose(X, Xt_inv)

@pytest.mark.parametrize('X, Xt_exp, poly, n_inputs', poly_test_cases)
def test_polynomial_forward_eps(X, Xt_exp, poly, n_inputs):
    X = np.hstack((
        np.zeros((X.shape[0], 1)),
        X,
    ))
    Xt_exp = np.hstack((
        np.zeros((X.shape[0], 1)),
        Xt_exp,
    ))
    poly.fit(X, n_inputs=n_inputs, episode_feature=True)
    Xt = poly.transform(X)
    np.testing.assert_allclose(Xt_exp, Xt)


@pytest.mark.parametrize('X, Xt_exp, poly, n_inputs', poly_test_cases)
def test_polynomial_inverse_eps(X, Xt_exp, poly, n_inputs):
    X = np.hstack((
        np.zeros((X.shape[0], 1)),
        X,
    ))
    Xt_exp = np.hstack((
        np.zeros((X.shape[0], 1)),
        Xt_exp,
    ))
    poly.fit(X, n_inputs=n_inputs, episode_feature=True)
    Xt = poly.transform(X)
    Xt_inv = poly.inverse_transform(Xt)
    np.testing.assert_allclose(X, Xt_inv)


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
    lf.fit(X, n_inputs=n_u, episode_feature=False)
    Xd_fit = lf.transform(X)
    np.testing.assert_allclose(Xd_exp, Xd_fit)


@pytest.mark.parametrize('n_delay_x, n_delay_u, n_u, X, Xd_exp',
                         delay_test_cases)
def test_delay_inverse(n_delay_x, n_delay_u, n_u, X, Xd_exp):
    lf = lifting_functions.Delay(n_delay_x=n_delay_x, n_delay_u=n_delay_u)
    lf.fit(X, n_inputs=n_u, episode_feature=False)
    Xd_fit = lf.transform(X)
    # Check inverse transform
    Xd_inv = lf.inverse_transform(Xd_fit)
    # If the number of delays for x and u are different, only the last samples
    # will be the same.
    np.testing.assert_allclose(X[-Xd_inv.shape[0]:, :], Xd_inv)
