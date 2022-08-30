import numpy as np
import pandas
import pytest
from sklearn import preprocessing

import pykoop


poly_test_cases = [
    # Order 1, no inputs
    (
        np.array([
            [0, 1, 2, 3, 4, 5],
            [0, -1, -2, -3, -4, -5],
            [0, 2, 4, 5, 6, 10],
        ]).T,
        np.array([
            [0, 1, 2, 3, 4, 5],
            [0, -1, -2, -3, -4, -5],
            [0, 2, 4, 5, 6, 10],
        ]).T,
        pykoop.PolynomialLiftingFn(order=1),
        0,
    ),
    # Order 2, no inputs
    (
        np.array([
            [0, 1, 2, 3, 4, 5],
            [0, 2, 4, 5, 6, 10],
        ]).T,
        np.array([
            [0, 1, 2, 3, 4, 5],
            [0, 2, 4, 5, 6, 10],
            [0, 1, 4, 9, 16, 25],
            [0, 2, 8, 15, 24, 50],
            [0, 4, 16, 25, 36, 100],
        ]).T,
        pykoop.PolynomialLiftingFn(order=2),
        0,
    ),
    # Order 1, 1 input
    (
        np.array([
            [0, 1, 2, 3, 4, 5],
            [0, -1, -2, -3, -4, -5],
            [0, 2, 4, 5, 6, 10],
        ]).T,
        np.array([
            [0, 1, 2, 3, 4, 5],
            [0, -1, -2, -3, -4, -5],
            [0, 2, 4, 5, 6, 10],
        ]).T,
        pykoop.PolynomialLiftingFn(order=1),
        1,
    ),
    # Order 2, 1 input
    (
        np.array([
            [0, 1, 2, 3, 4, 5],
            [0, 2, 4, 5, 6, 10],
        ]).T,
        np.array([
            [0, 1, 2, 3, 4, 5],
            [0, 1, 4, 9, 16, 25],
            [0, 2, 4, 5, 6, 10],
            [0, 2, 8, 15, 24, 50],
            [0, 4, 16, 25, 36, 100],
        ]).T,
        pykoop.PolynomialLiftingFn(order=2),
        1,
    ),
    # Order 2, 0 input
    (
        np.array([
            [0, 1, 2, 3, 4, 5],
            [0, 2, 4, 6, 8, 10],
            [1, 3, 5, 7, 9, 11],
        ]).T,
        np.array([
            [0, 1, 2, 3, 4, 5],
            [0, 2, 4, 6, 8, 10],
            [1, 3, 5, 7, 9, 11],
            [0, 1, 4, 9, 16, 25],
            [0, 2, 8, 18, 32, 50],
            [0, 3, 10, 21, 36, 55],
            [0, 4, 16, 36, 64, 100],
            [0, 6, 20, 42, 72, 110],
            [1, 9, 25, 49, 81, 121],
        ]).T,
        pykoop.PolynomialLiftingFn(order=2),
        0,
    ),
    # Order 2, 1 input
    (
        np.array([
            [0, 1, 2, 3, 4, 5],
            [0, 2, 4, 6, 8, 10],
            [1, 3, 5, 7, 9, 11],
        ]).T,
        np.array([
            # State
            [0, 1, 2, 3, 4, 5],
            [0, 2, 4, 6, 8, 10],
            [0, 1, 4, 9, 16, 25],
            [0, 2, 8, 18, 32, 50],
            [0, 4, 16, 36, 64, 100],
            # Input
            [1, 3, 5, 7, 9, 11],
            [0, 3, 10, 21, 36, 55],
            [0, 6, 20, 42, 72, 110],
            [1, 9, 25, 49, 81, 121],
        ]).T,
        pykoop.PolynomialLiftingFn(order=2),
        1,
    ),
    # Order 2, 2 input
    (
        np.array([
            [0, 1, 2, 3, 4, 5],
            [0, 2, 4, 6, 8, 10],
            [1, 3, 5, 7, 9, 11],
        ]).T,
        np.array([
            # State
            [0, 1, 2, 3, 4, 5],
            [0, 1, 4, 9, 16, 25],
            # Input
            [0, 2, 4, 6, 8, 10],
            [1, 3, 5, 7, 9, 11],
            [0, 2, 8, 18, 32, 50],
            [0, 3, 10, 21, 36, 55],
            [0, 4, 16, 36, 64, 100],
            [0, 6, 20, 42, 72, 110],
            [1, 9, 25, 49, 81, 121],
        ]).T,
        pykoop.PolynomialLiftingFn(order=2),
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


delay_test_cases_noeps = [
    (
        # Tests with no input
        0,
        0,
        0,
        np.array([
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
        ]).T,
        np.array([
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
        ]).T,
    ),
    (
        1,
        0,
        0,
        np.array([
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
        ]).T,
        np.array([
            [2, 3, 4],
            [-2, -3, -4],
            [1, 2, 3],
            [-1, -2, -3],
        ]).T,
    ),
    (
        2,
        0,
        0,
        np.array([
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
        ]).T,
        np.array([
            [3, 4],
            [-3, -4],
            [2, 3],
            [-2, -3],
            [1, 2],
            [-1, -2],
        ]).T,
    ),
    (
        3,
        0,
        0,
        np.array([
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
        ]).T,
        np.array([
            [4],
            [-4],
            [3],
            [-3],
            [2],
            [-2],
            [1],
            [-1],
        ]).T,
    ),
    (
        # Tests with input, same delays for x and u
        0,
        0,
        2,
        np.array([
            # State
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
            # Input
            [2, 3, 4, 5],
            [0, -1, -2, -3],
        ]).T,
        np.array([
            # State
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
            # Input
            [2, 3, 4, 5],
            [0, -1, -2, -3],
        ]).T,
    ),
    (
        1,
        1,
        2,
        np.array([
            # State
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
            # Input
            [2, 3, 4, 5],
            [0, -1, -2, -3],
        ]).T,
        np.array([
            # State
            [2, 3, 4],
            [-2, -3, -4],
            [1, 2, 3],
            [-1, -2, -3],
            # Input
            [3, 4, 5],
            [-1, -2, -3],
            [2, 3, 4],
            [0, -1, -2],
        ]).T,
    ),
    (
        2,
        2,
        2,
        np.array([
            # State
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
            # Input
            [2, 3, 4, 5],
            [0, -1, -2, -3],
        ]).T,
        np.array([
            # State
            [3, 4],
            [-3, -4],
            [2, 3],
            [-2, -3],
            [1, 2],
            [-1, -2],
            # Input
            [4, 5],
            [-2, -3],
            [3, 4],
            [-1, -2],
            [2, 3],
            [0, -1],
        ]).T,
    ),
    (
        3,
        3,
        2,
        np.array([
            # State
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
            # Input
            [2, 3, 4, 5],
            [0, -1, -2, -3],
        ]).T,
        np.array([
            # State
            [4],
            [-4],
            [3],
            [-3],
            [2],
            [-2],
            [1],
            [-1],
            # Input
            [5],
            [-3],
            [4],
            [-2],
            [3],
            [-1],
            [2],
            [0],
        ]).T,
    ),
    (
        # Tests with input, different delays for x and u
        0,
        1,
        2,
        np.array([
            # State
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
            # Input
            [2, 3, 4, 5],
            [0, -1, -2, -3],
        ]).T,
        np.array([
            # State
            [2, 3, 4],
            [-2, -3, -4],
            # Input
            [3, 4, 5],
            [-1, -2, -3],
            [2, 3, 4],
            [0, -1, -2],
        ]).T,
    ),
    (
        1,
        0,
        2,
        np.array([
            # State
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
            # Input
            [2, 3, 4, 5],
            [0, -1, -2, -3],
        ]).T,
        np.array([
            # State
            [2, 3, 4],
            [-2, -3, -4],
            [1, 2, 3],
            [-1, -2, -3],
            # Input
            [3, 4, 5],
            [-1, -2, -3],
        ]).T,
    )
]


@pytest.mark.parametrize('n_delays_state, n_delays_input, n_u, X, Xd_exp',
                         delay_test_cases_noeps)
def test_delay_forward_noeps(n_delays_state, n_delays_input, n_u, X, Xd_exp):
    lf = pykoop.DelayLiftingFn(n_delays_state=n_delays_state,
                               n_delays_input=n_delays_input)
    # Check forward transform
    lf.fit(X, n_inputs=n_u, episode_feature=False)
    Xd_fit = lf.transform(X)
    np.testing.assert_allclose(Xd_exp, Xd_fit)


@pytest.mark.parametrize('n_delays_state, n_delays_input, n_u, X, Xd_exp',
                         delay_test_cases_noeps)
def test_delay_inverse_noeps(n_delays_state, n_delays_input, n_u, X, Xd_exp):
    lf = pykoop.DelayLiftingFn(n_delays_state=n_delays_state,
                               n_delays_input=n_delays_input)
    lf.fit(X, n_inputs=n_u, episode_feature=False)
    Xd_fit = lf.transform(X)
    # Check inverse transform
    Xd_inv = lf.inverse_transform(Xd_fit)
    # If the number of delays for x and u are different, only the last samples
    # will be the same.
    np.testing.assert_allclose(X[-Xd_inv.shape[0]:, :], Xd_inv)


delay_test_cases_eps = [
    (
        0,
        0,
        0,
        np.array([
            [0, 0, 0, 0, 1, 1, 1],
            [1, 2, 3, 4, 5, 6, 7],
            [-1, -2, -3, -4, -5, -6, -7],
        ]).T,
        np.array([
            [0, 0, 0, 0, 1, 1, 1],
            [1, 2, 3, 4, 5, 6, 7],
            [-1, -2, -3, -4, -5, -6, -7],
        ]).T,
    ),
    (
        1,
        1,
        0,
        np.array([
            [0, 0, 0, 0, 1, 1, 1],
            [1, 2, 3, 4, 5, 6, 7],
            [-1, -2, -3, -4, -5, -6, -7],
        ]).T,
        np.array([
            [0, 0, 0, 1, 1],
            [2, 3, 4, 6, 7],
            [-2, -3, -4, -6, -7],
            [1, 2, 3, 5, 6],
            [-1, -2, -3, -5, -6],
        ]).T,
    ),
    (
        1,
        1,
        1,
        np.array([
            [0, 0, 0, 0, 1, 1, 1],
            [1, 2, 3, 4, 5, 6, 7],
            [-1, -2, -3, -4, -5, -6, -7],
        ]).T,
        np.array([
            [0, 0, 0, 1, 1],
            [2, 3, 4, 6, 7],
            [1, 2, 3, 5, 6],
            [-2, -3, -4, -6, -7],
            [-1, -2, -3, -5, -6],
        ]).T,
    ),
    (
        2,
        2,
        0,
        np.array([
            [0, 0, 0, 0, 1, 1, 1],
            [1, 2, 3, 4, 5, 6, 7],
            [-1, -2, -3, -4, -5, -6, -7],
        ]).T,
        np.array([
            [0, 0, 1],
            [3, 4, 7],
            [-3, -4, -7],
            [2, 3, 6],
            [-2, -3, -6],
            [1, 2, 5],
            [-1, -2, -5],
        ]).T,
    ),
]


@pytest.mark.parametrize('n_delays_state, n_delays_input, n_u, X, Xd_exp',
                         delay_test_cases_eps)
def test_delay_forward_eps(n_delays_state, n_delays_input, n_u, X, Xd_exp):
    lf = pykoop.DelayLiftingFn(n_delays_state=n_delays_state,
                               n_delays_input=n_delays_input)
    # Check forward transform
    lf.fit(X, n_inputs=n_u, episode_feature=True)
    Xd_fit = lf.transform(X)
    np.testing.assert_allclose(Xd_exp, Xd_fit)


@pytest.mark.parametrize('n_delays_state, n_delays_input, n_u, X, Xd_exp',
                         delay_test_cases_eps)
def test_delay_inverse_eps(n_delays_state, n_delays_input, n_u, X, Xd_exp):
    lf = pykoop.DelayLiftingFn(n_delays_state=n_delays_state,
                               n_delays_input=n_delays_input)
    lf.fit(X, n_inputs=n_u, episode_feature=True)
    Xd_fit = lf.transform(X)
    # Check inverse transform
    Xd_inv = lf.inverse_transform(Xd_fit)
    # If the number of delays for x and u are different, only the last samples
    # will be the same in each episode. Must compare the last samples of each
    # episode to ensure correctness.
    for i in pandas.unique(X[:, 0]):
        X_i = X[X[:, 0] == i, :]
        Xd_inv_i = Xd_inv[Xd_inv[:, 0] == i, :]
        np.testing.assert_allclose(X_i[-Xd_inv_i.shape[0]:, :], Xd_inv_i)


episode_indep_test_cases = [
    (
        pykoop.SkLearnLiftingFn(preprocessing.MaxAbsScaler()),
        np.array([
            [1., -1., 2.],
            [2., 0., 0.],
            [0., 1., -1.],
        ]),
        np.array([
            [0.5, -1., 1.],
            [1., 0., 0.],
            [0., 1., -0.5],
        ]),
        0,
        False,
    ),
    (
        pykoop.SkLearnLiftingFn(preprocessing.StandardScaler()),
        np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ]),
        np.array([
            [0, -1, -1],
            [0, -1, -1],
            [0, 1, 1],
            [0, 1, 1],
        ]),
        0,
        True,
    ),
    (
        pykoop.SkLearnLiftingFn(
            preprocessing.FunctionTransformer(
                func=np.log1p,
                inverse_func=lambda x: np.exp(x) - 1,
            )),
        np.array([
            [0, 1],
            [2, 3],
        ]),
        np.array([
            [0., 0.69314718],
            [1.09861229, 1.38629436],
        ]),
        0,
        False,
    ),
    (
        pykoop.BilinearInputLiftingFn(),
        np.array([
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
        ]).T,
        np.array([
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
        ]).T,
        0,
        False,
    ),
    (
        pykoop.BilinearInputLiftingFn(),
        np.array([
            # States
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
            # Inputs
            [5, 4, 3, 2, 1, 1],
        ]).T,
        np.array([
            # x
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
            # u
            [5, 4, 3, 2, 1, 1],
            # x * u1
            [0, 4, 6, 6, 4, 5],
            [5, 8, 9, 8, 5, 6],
            [30, 20, 12, 6, 2, 1],
        ]).T,
        1,
        False,
    ),
    (
        pykoop.BilinearInputLiftingFn(),
        np.array([
            # States
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            # Inputs
            [6, 5, 4, 3, 2, 1],
            [5, 4, 3, 2, 1, 1],
        ]).T,
        np.array([
            # x
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            # u
            [6, 5, 4, 3, 2, 1],
            [5, 4, 3, 2, 1, 1],
            # x * u1
            [0, 5, 8, 9, 8, 5],
            [6, 10, 12, 12, 10, 6],
            # x * u2
            [0, 4, 6, 6, 4, 5],
            [5, 8, 9, 8, 5, 6],
        ]).T,
        2,
        False,
    ),
    (
        pykoop.BilinearInputLiftingFn(),
        np.array([
            # States
            [0, 1, 2, 3, 4, 5],
            # Inputs
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
            [5, 4, 3, 2, 1, 1],
        ]).T,
        np.array([
            # x
            [0, 1, 2, 3, 4, 5],
            # u
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
            [5, 4, 3, 2, 1, 1],
            # x * u1
            [0, 2, 6, 12, 20, 30],
            # x * u2
            [0, 5, 8, 9, 8, 5],
            # x * u3
            [0, 4, 6, 6, 4, 5],
        ]).T,
        3,
        False,
    ),
]


@pytest.mark.parametrize('lf, X, Xt_exp, n_inputs, episode_feature',
                         episode_indep_test_cases)
def test_lifting_fn_transform(lf, X, Xt_exp, n_inputs, episode_feature):
    lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
    Xt = lf.transform(X)
    np.testing.assert_allclose(Xt, Xt_exp)


@pytest.mark.parametrize('lf, X, Xt_exp, n_inputs, episode_feature',
                         episode_indep_test_cases)
def test_lifting_fn_inverse(lf, X, Xt_exp, n_inputs, episode_feature):
    lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
    Xt = lf.transform(X)
    Xi = lf.inverse_transform(Xt)
    np.testing.assert_allclose(Xi, X)
