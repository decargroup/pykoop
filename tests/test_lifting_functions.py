"""Test :mod:`lifting_functions`."""

import numpy as np
import pandas
import pytest
import sklearn.utils.estimator_checks
from sklearn import preprocessing

import pykoop


@pytest.mark.parametrize(
    'lf, X, Xt_exp, n_inputs, episode_feature',
    [
        # Polynomial, no episodes
        (
            pykoop.PolynomialLiftingFn(order=1),
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
            0,
            False,
        ),
        (
            pykoop.PolynomialLiftingFn(order=2),
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
            0,
            False,
        ),
        (
            pykoop.PolynomialLiftingFn(order=1),
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
            1,
            False,
        ),
        (
            pykoop.PolynomialLiftingFn(order=2),
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
            1,
            False,
        ),
        (
            pykoop.PolynomialLiftingFn(order=2),
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
            0,
            False,
        ),
        (
            pykoop.PolynomialLiftingFn(order=2),
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
            1,
            False,
        ),
        (
            pykoop.PolynomialLiftingFn(order=2),
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
            2,
            False,
        ),
        # Polynomial, episodes
        (
            pykoop.PolynomialLiftingFn(order=1),
            np.array([
                [0, 0, 0, 0, 1, 1],
                [0, 1, 2, 3, 4, 5],
                [0, -1, -2, -3, -4, -5],
                [0, 2, 4, 5, 6, 10],
            ]).T,
            np.array([
                [0, 0, 0, 0, 1, 1],
                [0, 1, 2, 3, 4, 5],
                [0, -1, -2, -3, -4, -5],
                [0, 2, 4, 5, 6, 10],
            ]).T,
            0,
            True,
        ),
        (
            pykoop.PolynomialLiftingFn(order=2),
            np.array([
                [0, 0, 0, 0, 1, 1],
                [0, 1, 2, 3, 4, 5],
                [0, 2, 4, 5, 6, 10],
            ]).T,
            np.array([
                [0, 0, 0, 0, 1, 1],
                [0, 1, 2, 3, 4, 5],
                [0, 2, 4, 5, 6, 10],
                [0, 1, 4, 9, 16, 25],
                [0, 2, 8, 15, 24, 50],
                [0, 4, 16, 25, 36, 100],
            ]).T,
            0,
            True,
        ),
        (
            pykoop.PolynomialLiftingFn(order=1),
            np.array([
                [0, 0, 0, 0, 1, 1],
                [0, 1, 2, 3, 4, 5],
                [0, -1, -2, -3, -4, -5],
                [0, 2, 4, 5, 6, 10],
            ]).T,
            np.array([
                [0, 0, 0, 0, 1, 1],
                [0, 1, 2, 3, 4, 5],
                [0, -1, -2, -3, -4, -5],
                [0, 2, 4, 5, 6, 10],
            ]).T,
            1,
            True,
        ),
        (
            pykoop.PolynomialLiftingFn(order=2),
            np.array([
                [0, 0, 0, 0, 1, 1],
                [0, 1, 2, 3, 4, 5],
                [0, 2, 4, 5, 6, 10],
            ]).T,
            np.array([
                [0, 0, 0, 0, 1, 1],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 4, 9, 16, 25],
                [0, 2, 4, 5, 6, 10],
                [0, 2, 8, 15, 24, 50],
                [0, 4, 16, 25, 36, 100],
            ]).T,
            1,
            True,
        ),
        (
            pykoop.PolynomialLiftingFn(order=2),
            np.array([
                [0, 0, 0, 0, 1, 1],
                [0, 1, 2, 3, 4, 5],
                [0, 2, 4, 6, 8, 10],
                [1, 3, 5, 7, 9, 11],
            ]).T,
            np.array([
                [0, 0, 0, 0, 1, 1],
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
            0,
            True,
        ),
        (
            pykoop.PolynomialLiftingFn(order=2),
            np.array([
                [0, 0, 0, 0, 1, 1],
                [0, 1, 2, 3, 4, 5],
                [0, 2, 4, 6, 8, 10],
                [1, 3, 5, 7, 9, 11],
            ]).T,
            np.array([
                # Episode
                [0, 0, 0, 0, 1, 1],
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
            1,
            True,
        ),
        (
            pykoop.PolynomialLiftingFn(order=2),
            np.array([
                [0, 0, 0, 0, 1, 1],
                [0, 1, 2, 3, 4, 5],
                [0, 2, 4, 6, 8, 10],
                [1, 3, 5, 7, 9, 11],
            ]).T,
            np.array([
                # Episode
                [0, 0, 0, 0, 1, 1],
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
            2,
            True,
        ),
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
    ],
)
class TestLiftingFnTransform:
    """Test lifting function transform and inverse transform."""

    def test_transform(self, lf, X, Xt_exp, n_inputs, episode_feature):
        """Test lifting function transform."""
        lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        Xt = lf.transform(X)
        np.testing.assert_allclose(Xt_exp, Xt)

    def test_inverse_transform(self, lf, X, Xt_exp, n_inputs, episode_feature):
        """Test lifting function inverse transform."""
        lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        Xt = lf.transform(X)
        Xt_inv = lf.inverse_transform(Xt)
        np.testing.assert_allclose(X, Xt_inv)


@pytest.mark.parametrize(
    'lf, X, Xt_exp, n_inputs, episode_feature',
    [
        # Delay, no episodes
        (
            pykoop.DelayLiftingFn(n_delays_state=0, n_delays_input=0),
            np.array([
                [1, 2, 3, 4],
                [-1, -2, -3, -4],
            ]).T,
            np.array([
                [1, 2, 3, 4],
                [-1, -2, -3, -4],
            ]).T,
            0,
            False,
        ),
        (
            pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=0),
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
            0,
            False,
        ),
        (
            pykoop.DelayLiftingFn(n_delays_state=2, n_delays_input=0),
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
            0,
            False,
        ),
        (
            pykoop.DelayLiftingFn(n_delays_state=3, n_delays_input=0),
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
            0,
            False,
        ),
        (
            pykoop.DelayLiftingFn(n_delays_state=0, n_delays_input=0),
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
            2,
            False,
        ),
        (
            pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=1),
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
            2,
            False,
        ),
        (
            pykoop.DelayLiftingFn(n_delays_state=2, n_delays_input=2),
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
            2,
            False,
        ),
        (
            pykoop.DelayLiftingFn(n_delays_state=3, n_delays_input=3),
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
            2,
            False,
        ),
        (
            pykoop.DelayLiftingFn(n_delays_state=0, n_delays_input=1),
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
            2,
            False,
        ),
        (
            pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=0),
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
            2,
            False,
        ),
        # Delay, episodes
        (
            pykoop.DelayLiftingFn(n_delays_state=0, n_delays_input=0),
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
            0,
            True,
        ),
        (
            pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=1),
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
            0,
            True,
        ),
        (
            pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=1),
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
            1,
            True,
        ),
        (
            pykoop.DelayLiftingFn(n_delays_state=2, n_delays_input=2),
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
            0,
            True,
        ),
        (
            pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=0),
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
            ]).T,
            1,
            True,
        ),
        (
            pykoop.DelayLiftingFn(n_delays_state=0, n_delays_input=1),
            np.array([
                [0, 0, 0, 0, 1, 1, 1],
                [1, 2, 3, 4, 5, 6, 7],
                [-1, -2, -3, -4, -5, -6, -7],
            ]).T,
            np.array([
                [0, 0, 0, 1, 1],
                [2, 3, 4, 6, 7],
                [-2, -3, -4, -6, -7],
                [-1, -2, -3, -5, -6],
            ]).T,
            1,
            True,
        ),
    ],
)
class TestDelayLiftingFnTransform:
    """Test :class:`DelayLiftingFn` transform and inverse transform."""

    def test_transform(self, lf, X, Xt_exp, n_inputs, episode_feature):
        """Test :class:`DelayLiftingFn` transform."""
        lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        Xt = lf.transform(X)
        np.testing.assert_allclose(Xt_exp, Xt)

    def test_inverse_transform(self, lf, X, Xt_exp, n_inputs, episode_feature):
        """Test :class:`DelayLiftingFn` inverse transform."""
        lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        Xt = lf.transform(X)
        Xt_inv = lf.inverse_transform(Xt)
        # If the number of delays for x and u are different, only the last
        # samples will be the same in each episode. Must compare the last
        # samples of each episode to ensure correctness.
        if episode_feature:
            episodes = []
            for i in pandas.unique(X[:, 0]):
                # Select episode and inverse
                X_i = X[X[:, 0] == i, :]
                Xt_inv_i = Xt_inv[Xt_inv[:, 0] == i, :]
                episodes.append(X_i[-Xt_inv_i.shape[0]:, :])
            Xt_inv_trimmed = np.vstack(episodes)
        else:
            Xt_inv_trimmed = X[-Xt_inv.shape[0]:, :]
        np.testing.assert_allclose(Xt_inv_trimmed, Xt_inv)


class TestSklearn:
    """Test scikit-learn compatibility."""

    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.PolynomialLiftingFn(),
        pykoop.DelayLiftingFn(),
        pykoop.SkLearnLiftingFn(preprocessing.MaxAbsScaler()),
        pykoop.BilinearInputLiftingFn(),
    ])
    def test_compatible_estimator(self, estimator, check):
        """Test scikit-learn compatibility of estimators."""
        check(estimator)
