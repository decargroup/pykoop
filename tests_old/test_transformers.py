import numpy as np
import pandas
import pytest
from sklearn import preprocessing

import pykoop

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
