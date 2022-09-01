import numpy as np
import pytest
from scipy import integrate, linalg

import pykoop
import pykoop.dynamic_models
import pykoop.koopman_pipeline


def test_kp_fit():
    """Test Koopman pipeline fit on a mass-spring-damper."""
    t_range = (0, 10)
    t_step = 0.1
    msd = pykoop.dynamic_models.MassSpringDamper(0.5, 0.7, 0.6)

    def u(t):
        return 0.1 * np.sin(t)

    # Solve ODE for training data
    x0 = msd.x0(np.array([0, 0]))
    t, x = msd.simulate(t_range, t_step, x0, u, rtol=1e-8, atol=1e-8)
    # Split the data
    X = np.hstack((
        np.zeros((t.shape[0], 1)),
        x,
        np.reshape(u(t), (-1, 1)),
    ))
    kp = pykoop.KoopmanPipeline(
        lifting_functions=[(
            'dl',
            pykoop.DelayLiftingFn(
                n_delays_state=0,
                n_delays_input=0,
            ),
        )],
        regressor=pykoop.Edmd(),
    )
    kp.fit(X, n_inputs=1, episode_feature=True)
    # Compute discrete-time A and B matrices
    Ad = linalg.expm(msd.A * t_step)

    def integrand(s):
        return linalg.expm(msd.A * (t_step - s)).ravel()

    Bd = integrate.quad_vec(integrand, 0, t_step)[0].reshape((2, 2)) @ msd.B
    U_exp = np.hstack((Ad, Bd)).T
    np.testing.assert_allclose(kp.regressor_.coef_, U_exp, atol=0.1)



@pytest.mark.parametrize(
    'params',
    [
        {
            'X_predicted': np.array([
                [1, 2, 3, 4],
                [2, 3, 3, 2],
            ]).T,
            'X_expected': np.array([
                [1, 2, 3, 4],
                [2, 3, 3, 2],
            ]).T,
            'n_steps': None,
            'discount_factor': 1,
            'regression_metric': 'neg_mean_squared_error',
            'min_samples': 1,
            'episode_feature': False,
            'score_exp': 0,
        },
        {
            'X_predicted': np.array([
                [1, 2],
                [2, 3],
            ]).T,
            'X_expected': np.array([
                [1, 4],
                [2, 2],
            ]).T,
            'n_steps': None,
            'discount_factor': 1,
            'regression_metric': 'neg_mean_squared_error',
            'min_samples': 1,
            'episode_feature': False,
            'score_exp': -np.mean([2**2, 1]),
        },
        {
            'X_predicted': np.array([
                [1, 2, 3, 5],
            ]).T,
            'X_expected': np.array([
                [1, 2, 3, 3],
            ]).T,
            'n_steps': None,
            'discount_factor': 1,
            'regression_metric': 'neg_mean_squared_error',
            'multistep': True,
            'min_samples': 1,
            'episode_feature': False,
            'score_exp': -np.mean([0, 0, 2**2]),
        },
        {
            'X_predicted': np.array([
                [1, 2, 3, 5],
            ]).T,
            'X_expected': np.array([
                [1, 2, 3, 3],
            ]).T,
            'n_steps': None,
            'discount_factor': 1,
            'regression_metric': 'neg_mean_absolute_error',
            'multistep': True,
            'min_samples': 1,
            'episode_feature': False,
            'score_exp': -np.mean([0, 0, 2]),
        },
        {
            'X_predicted': np.array([
                [1, 2, 3, 5],
            ]).T,
            'X_expected': np.array([
                [1, 2, 3, 4],
            ]).T,
            'n_steps': None,
            'discount_factor': 1,
            'regression_metric': 'neg_mean_squared_error',
            'min_samples': 2,
            'episode_feature': False,
            'score_exp': -np.mean([0, 1]),
        },
        {
            'X_predicted': np.array([
                [1, 2, 3, 5],
            ]).T,
            'X_expected': np.array([
                [1, 2, 3, 4],
            ]).T,
            'n_steps': 2,
            'discount_factor': 1,
            'regression_metric': 'neg_mean_squared_error',
            'min_samples': 1,
            'episode_feature': False,
            'score_exp': 0,
        },
        {
            'X_predicted': np.array([
                [1, 2, 3, 5],
            ]).T,
            'X_expected': np.array([
                [1, 2, 3, 4],
            ]).T,
            'n_steps': None,
            'discount_factor': 0.5,
            'regression_metric': 'neg_mean_squared_error',
            'min_samples': 1,
            'episode_feature': False,
            # (0 * 1 + 0 * 0.5 + 1 * 0.25) / (1 + 0.5 + 0.25)
            'score_exp': -0.25 / 1.75,
        },
        {
            'X_predicted': np.array([
                [0, 0, 0, 1, 1, 1],
                [1, 2, 3, 4, 5, 6],
            ]).T,
            'X_expected': np.array([
                [0, 0, 0, 1, 1, 1],
                [1, 2, 4, 4, 6, 6],
            ]).T,
            'n_steps': None,
            'discount_factor': 1,
            'regression_metric': 'neg_mean_squared_error',
            'min_samples': 1,
            'episode_feature': True,
            'score_exp': -np.mean([0, 1, 1, 0]),
        },
        {
            'X_predicted': np.array([
                [0, 0, 0, 1, 1, 1],
                [1, 2, 3, 4, 5, 6],
            ]).T,
            'X_expected': np.array([
                [0, 0, 0, 1, 1, 1],
                [1, 2, 4, 4, 6, 6],
            ]).T,
            'n_steps': 1,
            'discount_factor': 1,
            'regression_metric': 'neg_mean_squared_error',
            'min_samples': 1,
            'episode_feature': True,
            'score_exp': -np.mean([0, 1]),
        },
        {
            'X_predicted': np.array([
                [0, 0, 0, 1, 1, 1],
                [1, 2, 3, 4, 5, 6],
            ]).T,
            'X_expected': np.array([
                [0, 0, 0, 1, 1, 1],
                [1, 2, 4, 4, 6, 6],
            ]).T,
            'n_steps': None,
            'discount_factor': 0.5,
            'regression_metric': 'neg_mean_squared_error',
            'min_samples': 1,
            'episode_feature': True,
            'score_exp': -(0.5 + 1) / (1 + 0.5 + 1 + 0.5),
        },
        {
            'X_predicted': np.array([
                [0, 0, 0, 1, 1, 1],
                [1, 2, 3, 4, 5, 6],
            ]).T,
            'X_expected': np.array([
                [0, 0, 0, 1, 1, 1],
                [1, 2, 4, 4, 6, 6],
            ]).T,
            'n_steps': 1,
            'discount_factor': 0.5,
            'regression_metric': 'neg_mean_squared_error',
            'min_samples': 1,
            'episode_feature': True,
            'score_exp': -(0 + 1) / (1 + 0 + 1 + 0),
        },
    ])
def test_score_state(params):
    score = pykoop.score_state(
        params['X_predicted'],
        params['X_expected'],
        params['n_steps'],
        params['discount_factor'],
        params['regression_metric'],
        params['min_samples'],
        params['episode_feature'],
    )
    np.testing.assert_allclose(score, params['score_exp'])


@pytest.mark.parametrize(
    'X, n_steps, discount_factor, episode_feature, w_exp',
    [
        (
            np.array([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ]).T,
            2,
            1,
            False,
            np.array([1, 1, 0, 0]),
        ),
        (
            np.array([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ]).T,
            3,
            0.5,
            False,
            np.array([1, 0.5, 0.25, 0]),
        ),
        (
            np.array([
                [0, 0, 0, 1, 1, 1, 1],
                [1, 2, 3, 4, 5, 6, 7],
                [5, 6, 7, 8, 9, 10, 11],
            ]).T,
            2,
            0.5,
            True,
            np.array([1, 0.5, 0, 1, 0.5, 0, 0]),
        ),
        (
            np.array([
                [0, 0, 0, 1, 1, 1, 1],
                [1, 2, 3, 4, 5, 6, 7],
                [5, 6, 7, 8, 9, 10, 11],
            ]).T,
            10,
            1,
            True,
            np.array([1, 1, 1, 1, 1, 1, 1]),
        ),
        (
            np.array([
                [0, 0, 0, 1, 1, 1, 1],
                [1, 2, 3, 4, 5, 6, 7],
                [5, 6, 7, 8, 9, 10, 11],
            ]).T,
            10,
            0.1,
            True,
            np.array([1, 0.1, 0.01, 1, 0.1, 0.01, 0.001]),
        ),
    ],
)
def test_weights_from_data_matrix(
    X,
    n_steps,
    discount_factor,
    episode_feature,
    w_exp,
):
    w = pykoop.koopman_pipeline._weights_from_data_matrix(
        X,
        n_steps,
        discount_factor,
        episode_feature,
    )
    np.testing.assert_allclose(w, w_exp)


