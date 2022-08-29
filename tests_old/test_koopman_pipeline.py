import numpy as np
import pytest
from scipy import integrate, linalg

import pykoop
import pykoop.dynamic_models
import pykoop.koopman_pipeline


def test_kp_transform_no_lf():
    """Test Koopman pipeline transformer with no lifting functions."""
    # Create test matrix
    X = np.array([
        [1, 2, 3, 4, 5, 6],
        [-1, -2, -3, -4, -5, -6],
        [2, 4, 6, 8, 10, 12],
    ]).T
    # Create basic pipeline
    kp = pykoop.KoopmanPipeline(
        lifting_functions=None,
        regressor=None,
    )
    # Fit pipeline
    kp.fit_transformers(X, n_inputs=1)
    # Check dimensions
    assert kp.n_features_in_ == 3
    assert kp.n_states_in_ == 2
    assert kp.n_inputs_in_ == 1
    assert kp.n_features_out_ == 3
    assert kp.n_states_out_ == 2
    assert kp.n_inputs_out_ == 1
    # Transform
    Xt = kp.transform(X)
    np.testing.assert_allclose(Xt, X)
    # Inverse
    Xi = kp.inverse_transform(Xt)
    np.testing.assert_allclose(Xi, X)


def test_kp_transform_delay_lf():
    """Test Koopman pipeline transformer with delay lifting function."""
    # Create test matrix
    X = np.array([
        [1, 2, 3, 4, 5, 6],
        [-1, -2, -3, -4, -5, -6],
        [2, 4, 6, 8, 10, 12],
    ]).T
    Xt_exp = np.array([
        # State
        [2, 3, 4, 5, 6],
        [-2, -3, -4, -5, -6],
        [1, 2, 3, 4, 5],
        [-1, -2, -3, -4, -5],
        # Input
        [4, 6, 8, 10, 12],
        [2, 4, 6, 8, 10],
    ]).T
    kp = pykoop.KoopmanPipeline(
        lifting_functions=[(
            'dl',
            pykoop.DelayLiftingFn(
                n_delays_state=1,
                n_delays_input=1,
            ),
        )],
        regressor=None,
    )
    # Fit pipeline
    kp.fit_transformers(X, n_inputs=1)
    # Check dimensions
    assert kp.n_features_in_ == 3
    assert kp.n_states_in_ == 2
    assert kp.n_inputs_in_ == 1
    assert kp.n_features_out_ == 6
    assert kp.n_states_out_ == 4
    assert kp.n_inputs_out_ == 2
    # Transform
    Xt = kp.transform(X)
    np.testing.assert_allclose(Xt, Xt_exp)
    # Inverse
    Xi = kp.inverse_transform(Xt)
    np.testing.assert_allclose(Xi, X)


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
    'X, X_unsh_exp, X_sh_exp, n_inputs, episode_feature',
    [
        # Test without input or episode feature
        (
            np.array([
                [1, 2, 3, 4, 5],
                [-1, -2, -3, -4, -5],
            ]).T,
            np.array([
                [1, 2, 3, 4],
                [-1, -2, -3, -4],
            ]).T,
            np.array([
                [2, 3, 4, 5],
                [-2, -3, -4, -5],
            ]).T,
            0,
            False,
        ),
        # Test with input, without episode feature
        (
            np.array([
                [1, 2, 3, 4, 5],
                [-1, -2, -3, -4, -5],
            ]).T,
            np.array([
                [1, 2, 3, 4],
                [-1, -2, -3, -4],
            ]).T,
            np.array([
                [2, 3, 4, 5],
            ]).T,
            1,
            False,
        ),
        # Test without input, with episode feature
        (
            np.array([
                [0, 0, 0, 1, 1],
                [1, 2, 3, 4, 5],
                [-1, -2, -3, -4, -5],
            ]).T,
            np.array([
                [0, 0, 1],
                [1, 2, 4],
                [-1, -2, -4],
            ]).T,
            np.array([
                [0, 0, 1],
                [2, 3, 5],
                [-2, -3, -5],
            ]).T,
            0,
            True,
        ),
        # Test with input and episode feature
        (
            np.array([
                [0, 0, 1, 1, 1],
                [1, 2, 3, 4, 5],
                [-1, -2, -3, -4, -5],
            ]).T,
            np.array([
                [0, 1, 1],
                [1, 3, 4],
                [-1, -3, -4],
            ]).T,
            np.array([
                [0, 1, 1],
                [2, 4, 5],
            ]).T,
            1,
            True,
        ),
    ],
)
def test_shift_episodes(X, X_unsh_exp, X_sh_exp, n_inputs, episode_feature):
    """Test episode shifting."""
    X_unsh, X_sh = pykoop.shift_episodes(X, n_inputs, episode_feature)
    np.testing.assert_allclose(X_unsh, X_unsh_exp)
    np.testing.assert_allclose(X_sh, X_sh_exp)


split_combine_episode_scenarios = [
    (
        # Multiple episodes
        np.array([
            [0, 0, 0, 1, 1, 1],
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
        ]).T,
        [
            (
                0,
                np.array([
                    [1, 2, 3],
                    [6, 5, 4],
                ]).T,
            ),
            (
                1,
                np.array([
                    [4, 5, 6],
                    [3, 2, 1],
                ]).T,
            ),
        ],
        True,
    ),
    (
        # One episode
        np.array([
            [0, 0, 0, 0, 0, 0],
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
        ]).T,
        [
            (
                0,
                np.array([
                    [1, 2, 3, 4, 5, 6],
                    [6, 5, 4, 3, 2, 1],
                ]).T,
            ),
        ],
        True,
    ),
    (
        # No episode feature
        np.array([
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
        ]).T,
        [
            (
                0,
                np.array([
                    [1, 2, 3, 4, 5, 6],
                    [6, 5, 4, 3, 2, 1],
                ]).T,
            ),
        ],
        False,
    ),
    (
        # Out-of-order episodes
        np.array([
            [2, 2, 2, 0, 0, 1],
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
        ]).T,
        [
            (
                2,
                np.array([
                    [1, 2, 3],
                    [6, 5, 4],
                ]).T,
            ),
            (
                0,
                np.array([
                    [4, 5],
                    [3, 2],
                ]).T,
            ),
            (
                1,
                np.array([
                    [6],
                    [1],
                ]).T,
            ),
        ],
        True,
    ),
]


@pytest.mark.parametrize('X, episodes, episode_feature',
                         split_combine_episode_scenarios)
def test_split_episodes(X, episodes, episode_feature):
    # Split episodes
    episodes_actual = pykoop.split_episodes(X, episode_feature=episode_feature)
    # Compare every episode
    for actual, expected in zip(episodes_actual, episodes):
        i_actual, X_actual = actual
        i_expected, X_expected = expected
        assert i_actual == i_expected
        np.testing.assert_allclose(X_actual, X_expected)


@pytest.mark.parametrize('X, episodes, episode_feature',
                         split_combine_episode_scenarios)
def test_combine_episodes(X, episodes, episode_feature):
    X_actual = pykoop.combine_episodes(episodes,
                                       episode_feature=episode_feature)
    np.testing.assert_allclose(X_actual, X)


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


@pytest.mark.parametrize('X, min_samples, n_inputs, episode_feature, ic_exp', [
    (
        np.array([
            [1, 2, 3, 4],
            [4, 5, 6, 7],
        ]).T,
        1,
        0,
        False,
        np.array([
            [1],
            [4],
        ]).T,
    ),
    (
        np.array([
            [1, 2, 3, 4],
            [4, 5, 6, 7],
        ]).T,
        2,
        0,
        False,
        np.array([
            [1, 2],
            [4, 5],
        ]).T,
    ),
    (
        np.array([
            [1, 2, 3, 4],
            [4, 5, 6, 7],
            [5, 5, 5, 5],
        ]).T,
        1,
        1,
        False,
        np.array([
            [1],
            [4],
        ]).T,
    ),
    (
        np.array([
            [0, 0, 1, 1],
            [1, 2, 3, 4],
            [4, 5, 6, 7],
        ]).T,
        1,
        0,
        True,
        np.array([
            [0, 1],
            [1, 3],
            [4, 6],
        ]).T,
    ),
    (
        np.array([
            [0, 0, 1, 1],
            [1, 2, 3, 4],
            [4, 5, 6, 7],
            [9, 9, 9, 9],
        ]).T,
        1,
        1,
        True,
        np.array([
            [0, 1],
            [1, 3],
            [4, 6],
        ]).T,
    ),
    (
        np.array([
            [0, 0, 0, 1, 1, 1],
            [1, 2, 2, 3, 4, 5],
            [4, 5, 5, 6, 7, 6],
            [9, 9, 9, 9, 9, 6],
        ]).T,
        2,
        1,
        True,
        np.array([
            [0, 0, 1, 1],
            [1, 2, 3, 4],
            [4, 5, 6, 7],
        ]).T,
    ),
])
def test_extract_initial_conditions(
    X,
    min_samples,
    n_inputs,
    episode_feature,
    ic_exp,
):
    ic = pykoop.extract_initial_conditions(
        X,
        min_samples,
        n_inputs,
        episode_feature,
    )
    np.testing.assert_allclose(ic, ic_exp)


@pytest.mark.parametrize('X, n_inputs, episode_feature, u_exp', [
    (
        np.array([
            [1, 2, 3, 4],
            [6, 7, 8, 9],
        ]).T,
        1,
        False,
        np.array([
            [6, 7, 8, 9],
        ]).T,
    ),
    (
        np.array([
            [1, 2, 3, 4],
            [6, 7, 8, 9],
        ]).T,
        0,
        False,
        np.array([]).reshape((0, 4)).T,
    ),
    (
        np.array([
            [0, 0, 1, 1],
            [1, 2, 3, 4],
            [6, 7, 8, 9],
        ]).T,
        1,
        True,
        np.array([
            [0, 0, 1, 1],
            [6, 7, 8, 9],
        ]).T,
    ),
])
def test_extract_input(X, n_inputs, episode_feature, u_exp):
    u = pykoop.extract_input(X, n_inputs, episode_feature)
    np.testing.assert_allclose(u, u_exp)


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


@pytest.mark.parametrize('kp', [
    pykoop.KoopmanPipeline(
        lifting_functions=[
            ('dl', pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=1))
        ],
        regressor=pykoop.Edmd(),
    ),
    pykoop.KoopmanPipeline(
        lifting_functions=[
            ('dla', pykoop.DelayLiftingFn(n_delays_state=2, n_delays_input=2)),
            ('dlb', pykoop.DelayLiftingFn(n_delays_state=2, n_delays_input=2)),
        ],
        regressor=pykoop.Edmd(),
    ),
    pykoop.KoopmanPipeline(
        lifting_functions=[
            ('dla', pykoop.DelayLiftingFn(n_delays_state=2, n_delays_input=1)),
            ('ply', pykoop.PolynomialLiftingFn(order=2)),
            ('dlb', pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=2)),
        ],
        regressor=pykoop.Edmd(),
    ),
    pykoop.KoopmanPipeline(
        lifting_functions=[
            (
                'sp',
                pykoop.SplitPipeline(
                    lifting_functions_state=[
                        ('pl', pykoop.PolynomialLiftingFn(order=2)),
                    ],
                    lifting_functions_input=None,
                ),
            ),
            ('dl', pykoop.DelayLiftingFn(n_delays_state=2, n_delays_input=2)),
        ],
        regressor=pykoop.Edmd(),
    ),
    pykoop.KoopmanPipeline(
        lifting_functions=[
            ('dla', pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=1)),
            (
                'sp',
                pykoop.SplitPipeline(
                    lifting_functions_state=[
                        ('pla', pykoop.PolynomialLiftingFn(order=2)),
                        ('plb', pykoop.PolynomialLiftingFn(order=2)),
                    ],
                    lifting_functions_input=[
                        ('plc', pykoop.PolynomialLiftingFn(order=2)),
                    ],
                ),
            ),
            ('dlb', pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=1)),
        ],
        regressor=pykoop.Edmd(),
    ),
])
def test_multistep_prediction(kp):
    # Set up problem
    t_range = (0, 5)
    t_step = 0.1
    msd = pykoop.dynamic_models.MassSpringDamper(0.5, 0.7, 0.6)

    def u(t):
        return 0.1 * np.sin(t)

    # Solve ODE for training data
    x0 = np.array([0, 0])
    t, x = msd.simulate(t_range, t_step, x0, u, rtol=1e-8, atol=1e-8)

    # Compute input at every training point
    u_sim = np.reshape(u(t), (-1, 1))
    # Format data
    X = np.hstack((
        np.zeros((t.shape[0] - 1, 1)),
        x[:-1, :],
        u_sim[:-1, :],
    ))
    # Fit estimator
    kp.fit(X, n_inputs=1, episode_feature=True)
    n_samp = kp.min_samples_

    # Predict using ``predict_multistep``
    X_sim = np.empty(x.shape)
    X_sim[:n_samp, :] = x[:n_samp, :]
    X_sim = np.hstack((
        np.zeros_like(u_sim),
        X_sim,
        u_sim,
    ))
    X_sim = kp.predict_multistep(X_sim)[:, 1:]

    # Predict manually
    X_sim_exp = np.empty(x.shape)
    X_sim_exp[:, :n_samp] = x[:, :n_samp]
    for k in range(n_samp, t.shape[0]):
        X = np.hstack((
            np.zeros((n_samp, 1)),
            X_sim_exp[(k - n_samp):k, :],
            u_sim[(k - n_samp):k, :],
        ))
        Xp = kp.predict(X)
        X_sim_exp[[k], :] = Xp[[-1], 1:]

    np.testing.assert_allclose(X_sim, X_sim_exp)


test_split_lf_params = [
    # Basic, without episode feature
    (
        pykoop.SplitPipeline(
            lifting_functions_state=None,
            lifting_functions_input=None,
        ),
        np.array([
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [4, 5, 6, 7, 8, 9],
            [0, 8, 7, 6, 5, 4],
        ]).T,
        np.array([
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [4, 5, 6, 7, 8, 9],
            [0, 8, 7, 6, 5, 4],
        ]).T,
        2,
        False,
        {
            'n_features_in_': 4,
            'n_states_in_': 2,
            'n_inputs_in_': 2,
            'n_features_out_': 4,
            'n_states_out_': 2,
            'n_inputs_out_': 2,
            'min_samples_': 1,
        },
    ),
    # Basic, with episode feature
    (
        pykoop.SplitPipeline(
            lifting_functions_state=None,
            lifting_functions_input=None,
        ),
        np.array([
            [0, 0, 0, 0, 1, 1],
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [4, 5, 6, 7, 8, 9],
            [0, 8, 7, 6, 5, 4],
        ]).T,
        np.array([
            [0, 0, 0, 0, 1, 1],
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [4, 5, 6, 7, 8, 9],
            [0, 8, 7, 6, 5, 4],
        ]).T,
        2,
        True,
        {
            'n_features_in_': 5,
            'n_states_in_': 2,
            'n_inputs_in_': 2,
            'n_features_out_': 5,
            'n_states_out_': 2,
            'n_inputs_out_': 2,
            'min_samples_': 1,
        },
    ),
    # Lifting only state
    (
        pykoop.SplitPipeline(
            lifting_functions_state=[(
                'pl',
                pykoop.PolynomialLiftingFn(order=2),
            )],
            lifting_functions_input=None,
        ),
        np.array([
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [4, 5, 6, 7, 8, 9],
            [0, 8, 7, 6, 5, 4],
        ]).T,
        np.array([
            # State
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [0, 1, 4, 9, 16, 25],
            [0, 4, 6, 6, 4, 0],
            [25, 16, 9, 4, 1, 0],
            # Input
            [4, 5, 6, 7, 8, 9],
            [0, 8, 7, 6, 5, 4],
        ]).T,
        2,
        False,
        {
            'n_features_in_': 4,
            'n_states_in_': 2,
            'n_inputs_in_': 2,
            'n_features_out_': 7,
            'n_states_out_': 5,
            'n_inputs_out_': 2,
            'min_samples_': 1,
        },
    ),
    # Lifting only input
    (
        pykoop.SplitPipeline(
            lifting_functions_state=None,
            lifting_functions_input=[(
                'pl',
                pykoop.PolynomialLiftingFn(order=2),
            )],
        ),
        np.array([
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [4, 5, 6, 7, 8, 9],
            [0, 8, 7, 6, 5, 4],
        ]).T,
        np.array([
            # State
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            # Input
            [4, 5, 6, 7, 8, 9],
            [0, 8, 7, 6, 5, 4],
            [16, 25, 36, 49, 64, 81],
            [0, 40, 42, 42, 40, 36],
            [0, 64, 49, 36, 25, 16],
        ]).T,
        2,
        False,
        {
            'n_features_in_': 4,
            'n_states_in_': 2,
            'n_inputs_in_': 2,
            'n_features_out_': 7,
            'n_states_out_': 2,
            'n_inputs_out_': 5,
            'min_samples_': 1,
        },
    ),
    # Lifting both
    (
        pykoop.SplitPipeline(
            lifting_functions_state=[(
                'pla',
                pykoop.PolynomialLiftingFn(order=2, interaction_only=True),
            )],
            lifting_functions_input=[(
                'plb',
                pykoop.PolynomialLiftingFn(order=2),
            )],
        ),
        np.array([
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [4, 5, 6, 7, 8, 9],
            [0, 8, 7, 6, 5, 4],
        ]).T,
        np.array([
            # State
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [0, 4, 6, 6, 4, 0],
            # Input
            [4, 5, 6, 7, 8, 9],
            [0, 8, 7, 6, 5, 4],
            [16, 25, 36, 49, 64, 81],
            [0, 40, 42, 42, 40, 36],
            [0, 64, 49, 36, 25, 16],
        ]).T,
        2,
        False,
        {
            'n_features_in_': 4,
            'n_states_in_': 2,
            'n_inputs_in_': 2,
            'n_features_out_': 8,
            'n_states_out_': 3,
            'n_inputs_out_': 5,
            'min_samples_': 1,
        },
    ),
]


@pytest.mark.parametrize('kp', [
    pykoop.KoopmanPipeline(
        lifting_functions=[
            ('dl', pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=1))
        ],
        regressor=pykoop.Edmd(),
    ),
    pykoop.KoopmanPipeline(
        lifting_functions=[
            ('dla', pykoop.DelayLiftingFn(n_delays_state=2, n_delays_input=2)),
            ('dlb', pykoop.DelayLiftingFn(n_delays_state=2, n_delays_input=2)),
        ],
        regressor=pykoop.Edmd(),
    ),
    pykoop.KoopmanPipeline(
        lifting_functions=[
            ('dla', pykoop.DelayLiftingFn(n_delays_state=2, n_delays_input=1)),
            ('ply', pykoop.PolynomialLiftingFn(order=2)),
            ('dlb', pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=2)),
        ],
        regressor=pykoop.Edmd(),
    ),
    pykoop.KoopmanPipeline(
        lifting_functions=[
            (
                'sp',
                pykoop.SplitPipeline(
                    lifting_functions_state=[
                        ('pl', pykoop.PolynomialLiftingFn(order=2)),
                    ],
                    lifting_functions_input=None,
                ),
            ),
            ('dl', pykoop.DelayLiftingFn(n_delays_state=2, n_delays_input=2)),
        ],
        regressor=pykoop.Edmd(),
    ),
    pykoop.KoopmanPipeline(
        lifting_functions=[
            ('dla', pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=1)),
            (
                'sp',
                pykoop.SplitPipeline(
                    lifting_functions_state=[
                        ('pla', pykoop.PolynomialLiftingFn(order=2)),
                        ('plb', pykoop.PolynomialLiftingFn(order=2)),
                    ],
                    lifting_functions_input=[
                        ('plc', pykoop.PolynomialLiftingFn(order=2)),
                    ],
                ),
            ),
            ('dlb', pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=1)),
        ],
        regressor=pykoop.Edmd(),
    ),
])
def test_predict_state(kp):
    # Set up problem
    t_range = (0, 5)
    t_step = 0.1
    msd = pykoop.dynamic_models.MassSpringDamper(0.5, 0.7, 0.6)

    def u(t):
        return 0.1 * np.sin(t)

    # Solve ODE for training data
    x0 = np.array([0, 0])
    t, x = msd.simulate(t_range, t_step, x0, u, rtol=1e-8, atol=1e-8)

    # Compute input at every training point
    u_sim = np.reshape(u(t), (-1, 1))
    # Format data
    X = np.hstack((
        np.zeros((t.shape[0] - 1, 1)),
        x[:-1, :],
        u_sim[:-1, :],
    ))
    # Fit estimator
    kp.fit(X, n_inputs=1, episode_feature=True)
    n_samp = kp.min_samples_

    X_sim = kp.predict_state(
        x[:n_samp, :],
        u_sim,
        episode_feature=False,
        relift_state=True,
    )

    # Predict manually
    X_sim_exp = np.empty(x.shape)
    X_sim_exp[:, :n_samp] = x[:, :n_samp]
    for k in range(n_samp, t.shape[0]):
        X = np.hstack((
            np.zeros((n_samp, 1)),
            X_sim_exp[(k - n_samp):k, :],
            u_sim[(k - n_samp):k, :],
        ))
        Xp = kp.predict(X)
        X_sim_exp[[k], :] = Xp[[-1], 1:]

    np.testing.assert_allclose(X_sim, X_sim_exp)


@pytest.mark.parametrize('lf, X, Xt_exp, n_inputs, episode_feature, attr_exp',
                         test_split_lf_params)
def test_split_lifting_fn_attrs(lf, X, Xt_exp, n_inputs, episode_feature,
                                attr_exp):
    # Fit estimator
    lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
    # Check attributes
    attr = {key: getattr(lf, key) for key in attr_exp.keys()}
    assert attr == attr_exp


@pytest.mark.parametrize('lf, X, Xt_exp, n_inputs, episode_feature, attr_exp',
                         test_split_lf_params)
def test_split_lifting_fn_transform(lf, X, Xt_exp, n_inputs, episode_feature,
                                    attr_exp):
    # Fit estimator
    lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
    Xt = lf.transform(X)
    np.testing.assert_allclose(Xt, Xt_exp)


@pytest.mark.parametrize('lf, X, Xt_exp, n_inputs, episode_feature, attr_exp',
                         test_split_lf_params)
def test_split_lifting_fn_inverse_transform(lf, X, Xt_exp, n_inputs,
                                            episode_feature, attr_exp):
    # Fit estimator
    lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
    Xt = lf.transform(X)
    Xi = lf.inverse_transform(Xt)
    np.testing.assert_allclose(Xi, X)


def test_strip_initial_conditons():
    X1 = np.array([
        [0, 0, 1, 1, 1, 2, 2, 2],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 3, 4, 5, 6, 7, 8, 9],
    ]).T
    X2 = np.array([
        [0, 0, 1, 1, 1, 2, 2, 2],
        [-1, 2, -1, 4, 5, -1, 7, 8],
        [-1, 3, -1, 5, 6, -1, 8, 9],
    ]).T
    X1s = pykoop.strip_initial_conditions(X1, 1, True)
    X2s = pykoop.strip_initial_conditions(X2, 1, True)
    np.testing.assert_allclose(X1s, X2s)


@pytest.fixture(params=[
    pykoop.PolynomialLiftingFn(order=2),
    pykoop.KoopmanPipeline(
        lifting_functions=[
            ('p', pykoop.PolynomialLiftingFn(order=2)),
        ],
        regressor=pykoop.Edmd(),
    )
])
def lift_retract_fixture(request):
    lf = request.param
    X = np.array([
        [0, 1, 2, 3, 4, 5],
        [0, 2, 4, 5, 6, 10],
    ]).T
    Xt = np.array([
        [0, 1, 2, 3, 4, 5],
        [0, 1, 4, 9, 16, 25],
        [0, 2, 4, 5, 6, 10],
        [0, 2, 8, 15, 24, 50],
        [0, 4, 16, 25, 36, 100],
    ]).T
    return (lf, X, Xt)


def test_lift_retract_ff(lift_retract_fixture):
    """Test with no episode feature during fit or lift/retract."""
    lf, X, Xt = lift_retract_fixture
    lf.fit(X, n_inputs=1, episode_feature=False)
    # Test lift
    Xt_l = lf.lift(X, episode_feature=False)
    np.testing.assert_allclose(Xt_l, Xt)
    # Test retract
    X_r = lf.retract(Xt_l, episode_feature=False)
    np.testing.assert_allclose(X_r, X)
    # Test lift state
    Xt_l = lf.lift_state(X[:, :lf.n_states_in_], episode_feature=False)
    np.testing.assert_allclose(Xt_l, Xt[:, :lf.n_states_out_])
    # # Test retract state
    X_r = lf.retract_state(Xt_l, episode_feature=False)
    np.testing.assert_allclose(X_r, X[:, :lf.n_states_in_])
    # Test lift input
    Xt_l = lf.lift_input(X, episode_feature=False)
    np.testing.assert_allclose(Xt_l, Xt[:, lf.n_states_out_:])
    # Test retract input
    X_r = lf.retract_input(Xt_l, episode_feature=False)
    np.testing.assert_allclose(X_r, X[:, lf.n_states_in_:])


def test_lift_retract_tt(lift_retract_fixture):
    """Test with episode feature during fit and lift/retract."""
    lf, X_noep, Xt_noep = lift_retract_fixture
    # Add episode features
    X = np.hstack((
        np.array([0, 0, 0, 1, 1, 1]).reshape((-1, 1)),
        X_noep,
    ))
    Xt = np.hstack((
        np.array([0, 0, 0, 1, 1, 1]).reshape((-1, 1)),
        Xt_noep,
    ))
    lf.fit(X, n_inputs=1, episode_feature=True)
    # Test lift
    Xt_l = lf.lift(X, episode_feature=True)
    np.testing.assert_allclose(Xt_l, Xt)
    # Test retract
    X_r = lf.retract(Xt_l, episode_feature=True)
    np.testing.assert_allclose(X_r, X)
    # Test lift state
    Xt_l = lf.lift_state(X[:, :lf.n_states_in_ + 1], episode_feature=True)
    np.testing.assert_allclose(Xt_l, Xt[:, :lf.n_states_out_ + 1])
    # # Test retract state
    X_r = lf.retract_state(Xt_l, episode_feature=True)
    np.testing.assert_allclose(X_r, X[:, :lf.n_states_in_ + 1])
    # Test lift input
    Xt_l = lf.lift_input(X, episode_feature=True)
    np.testing.assert_allclose(Xt_l[:, 0], Xt[:, 0])
    np.testing.assert_allclose(
        Xt_l[:, 1:],
        Xt[:, lf.n_states_out_ + 1:],
    )
    # Test retract input
    X_r = lf.retract_input(Xt_l, episode_feature=True)
    np.testing.assert_allclose(X_r[:, 0], X[:, 0])
    np.testing.assert_allclose(X_r[:, 1:], X[:, lf.n_states_in_ + 1:])


def test_lift_retract_ft(lift_retract_fixture):
    """Test with episode feature during fit but not lift/retract."""
    lf, X_noep, Xt_noep = lift_retract_fixture
    # Add episode features
    X = np.hstack((
        np.array([0, 0, 0, 1, 1, 1]).reshape((-1, 1)),
        X_noep,
    ))
    Xt = np.hstack((
        np.array([0, 0, 0, 1, 1, 1]).reshape((-1, 1)),
        Xt_noep,
    ))
    lf.fit(X, n_inputs=1, episode_feature=True)
    # Test lift
    Xt_l = lf.lift(X_noep, episode_feature=False)
    np.testing.assert_allclose(Xt_l, Xt_noep)
    # Test retract
    X_r = lf.retract(Xt_l, episode_feature=False)
    np.testing.assert_allclose(X_r, X_noep)
    # Test lift state
    Xt_l = lf.lift_state(X_noep[:, :lf.n_states_in_], episode_feature=False)
    np.testing.assert_allclose(Xt_l, Xt_noep[:, :lf.n_states_out_])
    # # Test retract state
    X_r = lf.retract_state(Xt_l, episode_feature=False)
    np.testing.assert_allclose(X_r, X_noep[:, :lf.n_states_in_])
    # Test lift input
    Xt_l = lf.lift_input(X_noep, episode_feature=False)
    np.testing.assert_allclose(Xt_l, Xt_noep[:, lf.n_states_out_:])
    # Test retract input
    X_r = lf.retract_input(Xt_l, episode_feature=False)
    np.testing.assert_allclose(X_r, X_noep[:, lf.n_states_in_:])


def test_lift_retract_tf(lift_retract_fixture):
    """Test with no episode feature during fit but with one in lift/retract."""
    lf, X_noep, Xt_noep = lift_retract_fixture
    # Add episode features
    X = np.hstack((
        np.array([0, 0, 0, 1, 1, 1]).reshape((-1, 1)),
        X_noep,
    ))
    Xt = np.hstack((
        np.array([0, 0, 0, 1, 1, 1]).reshape((-1, 1)),
        Xt_noep,
    ))
    lf.fit(X_noep, n_inputs=1, episode_feature=False)
    # Test lift
    Xt_l = lf.lift(X, episode_feature=True)
    np.testing.assert_allclose(Xt_l, Xt)
    # Test retract
    X_r = lf.retract(Xt_l, episode_feature=True)
    np.testing.assert_allclose(X_r, X)
    # Test lift state
    Xt_l = lf.lift_state(X[:, :lf.n_states_in_ + 1], episode_feature=True)
    np.testing.assert_allclose(Xt_l, Xt[:, :lf.n_states_out_ + 1])
    # # Test retract state
    X_r = lf.retract_state(Xt_l, episode_feature=True)
    np.testing.assert_allclose(X_r, X[:, :lf.n_states_in_ + 1])
    # Test lift input
    Xt_l = lf.lift_input(X, episode_feature=True)
    np.testing.assert_allclose(Xt_l[:, 0], Xt[:, 0])
    np.testing.assert_allclose(
        Xt_l[:, 1:],
        Xt[:, lf.n_states_out_ + 1:],
    )
    # Test retract input
    X_r = lf.retract_input(Xt_l, episode_feature=True)
    np.testing.assert_allclose(X_r[:, 0], X[:, 0])
    np.testing.assert_allclose(X_r[:, 1:], X[:, lf.n_states_in_ + 1:])
