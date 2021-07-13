import numpy as np
import pytest
from dynamics import mass_spring_damper
from scipy import integrate, linalg

from pykoop import dmd, koopman_pipeline, lifting_functions


def test_kp_transform_no_lf():
    """Test Koopman pipeline transformer with no lifting functions."""
    # Create test matrix
    X = np.array([
        [1, 2, 3, 4, 5, 6],
        [-1, -2, -3, -4, -5, -6],
        [2, 4, 6, 8, 10, 12],
    ]).T
    # Create basic pipeline
    kp = koopman_pipeline.KoopmanPipeline(
        preprocessors=None,
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


def test_kp_transform_angle_pp():
    """Test Koopman pipeline transformer angle preprocessor."""
    # Create test matrix
    X = np.array([
        [1, 2, 3, 4, 5, 6],
        [0, (np.pi / 2), (np.pi / 3), (np.pi / 2), 0, (-np.pi / 2)],
        [2, 4, 6, 8, 10, 12],
    ]).T
    Xt_exp = np.array([
        [1, 2, 3, 4, 5, 6],
        [1, 0, 0.5, 0, 1, 0],
        [0, 1, (np.sqrt(3) / 2), 1, 0, -1],
        [2, 4, 6, 8, 10, 12],
    ]).T
    # Create basic pipeline
    kp = koopman_pipeline.KoopmanPipeline(
        preprocessors=[
            lifting_functions.AnglePreprocessor(angle_features=np.array([1]))
        ],
        lifting_functions=None,
        regressor=None,
    )
    # Fit pipeline
    kp.fit_transformers(X, n_inputs=1)
    # Check dimensions
    assert kp.n_features_in_ == 3
    assert kp.n_states_in_ == 2
    assert kp.n_inputs_in_ == 1
    assert kp.n_features_out_ == 4
    assert kp.n_states_out_ == 3
    assert kp.n_inputs_out_ == 1
    # Transform
    Xt = kp.transform(X)
    np.testing.assert_allclose(Xt, Xt_exp, atol=1e-16)
    # Inverse (preprocessor is not inverted)
    Xi = kp.inverse_transform(Xt)
    np.testing.assert_allclose(Xi, Xt_exp, atol=1e-16)


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
    # Create basic pipeline. Loop to test cases where preprocessor is ``None``,
    # and where preprocessor is a passthrough (``AnglePreprocessor`` with no
    # angles).
    preprocessors = [
        None,
        [lifting_functions.AnglePreprocessor(angle_features=None)],
    ]
    for pp in preprocessors:
        kp = koopman_pipeline.KoopmanPipeline(
            preprocessors=pp,
            lifting_functions=[
                lifting_functions.DelayLiftingFn(
                    n_delays_state=1,
                    n_delays_input=1,
                )
            ],
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
    msd = mass_spring_damper.MassSpringDamper(0.5, 0.7, 0.6)

    def u(t):
        return 0.1 * np.sin(t)

    def ivp(t, x):
        return msd.f(t, x, u(t))

    # Solve ODE for training data
    x0 = msd.x0(np.array([0, 0]))
    sol = integrate.solve_ivp(ivp,
                              t_range,
                              x0,
                              t_eval=np.arange(*t_range, t_step),
                              rtol=1e-8,
                              atol=1e-8)
    # Split the data
    X = np.vstack((
        np.zeros_like(sol.t),
        sol.y,
        u(sol.t),
    )).T
    kp = koopman_pipeline.KoopmanPipeline(
        preprocessors=[
            lifting_functions.AnglePreprocessor(angle_features=None)
        ],
        lifting_functions=[
            lifting_functions.DelayLiftingFn(
                n_delays_state=0,
                n_delays_input=0,
            )
        ],
        regressor=dmd.Edmd(),
    )
    kp.fit(X, n_inputs=1, episode_feature=True)
    # Compute discrete-time A and B matrices
    Ad = linalg.expm(msd._A * t_step)

    def integrand(s):
        return linalg.expm(msd._A * (t_step - s)).ravel()

    Bd = integrate.quad_vec(integrand, 0, t_step)[0].reshape((2, 2)) @ msd._B
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
    X_unsh, X_sh = koopman_pipeline._shift_episodes(X, n_inputs,
                                                    episode_feature)
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
    episodes_actual = koopman_pipeline._split_episodes(
        X, episode_feature=episode_feature)
    # Compare every episode
    for actual, expected in zip(episodes_actual, episodes):
        i_actual, X_actual = actual
        i_expected, X_expected = expected
        assert i_actual == i_expected
        np.testing.assert_allclose(X_actual, X_expected)


@pytest.mark.parametrize('X, episodes, episode_feature',
                         split_combine_episode_scenarios)
def test_combine_episodes(X, episodes, episode_feature):
    X_actual = koopman_pipeline._combine_episodes(
        episodes, episode_feature=episode_feature)
    np.testing.assert_allclose(X_actual, X)


@pytest.mark.parametrize('kp', [
    koopman_pipeline.KoopmanPipeline(
        preprocessors=None,
        lifting_functions=[
            lifting_functions.DelayLiftingFn(n_delays_state=1,
                                             n_delays_input=1)
        ],
        regressor=dmd.Edmd(),
    ),
    koopman_pipeline.KoopmanPipeline(
        preprocessors=None,
        lifting_functions=[
            lifting_functions.DelayLiftingFn(n_delays_state=2,
                                             n_delays_input=2),
            lifting_functions.DelayLiftingFn(n_delays_state=2,
                                             n_delays_input=2),
        ],
        regressor=dmd.Edmd(),
    ),
    koopman_pipeline.KoopmanPipeline(
        preprocessors=None,
        lifting_functions=[
            lifting_functions.DelayLiftingFn(n_delays_state=2,
                                             n_delays_input=1),
            lifting_functions.PolynomialLiftingFn(order=2),
            lifting_functions.DelayLiftingFn(n_delays_state=1,
                                             n_delays_input=2),
        ],
        regressor=dmd.Edmd(),
    ),
    koopman_pipeline.KoopmanPipeline(
        preprocessors=None,
        lifting_functions=[
            koopman_pipeline.SplitLiftingFn(
                lifting_functions_state=[
                    lifting_functions.PolynomialLiftingFn(order=2),
                ],
                lifting_functions_input=None,
            ),
            lifting_functions.DelayLiftingFn(n_delays_state=2,
                                             n_delays_input=2),
        ],
        regressor=dmd.Edmd(),
    ),
    koopman_pipeline.KoopmanPipeline(
        preprocessors=None,
        lifting_functions=[
            lifting_functions.DelayLiftingFn(n_delays_state=1,
                                             n_delays_input=1),
            koopman_pipeline.SplitLiftingFn(
                lifting_functions_state=[
                    lifting_functions.PolynomialLiftingFn(order=2),
                ],
                lifting_functions_input=[
                    lifting_functions.PolynomialLiftingFn(order=2),
                ],
            ),
            lifting_functions.DelayLiftingFn(n_delays_state=1,
                                             n_delays_input=1),
        ],
        regressor=dmd.Edmd(),
    ),
])
def test_multistep_prediction(kp):
    # Set up problem
    t_range = (0, 5)
    t_step = 0.1
    msd = mass_spring_damper.MassSpringDamper(mass=0.5,
                                              stiffness=0.7,
                                              damping=0.6)

    def u(t):
        return 0.1 * np.sin(t)

    def ivp(t, x):
        return msd.f(t, x, u(t))

    # Solve ODE for training data
    x0 = msd.x0(np.array([0, 0]))
    sol = integrate.solve_ivp(ivp,
                              t_range,
                              x0,
                              t_eval=np.arange(*t_range, t_step),
                              rtol=1e-8,
                              atol=1e-8)

    # Compute input at every training point
    u_sim = np.reshape(u(sol.t), (1, -1))
    # Format data
    X = np.vstack((
        np.zeros((1, sol.t.shape[0] - 1)),
        sol.y[:, :-1],
        u_sim[:, :-1],
    ))
    # Fit estimator
    kp.fit(X.T, n_inputs=1, episode_feature=True)
    n_samp = kp.min_samples_

    # Predict using ``predict_multistep``
    X_sim = np.empty(sol.y.shape)
    X_sim[:, :n_samp] = sol.y[:, :n_samp]
    X_sim = np.vstack((
        np.zeros_like(u_sim),
        X_sim,
        u_sim,
    ))
    X_sim = kp.predict_multistep(X_sim.T).T[1:, :]

    # Predict manually
    X_sim_exp = np.empty(sol.y.shape)
    X_sim_exp[:, :n_samp] = sol.y[:, :n_samp]
    for k in range(n_samp, sol.t.shape[0]):
        X = np.vstack((
            np.zeros((1, n_samp)),
            X_sim_exp[:, (k - n_samp):k],
            u_sim[:, (k - n_samp):k],
        ))
        Xp = kp.predict(X.T).T
        X_sim_exp[:, [k]] = Xp[1:, [-1]]

    np.testing.assert_allclose(X_sim, X_sim_exp)


test_split_lf_params = [
    # Basic, without episode feature
    (
        koopman_pipeline.SplitLiftingFn(
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
        koopman_pipeline.SplitLiftingFn(
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
        koopman_pipeline.SplitLiftingFn(
            lifting_functions_state=[
                lifting_functions.PolynomialLiftingFn(order=2)
            ],
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
        koopman_pipeline.SplitLiftingFn(
            lifting_functions_state=None,
            lifting_functions_input=[
                lifting_functions.PolynomialLiftingFn(order=2)
            ],
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
        koopman_pipeline.SplitLiftingFn(
            lifting_functions_state=[
                lifting_functions.PolynomialLiftingFn(order=2,
                                                      interaction_only=True)
            ],
            lifting_functions_input=[
                lifting_functions.PolynomialLiftingFn(order=2)
            ],
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
