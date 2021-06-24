import numpy as np
import pytest
from dynamics import mass_spring_damper
from pykoop import dmd, koopman_pipeline, lifting_functions
from scipy import integrate, linalg


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
        preprocessors=[(
            'angle',
            lifting_functions.AnglePreprocessor(angle_features=np.array([1])),
        )],
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
        [(
            'angle',
            lifting_functions.AnglePreprocessor(angle_features=None),
        )],
    ]
    for pp in preprocessors:
        kp = koopman_pipeline.KoopmanPipeline(
            preprocessors=pp,
            lifting_functions=[(
                'delay',
                lifting_functions.DelayLiftingFn(
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
        preprocessors=[(
            'passthrough-angles',
            lifting_functions.AnglePreprocessor(angle_features=None),
        )],
        lifting_functions=[(
            'passthrough-delay',
            lifting_functions.DelayLiftingFn(
                n_delays_state=0,
                n_delays_input=0,
            ),
        )],
        regressor=('edmd', dmd.Edmd()),
    )
    kp.fit(X, n_inputs=1, episode_feature=True)
    # Compute discrete-time A and B matrices
    Ad = linalg.expm(msd._A * t_step)
    integrand = lambda s: linalg.expm(msd._A * (t_step - s)).ravel()
    Bd = integrate.quad_vec(integrand, 0, t_step)[0].reshape((2, 2)) @ msd._B
    U_exp = np.hstack((Ad, Bd)).T
    np.testing.assert_allclose(kp.regressor_[1].coef_, U_exp, atol=0.1)


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
                [1, 2, 4],
                [-1, -2, -4],
            ]).T,
            np.array([
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
                [1, 3, 4],
                [-1, -3, -4],
            ]).T,
            np.array([
                [2, 4, 5],
            ]).T,
            1,
            True,
        ),
    ],
)
def test_shift_episodes(X, X_unsh_exp, X_sh_exp, n_inputs, episode_feature):
    """Test episode shifting."""
    X_unsh, X_sh = koopman_pipeline.shift_episodes(X, n_inputs,
                                                   episode_feature)
    np.testing.assert_allclose(X_unsh, X_unsh_exp)
    np.testing.assert_allclose(X_sh, X_sh_exp)
