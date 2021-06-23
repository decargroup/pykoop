import numpy as np
import pytest

from pykoop import koopman_pipeline, lifting_functions


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
        estimator=None,
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
        estimator=None,
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
            estimator=None,
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
