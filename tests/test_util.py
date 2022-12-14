"""Test :mod:`utils`."""

import numpy as np
import pytest
import sklearn.utils.estimator_checks

import pykoop


@pytest.mark.parametrize(
    'names_in, X, names_out, Xt_exp, Xi_exp, episode_feature',
    [
        # No episode feature
        (
            np.array(['x0', 'x1', 'x2']),
            np.array([
                [0, 1, 2, 3],
                [0, np.pi, 0, -np.pi / 2],
                [-1, -2, -1, -2],
            ]).T,
            np.array(['x0', 'cos(x1)', 'sin(x1)', 'x2']),
            np.array([
                [0, 1, 2, 3],
                [1, -1, 1, 0],
                [0, 0, 0, -1],
                [-1, -2, -1, -2],
            ]).T,
            np.array([
                [0, 1, 2, 3],
                [0, np.pi, 0, -np.pi / 2],
                [-1, -2, -1, -2],
            ]).T,
            False,
        ),
        # Epsisode feature
        (
            np.array(['ep', 'x0', 'x1', 'x2']),
            np.array([
                # Episodes
                [0, 0, 1, 1],
                # Data
                [0, 1, 2, 3],
                [0, np.pi, 0, -np.pi / 2],
                [-1, -2, -1, -2],
            ]).T,
            np.array(['ep', 'x0', 'cos(x1)', 'sin(x1)', 'x2']),
            np.array([
                # Episodes
                [0, 0, 1, 1],
                # Data
                [0, 1, 2, 3],
                [1, -1, 1, 0],
                [0, 0, 0, -1],
                [-1, -2, -1, -2],
            ]).T,
            np.array([
                # Episodes
                [0, 0, 1, 1],
                # Data
                [0, 1, 2, 3],
                [0, np.pi, 0, -np.pi / 2],
                [-1, -2, -1, -2],
            ]).T,
            True,
        ),
        # Angle wraparound
        (
            np.array(['x0', 'x1', 'x2']),
            np.array([
                [0, 1, 2, 3],
                [2 * np.pi, np.pi, 0, -np.pi / 2],
                [-1, -2, -1, -2],
            ]).T,
            np.array(['x0', 'cos(x1)', 'sin(x1)', 'x2']),
            np.array([
                [0, 1, 2, 3],
                [1, -1, 1, 0],
                [0, 0, 0, -1],
                [-1, -2, -1, -2],
            ]).T,
            np.array([
                [0, 1, 2, 3],
                [0, np.pi, 0, -np.pi / 2],
                [-1, -2, -1, -2],
            ]).T,
            False,
        ),
    ],
)
class TestAnglePreprocessorTransform:
    """Test :class:`AnglePreprocessor` transform and inverse transform.

    Attributes
    ----------
    angle_feature : np.ndarray
        Array of feature indices that are angles.
    """

    angle_feature = np.array([1])

    def test_transform(self, names_in, X, names_out, Xt_exp, Xi_exp,
                       episode_feature):
        """Test :class:`AnglePreprocessor` transform."""
        pp = pykoop.AnglePreprocessor(angle_features=self.angle_feature)
        pp.fit(X, episode_feature=episode_feature)
        Xt = pp.transform(X)
        np.testing.assert_allclose(Xt_exp, Xt, atol=1e-15)

    def test_inverse_transform(self, names_in, X, names_out, Xt_exp, Xi_exp,
                               episode_feature):
        """Test :class:`AnglePreprocessor` inverse transform."""
        pp = pykoop.AnglePreprocessor(angle_features=self.angle_feature)
        pp.fit(X, episode_feature=episode_feature)
        Xt = pp.transform(X)
        Xi = pp.inverse_transform(Xt)
        np.testing.assert_allclose(Xi_exp, Xi, atol=1e-15)

    def test_feature_names_in(self, names_in, X, names_out, Xt_exp, Xi_exp,
                              episode_feature):
        """Test input feature names."""
        pp = pykoop.AnglePreprocessor(angle_features=self.angle_feature)
        pp.fit(X, episode_feature=episode_feature)
        names_in_actual = pp.get_feature_names_in()
        assert all(names_in == names_in_actual)
        assert names_in_actual.dtype == object

    def test_feature_names_out(self, names_in, X, names_out, Xt_exp, Xi_exp,
                               episode_feature):
        """Test input feature names."""
        pp = pykoop.AnglePreprocessor(angle_features=self.angle_feature)
        pp.fit(X, episode_feature=episode_feature)
        names_out_actual = pp.get_feature_names_out()
        assert all(names_out == names_out_actual)
        assert names_out_actual.dtype == object


@pytest.mark.parametrize(
    'angle_feature, lin, cos, sin',
    [
        # Mix of linear and angles
        (
            np.array([2, 3]),
            np.array([1, 1, 0, 0, 0, 0, 1], dtype=bool),
            np.array([0, 0, 1, 0, 1, 0, 0], dtype=bool),
            np.array([0, 0, 0, 1, 0, 1, 0], dtype=bool),
        ),
        # All linear
        (
            np.array([]),
            np.array([1, 1, 1, 1, 1], dtype=bool),
            np.array([0, 0, 0, 0, 0], dtype=bool),
            np.array([0, 0, 0, 0, 0], dtype=bool),
        ),
        # All angles
        (
            np.array([0, 1, 2, 3, 4]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool),
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool),
        ),
    ])
class TestAnglePreprocessorFeatureOrder:
    """Test :class:`AnglePreprocessor` feature order."""

    X = np.zeros((2, 5))

    def test_order_lin(self, angle_feature, lin, cos, sin):
        """Test :class:`AnglePreprocessor` linear feature order."""
        pp = pykoop.AnglePreprocessor(angle_features=angle_feature)
        pp.fit(self.X, episode_feature=False)
        np.testing.assert_allclose(lin, pp.lin_out_)

    def test_order_cos(self, angle_feature, lin, cos, sin):
        """Test :class:`AnglePreprocessor` cosine feature order."""
        pp = pykoop.AnglePreprocessor(angle_features=angle_feature)
        pp.fit(self.X, episode_feature=False)
        np.testing.assert_allclose(cos, pp.cos_out_)

    def test_order_sin(self, angle_feature, lin, cos, sin):
        """Test :class:`AnglePreprocessor` sine feature order."""
        pp = pykoop.AnglePreprocessor(angle_features=angle_feature)
        pp.fit(self.X, episode_feature=False)
        np.testing.assert_allclose(sin, pp.sin_out_)


@pytest.mark.parametrize(
    'names_in, X, names_out, n_inputs, episode_feature',
    [
        (
            np.array(['x_{0}', 'x_{1}', 'x_{2}']),
            np.array([
                [0, 1, 2, 3],
                [0, np.pi, 0, -np.pi / 2],
                [-1, -2, -1, -2],
            ]).T,
            np.array(['x_{0}', r'\cos{(x_{1})}', r'\sin{(x_{1})}', 'x_{2}']),
            0,
            False,
        ),
    ],
)
class TestLiftingFnLatexFeatureNames:
    """Test lifting function LaTeX feature names.

    Attributes
    ----------
    angle_feature : np.ndarray
        Array of feature indices that are angles.
    """

    angle_feature = np.array([1])

    def test_feature_names_in(self, names_in, X, names_out, n_inputs,
                              episode_feature):
        """Test input feature names."""
        pp = pykoop.AnglePreprocessor(angle_features=self.angle_feature)
        pp.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        names_in_actual = pp.get_feature_names_in(format='latex')
        assert np.all(names_in == names_in_actual)
        assert names_in_actual.dtype == object

    def test_feature_names_out(self, names_in, X, names_out, n_inputs,
                               episode_feature):
        """Test input feature names."""
        pp = pykoop.AnglePreprocessor(angle_features=self.angle_feature)
        pp.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        names_out_actual = pp.get_feature_names_out(format='latex')
        assert np.all(names_out == names_out_actual)
        assert names_out_actual.dtype == object


@pytest.mark.parametrize('fn', [
    pykoop.example_data_msd,
    pykoop.example_data_vdp,
    pykoop.example_data_pendulum,
    pykoop.example_data_duffing,
])
class TestExampleData:
    """Test example dynamic model data.

    Attributes
    ----------
    tol : float
        Tolerance for regression test.
    """

    tol = 1e-6

    def test_example_data(self, ndarrays_regression, fn):
        """Test example dynamic model data."""
        data = fn()
        ndarrays_regression.check(
            {
                'X_train': data['X_train'],
                'X_valid': data['X_valid'],
                'x0_valid': data['x0_valid'],
                'u_valid': data['u_valid'],
                'n_inputs': data['n_inputs'],
                'episode_feature': data['episode_feature'],
                't': data['t'],
            },
            default_tolerance=dict(atol=self.tol, rtol=0),
        )


class TestSkLearn:
    """Test ``scikit-learn`` compatibility."""

    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.AnglePreprocessor(),
    ])
    def test_compatible_estimator(self, estimator, check):
        """Test ``scikit-learn`` compatibility of estimators."""
        check(estimator)
