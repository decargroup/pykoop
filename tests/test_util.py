"""Test :mod:`utils`."""

import numpy as np
import pytest
import sklearn.utils.estimator_checks

import pykoop


@pytest.mark.parametrize(
    'X, Xt_exp, Xi_exp, episode_feature',
    [
        # No episode feature
        (
            np.array([
                [0, 1, 2, 3],
                [0, np.pi, 0, -np.pi / 2],
                [-1, -2, -1, -2],
            ]).T,
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
            np.array([
                # Episodes
                [0, 0, 1, 1],
                # Data
                [0, 1, 2, 3],
                [0, np.pi, 0, -np.pi / 2],
                [-1, -2, -1, -2],
            ]).T,
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
            np.array([
                [0, 1, 2, 3],
                [2 * np.pi, np.pi, 0, -np.pi / 2],
                [-1, -2, -1, -2],
            ]).T,
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

    def test_transform(self, X, Xt_exp, Xi_exp, episode_feature):
        """Test :class:`AnglePreprocessor` transform."""
        pp = pykoop.AnglePreprocessor(angle_features=self.angle_feature)
        pp.fit(X, episode_feature=episode_feature)
        Xt = pp.transform(X)
        np.testing.assert_allclose(Xt_exp, Xt, atol=1e-15)

    def test_inverse_transform(self, X, Xt_exp, Xi_exp, episode_feature):
        """Test :class:`AnglePreprocessor` inverse transform."""
        pp = pykoop.AnglePreprocessor(angle_features=self.angle_feature)
        pp.fit(X, episode_feature=episode_feature)
        Xt = pp.transform(X)
        Xi = pp.inverse_transform(Xt)
        np.testing.assert_allclose(Xi_exp, Xi, atol=1e-15)


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


class TestSklearn:
    """Test scikit-learn compatibility."""

    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.AnglePreprocessor(),
    ])
    def test_compatible_estimator(self, estimator, check):
        """Test scikit-learn compatibility of estimators."""
        check(estimator)