import numpy as np
import pytest
import sklearn.utils.estimator_checks

import pykoop


class TestAnglePreprocessor:

    def test_no_episodes(self):
        """Test angle preprocessing without an episode feature."""
        ang = np.array([1])
        pp = pykoop.AnglePreprocessor(angle_features=ang)
        X = np.array([
            [0, 1, 2, 3],
            [0, np.pi, 0, -np.pi / 2],
            [-1, -2, -1, -2],
        ]).T
        Xt_exp = np.array([
            [0, 1, 2, 3],
            [1, -1, 1, 0],
            [0, 0, 0, -1],
            [-1, -2, -1, -2],
        ]).T
        pp.fit(X, episode_feature=False)
        Xt = pp.transform(X)
        np.testing.assert_allclose(Xt_exp, Xt, atol=1e-15)
        Xi = pp.inverse_transform(Xt)
        np.testing.assert_allclose(X, Xi)

    def test_episodes(self):
        """Test angle preprocessing with an episode feature."""
        ang = np.array([1])
        pp = pykoop.AnglePreprocessor(angle_features=ang)
        X = np.array([
            # Episodes
            [0, 0, 1, 1],
            # Data
            [0, 1, 2, 3],
            [0, np.pi, 0, -np.pi / 2],
            [-1, -2, -1, -2],
        ]).T
        Xt_exp = np.array([
            # Episodes
            [0, 0, 1, 1],
            # Data
            [0, 1, 2, 3],
            [1, -1, 1, 0],
            [0, 0, 0, -1],
            [-1, -2, -1, -2],
        ]).T
        pp.fit(X, episode_feature=True)
        Xt = pp.transform(X)
        np.testing.assert_allclose(Xt_exp, Xt, atol=1e-15)
        Xi = pp.inverse_transform(Xt)
        np.testing.assert_allclose(X, Xi)

    def test_angle_wraparound(self):
        """Test angle preprocessing with wraparound."""
        ang = np.array([1])
        pp = pykoop.AnglePreprocessor(angle_features=ang)
        X = np.array([
            [0, 1, 2, 3],
            [2 * np.pi, np.pi, 0, -np.pi / 2],
            [-1, -2, -1, -2],
        ]).T
        Xt_exp = np.array([
            [0, 1, 2, 3],
            [1, -1, 1, 0],
            [0, 0, 0, -1],
            [-1, -2, -1, -2],
        ]).T
        Xi_exp = np.array([
            [0, 1, 2, 3],
            [0, np.pi, 0, -np.pi / 2],
            [-1, -2, -1, -2],
        ]).T
        pp.fit(X, episode_feature=False)
        Xt = pp.transform(X)
        np.testing.assert_allclose(Xt_exp, Xt, atol=1e-15)
        Xi = pp.inverse_transform(Xt)
        np.testing.assert_allclose(Xi_exp, Xi, atol=1e-15)

    def test_features_ordering(self):
        """Test angle preprocessing feature order."""
        X = np.zeros((2, 5))
        # Mix of linear and angles
        ang = np.array([2, 3])
        pp = pykoop.AnglePreprocessor(angle_features=ang)
        pp.fit(X, episode_feature=False)
        lin = np.array([1, 1, 0, 0, 0, 0, 1], dtype=bool)
        cos = np.array([0, 0, 1, 0, 1, 0, 0], dtype=bool)
        sin = np.array([0, 0, 0, 1, 0, 1, 0], dtype=bool)
        np.testing.assert_allclose(lin, pp.lin_out_)
        np.testing.assert_allclose(cos, pp.cos_out_)
        np.testing.assert_allclose(sin, pp.sin_out_)
        # All linear
        ang = np.array([])
        pp = pykoop.AnglePreprocessor(angle_features=ang)
        pp.fit(X, episode_feature=False)
        lin = np.array([1, 1, 1, 1, 1], dtype=bool)
        cos = np.array([0, 0, 0, 0, 0], dtype=bool)
        sin = np.array([0, 0, 0, 0, 0], dtype=bool)
        np.testing.assert_allclose(lin, pp.lin_out_)
        np.testing.assert_allclose(cos, pp.cos_out_)
        np.testing.assert_allclose(sin, pp.sin_out_)
        # All angles
        ang = np.array([0, 1, 2, 3, 4])
        pp = pykoop.AnglePreprocessor(angle_features=ang)
        pp.fit(X, episode_feature=False)
        lin = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        cos = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
        sin = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
        np.testing.assert_allclose(lin, pp.lin_out_)
        np.testing.assert_allclose(cos, pp.cos_out_)
        np.testing.assert_allclose(sin, pp.sin_out_)


class TestSklearn:
    """Test scikit-learn compatibility."""

    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.AnglePreprocessor(),
    ])
    def test_compatible_estimator(self, estimator, check):
        """Test scikit-learn compatibility of estimators."""
        check(estimator)
