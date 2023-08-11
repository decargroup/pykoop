"""Test :mod:`centers`."""

import numpy as np
import pytest
import sklearn.cluster
import sklearn.mixture
import sklearn.utils.estimator_checks
from scipy import stats

import pykoop


@pytest.mark.parametrize('est, X, centers', [
    (
        pykoop.GridCenters(),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
        np.array([
            [1, 3, 1, 3],
            [4, 4, 6, 6],
        ]).T,
    ),
    (
        pykoop.GridCenters(n_points_per_feature=3),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
        np.array([
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
            [4, 4, 4, 5, 5, 5, 6, 6, 6],
        ]).T,
    ),
    (
        pykoop.GridCenters(symmetric_range=True),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
        np.array([
            [-3, 3, -3, 3],
            [-6, -6, 6, 6],
        ]).T,
    ),
    (
        pykoop.GridCenters(n_points_per_feature=3, symmetric_range=True),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
        np.array([
            [-3, 0, 3, -3, 0, 3, -3, 0, 3],
            [-6, -6, -6, 0, 0, 0, 6, 6, 6],
        ]).T,
    ),
])
class TestGridCenters:
    """Test :class:`GridCenters`."""

    def test_grid_centers(self, est, X, centers):
        """Test center locations."""
        est.fit(X)
        np.testing.assert_allclose(est.centers_, centers)

    def test_n_centers(self, est, X, centers):
        """Test number of centers."""
        est.fit(X)
        assert est.n_centers_ == centers.shape[0]

    def test_range_max(self, est, X, centers):
        """Test maximum of range."""
        est.fit(X)
        max_exp = np.max(est.centers_, axis=0)
        np.testing.assert_allclose(est.range_max_, max_exp)

    def test_range_min(self, est, X, centers):
        """Test minimum of range."""
        est.fit(X)
        min_exp = np.min(est.centers_, axis=0)
        np.testing.assert_allclose(est.range_min_, min_exp)


@pytest.mark.parametrize('est, X', [
    (
        pykoop.UniformRandomCenters(random_state=1234),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
    (
        pykoop.UniformRandomCenters(n_centers=200, random_state=1234),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
    (
        pykoop.UniformRandomCenters(symmetric_range=True, random_state=1234),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
])
class TestUniformRandomCenters:
    """Test :class:`UniformRandomCenters`.

    Attributes
    ----------
    tol : float
        Tolerance for regression test.
    """

    tol = 1e-6

    def test_uniform_centers(self, ndarrays_regression, est, X):
        """Test center locations."""
        est.fit(X)
        idx = np.argsort(est.centers_[:, 0])
        ndarrays_regression.check(
            {
                'est.centers_': est.centers_[idx, :],
            },
            default_tolerance=dict(atol=self.tol, rtol=0),
        )

    def test_n_centers(self, est, X):
        """Test number of centers."""
        est.fit(X)
        assert est.n_centers_ == est.centers_.shape[0]

    def test_range_max(self, est, X):
        """Test maximum of range."""
        est.fit(X)
        max_exp = np.max(est.centers_, axis=0)
        assert np.all(est.range_max_ - max_exp > 0)

    def test_range_min(self, est, X):
        """Test minimum of range."""
        est.fit(X)
        min_exp = np.min(est.centers_, axis=0)
        assert np.all(min_exp - est.range_min_ > 0)


@pytest.mark.parametrize('est, X', [
    (
        pykoop.GaussianRandomCenters(random_state=1234),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
    (
        pykoop.GaussianRandomCenters(n_centers=200, random_state=1234),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
])
class TestGaussianRandomCenters:
    """Test :class:`GaussianRandomCenters`.

    Attributes
    ----------
    tol : float
        Tolerance for regression test.
    """

    tol = 1e-6

    def test_gaussian_centers(self, ndarrays_regression, est, X):
        """Test center locations."""
        est.fit(X)
        idx = np.argsort(est.centers_[:, 0])
        ndarrays_regression.check(
            {
                'est.centers_': est.centers_[idx, :],
            },
            default_tolerance=dict(atol=self.tol, rtol=0),
        )

    def test_n_centers(self, est, X):
        """Test number of centers."""
        est.fit(X)
        assert est.n_centers_ == est.centers_.shape[0]


@pytest.mark.parametrize('est, X', [
    (
        pykoop.QmcCenters(random_state=1234),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
    (
        pykoop.QmcCenters(n_centers=200, random_state=1234),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
    (
        pykoop.QmcCenters(symmetric_range=True, random_state=1234),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
    (
        pykoop.QmcCenters(
            n_centers=128,
            qmc=stats.qmc.Sobol,
            random_state=1234,
        ),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
    (
        pykoop.QmcCenters(
            n_centers=8,
            qmc=stats.qmc.Sobol,
            qmc_kw=dict(scramble=False),
            random_state=1234,
        ),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
])
class TestQmcCenters:
    """Test :class:`QmcCenters`.

    Attributes
    ----------
    tol : float
        Tolerance for regression test.
    """

    tol = 1e-6

    def test_qmc_centers(self, ndarrays_regression, est, X):
        """Test center locations."""
        est.fit(X)
        idx = np.argsort(est.centers_[:, 0])
        ndarrays_regression.check(
            {'est.centers_': est.centers_[idx, :]},
            default_tolerance=dict(atol=self.tol, rtol=0),
        )

    def test_n_centers(self, est, X):
        """Test number of centers."""
        est.fit(X)
        assert est.n_centers_ == est.centers_.shape[0]

    def test_range_max(self, est, X):
        """Test maximum of range."""
        est.fit(X)
        max_exp = np.max(est.centers_, axis=0)
        assert np.all(est.range_max_ - max_exp > 0)

    def test_range_min(self, est, X):
        """Test minimum of range."""
        est.fit(X)
        min_exp = np.min(est.centers_, axis=0)
        assert np.all(min_exp - est.range_min_ >= 0)


@pytest.mark.parametrize('est, X', [
    (
        pykoop.ClusterCenters(estimator=sklearn.cluster.KMeans(
            n_clusters=1,
            n_init=10,
            random_state=1234,
        )),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
    (
        pykoop.ClusterCenters(estimator=sklearn.cluster.KMeans(
            n_clusters=2,
            n_init=10,
            random_state=1234,
        )),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
])
class TestClusterCenters:
    """Test :class:`ClusterCenters`.

    Attributes
    ----------
    tol : float
        Tolerance for regression test.
    """

    tol = 1e-3

    def test_cluster_centers(self, ndarrays_regression, est, X):
        """Test center locations."""
        est.fit(X)
        idx = np.argsort(est.centers_[:, 0])
        ndarrays_regression.check(
            {
                'est.centers_': est.centers_[idx, :],
            },
            default_tolerance=dict(atol=self.tol, rtol=0),
        )

    def test_n_centers(self, est, X):
        """Test number of centers."""
        est.fit(X)
        assert est.n_centers_ == est.centers_.shape[0]


@pytest.mark.parametrize('est, X', [
    (
        pykoop.GaussianMixtureRandomCenters(
            estimator=sklearn.mixture.GaussianMixture(
                n_components=2,
                tol=1e-6,
                random_state=1234,
            )),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
    (
        pykoop.GaussianMixtureRandomCenters(
            estimator=sklearn.mixture.BayesianGaussianMixture(
                tol=1e-6,
                random_state=1234,
            )),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
])
class TestGaussianMixtureRandomCenters:
    """Test :class:`GaussianMixtureRandomCenters`."""

    def test_n_centers(self, est, X):
        """Test number of centers."""
        est.fit(X)
        assert est.n_centers_ == est.centers_.shape[0]


@pytest.mark.parametrize(
    'est, X, centers',
    [
        # Latch on fit
        (
            pykoop.DataCenters(centers=None),
            np.array([
                [1, 2, 3],
                [4, 5, 6],
            ]).T,
            np.array([
                [1, 2, 3],
                [4, 5, 6],
            ]).T,
        ),
        # Latch on instantiation
        (
            pykoop.DataCenters(centers=np.array([
                [1, 2, 3],
                [4, 5, 6],
            ]).T),
            np.array([
                [5, 5, 5],
                [5, 5, 5],
            ]).T,
            np.array([
                [1, 2, 3],
                [4, 5, 6],
            ]).T,
        ),
    ])
class TestDataCenters:
    """Test :class:`DataCenters`."""

    def test_data_centers(self, est, X, centers):
        """Test center locations."""
        est.fit(X)
        np.testing.assert_allclose(est.centers_, centers)

    def test_n_centers(self, est, X, centers):
        """Test number of centers."""
        est.fit(X)
        assert est.n_centers_ == centers.shape[0]


class TestSkLearn:
    """Test ``scikit-learn`` compatibility."""

    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.GridCenters(),
        pykoop.UniformRandomCenters(),
        pykoop.GaussianRandomCenters(),
        pykoop.QmcCenters(),
        pykoop.ClusterCenters(),
        pykoop.GaussianMixtureRandomCenters(),
        pykoop.DataCenters(),
    ])
    def test_compatible_estimator(self, estimator, check):
        """Test ``scikit-learn`` compatibility of estimators."""
        check(estimator)
