"""Test :mod:`kernel_approximation`."""

import numpy as np
import pytest
import sklearn.kernel_approximation
import sklearn.utils.estimator_checks

import pykoop


@pytest.mark.parametrize('est, kern, X', [
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='gaussian',
            n_components=int(1e4),
            shape=1,
            method='weight_only',
            random_state=1234,
        ),
        lambda x, y: np.exp(-(x - y).T @ (x - y)),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='gaussian',
            n_components=int(1e4),
            shape=3,
            method='weight_only',
            random_state=1234,
        ),
        lambda x, y: np.exp(3 * -(x - y).T @ (x - y)),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='gaussian',
            n_components=int(1e4),
            shape=1,
            method='weight_offset',
            random_state=1234,
        ),
        lambda x, y: np.exp(-(x - y).T @ (x - y)),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='laplacian',
            n_components=int(1e5),
            shape=1,
            method='weight_only',
            random_state=1234,
        ),
        lambda x, y: np.prod(np.exp(-np.sqrt(2) * np.abs(x - y))),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='laplacian',
            n_components=int(1e5),
            shape=3,
            method='weight_only',
            random_state=1234,
        ),
        lambda x, y: np.prod(np.exp(-np.sqrt(2 * 3) * np.abs(x - y))),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='laplacian',
            n_components=int(1e5),
            shape=1,
            method='weight_offset',
            random_state=1234,
        ),
        lambda x, y: np.prod(np.exp(-np.sqrt(2) * np.abs(x - y))),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='cauchy',
            n_components=int(1e5),
            shape=1,
            method='weight_only',
            random_state=1234,
        ),
        lambda x, y: np.prod(1 / (1 + 2 * (x - y)**2)),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='cauchy',
            n_components=int(1e5),
            shape=3,
            method='weight_only',
            random_state=1234,
        ),
        lambda x, y: np.prod(1 / (1 + 2 * 3 * (x - y)**2)),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='cauchy',
            n_components=int(1e6),
            shape=1,
            method='weight_offset',
            random_state=1234,
        ),
        lambda x, y: np.prod(1 / (1 + 2 * (x - y)**2)),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
    (
        pykoop.RandomBinningKernelApprox(
            kernel_or_ddot='laplacian',
            n_components=int(1e4),
            shape=1,
            random_state=1234,
        ),
        lambda x, y: np.prod(np.exp(-np.sqrt(2) * np.abs(x - y))),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
    (
        pykoop.RandomBinningKernelApprox(
            kernel_or_ddot='laplacian',
            n_components=int(1e4),
            shape=3,
            random_state=1234,
        ),
        lambda x, y: np.prod(np.exp(-np.sqrt(2 * 3) * np.abs(x - y))),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
])
class TestKernelApproximation:
    """Test kernel approximations against their corresponding kernels.

    Attributes
    ----------
    tol : float
        Tolerance for regression test.
    """

    tol = 1e-2

    def test_kernel_approximation(self, est, kern, X):
        """Test kernel approximations against kernel."""
        # Compute kernel
        K_true = np.zeros((X.shape[0], X.shape[0]))
        for i in range(K_true.shape[0]):
            for j in range(K_true.shape[1]):
                K_true[i, j] = kern(X[i, :], X[j, :])
        # Estimate kernel
        est.fit(X)
        Xt = est.transform(X)
        K_est = Xt @ Xt.T
        np.testing.assert_allclose(K_est, K_true, atol=self.tol, rtol=0)


@pytest.mark.parametrize('est, est_sk, X', [
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='gaussian',
            n_components=int(1e4),
            shape=1,
            method='weight_only',
            random_state=1234,
        ),
        sklearn.kernel_approximation.RBFSampler(
            gamma=1,
            n_components=int(2e4),
            random_state=1234,
        ),
        np.array([
            [0.01, 0.02, 0.03],
            [0.04, 0.05, 0.06],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='gaussian',
            n_components=int(1e4),
            shape=1,
            method='weight_offset',
            random_state=1234,
        ),
        sklearn.kernel_approximation.RBFSampler(
            gamma=1,
            n_components=int(1e4),
            random_state=1234,
        ),
        np.array([
            [0.01, 0.02, 0.03],
            [0.04, 0.05, 0.06],
        ]).T,
    ),
])
class TestKernelApproximationSklearn:
    """Test kernel approximations against ``scikit-learn`` implementation.

    Attributes
    ----------
    tol : float
        Tolerance for regression test.
    """

    tol = 1e-2

    def test_kernel_approximation(self, est, est_sk, X):
        """Test kernel approximations against ``scikit-learn``."""
        # Estimate kernel
        est.fit(X)
        Xt = est.transform(X)
        K_est = Xt @ Xt.T
        # Estimate kernel with ``scikit-learn``
        est_sk.fit(X)
        Xt_sk = est_sk.transform(X)
        K_sk = Xt_sk @ Xt_sk.T
        # Compare results
        np.testing.assert_allclose(K_est, K_sk, atol=self.tol, rtol=0)


@pytest.mark.parametrize('est, X', [
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='gaussian',
            n_components=int(1e4),
            shape=1,
            method='weight_only',
            random_state=1234,
        ),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='gaussian',
            n_components=int(1e4),
            shape=3,
            method='weight_only',
            random_state=1234,
        ),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='gaussian',
            n_components=int(1e4),
            shape=1,
            method='weight_offset',
            random_state=1234,
        ),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='laplacian',
            n_components=int(1e5),
            shape=1,
            method='weight_only',
            random_state=1234,
        ),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='laplacian',
            n_components=int(1e5),
            shape=3,
            method='weight_only',
            random_state=1234,
        ),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='laplacian',
            n_components=int(1e5),
            shape=1,
            method='weight_offset',
            random_state=1234,
        ),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='cauchy',
            n_components=int(1e5),
            shape=1,
            method='weight_only',
            random_state=1234,
        ),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='cauchy',
            n_components=int(1e5),
            shape=3,
            method='weight_only',
            random_state=1234,
        ),
        np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]).T,
    ),
    (
        pykoop.RandomFourierKernelApprox(
            kernel_or_ift='cauchy',
            n_components=int(1e6),
            shape=1,
            method='weight_offset',
            random_state=1234,
        ),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
    (
        pykoop.RandomBinningKernelApprox(
            kernel_or_ddot='laplacian',
            n_components=int(1e4),
            shape=1,
            random_state=1234,
        ),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
    (
        pykoop.RandomBinningKernelApprox(
            kernel_or_ddot='laplacian',
            n_components=int(1e4),
            shape=3,
            random_state=1234,
        ),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
    ),
])
class TestKernelApproximationRegression:
    """Regression test kernel approximations.

    Attributes
    ----------
    tol : float
        Tolerance for regression test.
    """

    tol = 1e-6

    def test_kernel_approximation(self, ndarrays_regression, est, X):
        """Regression test kernel approximations."""
        est.fit(X)
        Xt = est.transform(X)
        ndarrays_regression.check(
            {
                'Xt': Xt,
            },
            default_tolerance=dict(atol=self.tol, rtol=0),
        )


class TestSkLearn:
    """Test ``scikit-learn`` compatibility."""

    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.RandomFourierKernelApprox(
            method='weight_offset',
            random_state=1234,
        ),
        pykoop.RandomFourierKernelApprox(
            method='weight_only',
            random_state=1234,
        ),
        pykoop.RandomBinningKernelApprox(random_state=1234),
    ])
    def test_compatible_estimator(self, estimator, check):
        """Test ``scikit-learn`` compatibility of estimators."""
        check(estimator)
