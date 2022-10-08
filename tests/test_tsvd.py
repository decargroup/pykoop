"""Test :mod:`tsvd`."""

import sklearn.utils.estimator_checks

import pykoop


class TestSkLearn:
    """Test ``scikit-learn`` compatibility."""

    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.Tsvd(),
    ])
    def test_compatible_estimator(self, estimator, check):
        """Test ``scikit-learn`` compatibility of estimators."""
        check(estimator)
