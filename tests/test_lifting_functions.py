"""Test ``lifting_functions`` module."""

import sklearn.utils.estimator_checks
from sklearn import preprocessing

import pykoop


class TestSklearn:
    """Test scikit-learn compatibility."""

    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.PolynomialLiftingFn(),
        pykoop.DelayLiftingFn(),
        pykoop.SkLearnLiftingFn(preprocessing.MaxAbsScaler()),
        pykoop.BilinearInputLiftingFn(),
    ])
    def test_compatible_estimator(self, estimator, check):
        """Test scikit-learn compatibility of estimators."""
        check(estimator)
