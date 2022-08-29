"""Test ``regressors`` module."""

import numpy as np
import pytest
import sklearn.utils.estimator_checks

import pykoop


class TestSklearn:
    """Test scikit-learn compatibility."""

    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.Edmd(),
        pykoop.Dmdc(),
        pykoop.Dmd(),
    ])
    def test_compatible_estimator(self, estimator, check):
        """Test scikit-learn compatibility of estimators."""
        check(estimator)


class TestExceptions:
    """Test a selection of invalid estimator parameter."""

    X = np.array([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
    ])

    @pytest.mark.parametrize('estimator', [
        pykoop.Edmd(alpha=-1),
        pykoop.Dmdc(mode_type='blah'),
        pykoop.Dmd(mode_type='blah'),
    ])
    def test_invalid_params(self, estimator):
        """Test a selection of invalid estimator parameter."""
        with pytest.raises(ValueError):
            estimator.fit(self.X)
