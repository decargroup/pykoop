"""Test :mod:`centers`."""

import sklearn.utils.estimator_checks

import pykoop


@pytest.mark.parametrize('est, X, centers', [
    (
        pykoop.GridCenters(),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).T,
        np.array([
            [],
            [],
        ]).T,
    ),
])
class TestGridCenters:
    """Test :class:`GridCenters`."""

    pass


class TestSkLearn:
    """Test scikit-learn compatibility."""

    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.GridCenters(),
    ])
    def test_compatible_estimator(self, estimator, check):
        """Test scikit-learn compatibility of estimators."""
        check(estimator)
