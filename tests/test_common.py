import sklearn.utils.estimator_checks
import pykoop.dmd


@sklearn.utils.estimator_checks.parametrize_with_checks(
    [pykoop.dmd.EdmdRegressor()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
