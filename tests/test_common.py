import sklearn.utils.estimator_checks
import pykoop.dmd
import pykoop.lmi


@sklearn.utils.estimator_checks.parametrize_with_checks([
    pykoop.dmd.EdmdRegressor(),
    pykoop.lmi.LmiKoopBaseRegressor(),
])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
