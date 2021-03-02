import sklearn.utils.estimator_checks
import pykoop.dmd
import pykoop.lmi
import pykoop.lifting_functions


@sklearn.utils.estimator_checks.parametrize_with_checks([
    pykoop.dmd.Edmd(),
    pykoop.lmi.LmiEdmd(),
    pykoop.lmi.LmiEdmdTikhonovReg(),
    pykoop.lmi.LmiEdmdTwoNormReg(),
    pykoop.lmi.LmiEdmdNuclearNormReg(),
    pykoop.lmi.LmiEdmdSpectralRadiusConstr(tol=100),  # Loosen tolerance
    pykoop.lmi.LmiEdmdHinfReg(tol=100),  # Loosen tolerance
    pykoop.lifting_functions.Delay(),
])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
