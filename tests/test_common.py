import sklearn.utils.estimator_checks
import pykoop.dmd
import pykoop.lmi
import pykoop.lifting_functions


@sklearn.utils.estimator_checks.parametrize_with_checks([
    pykoop.dmd.Edmd(),
    pykoop.lmi.LmiEdmdTikhonovReg(alpha=0),
    pykoop.lmi.LmiEdmdTwoNormReg(alpha=1, ratio=1),
    pykoop.lmi.LmiEdmdNuclearNormReg(alpha=1, ratio=1),
    pykoop.lmi.LmiEdmdSpectralRadiusConstr(tol=100),  # Loosen tol
    pykoop.lmi.LmiEdmdHinfReg(alpha=1, ratio=1, tol=100),  # Loosen tol
    pykoop.lifting_functions.Delay(),
])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
