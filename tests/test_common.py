import sklearn.utils.estimator_checks
import pykoop.dmd
import pykoop.lmi
import pykoop.lifting_functions
from sklearn import preprocessing


@sklearn.utils.estimator_checks.parametrize_with_checks([
    pykoop.dmd.Edmd(),
    pykoop.lmi.LmiEdmdTikhonovReg(alpha=0),
    pykoop.lmi.LmiEdmdTwoNormReg(alpha=1, ratio=1),
    pykoop.lmi.LmiEdmdNuclearNormReg(alpha=1, ratio=1),
    pykoop.lmi.LmiEdmdSpectralRadiusConstr(tol=100),  # Loosen tol
    pykoop.lmi.LmiEdmdHinfReg(alpha=1, ratio=1, tol=100),  # Loosen tol
    pykoop.lifting_functions.AnglePreprocessor(),
    pykoop.lifting_functions.PolynomialLiftingFn(),
    pykoop.lifting_functions.DelayLiftingFn(),
    pykoop.lifting_functions.SkLearnLiftingFn(preprocessing.MaxAbsScaler()),
    pykoop.lifting_functions.BilinearInputLiftingFn(),
    pykoop.koopman_pipeline.KoopmanPipeline(
        regressor=pykoop.dmd.Edmd()
    ),
    pykoop.koopman_pipeline.KoopmanPipeline(
        lifting_functions=[
            pykoop.lifting_functions.PolynomialLiftingFn(),
        ],
        regressor=pykoop.dmd.Edmd(),
    ),
    pykoop.koopman_pipeline.SplitPipeline(
        lifting_functions_state=None,
        lifting_functions_input=None,
    ),
    pykoop.koopman_pipeline.SplitPipeline(
        lifting_functions_state=[
            pykoop.lifting_functions.PolynomialLiftingFn(),
        ],
        lifting_functions_input=None,
    ),
])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
