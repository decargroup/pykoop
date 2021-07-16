import pykoop
import pykoop.lmi_regressors
import sklearn.utils.estimator_checks
from sklearn import preprocessing


@sklearn.utils.estimator_checks.parametrize_with_checks([
    pykoop.Edmd(),
    pykoop.lmi_regressors.LmiEdmd(alpha=0),
    pykoop.lmi_regressors.LmiEdmd(alpha=1, reg_method='twonorm', ratio=1),
    pykoop.lmi_regressors.LmiEdmd(alpha=1, reg_method='nuclear', ratio=1),
    # pykoop.lmi.LmiEdmdTwoNormReg(alpha=1, ratio=1),
    # pykoop.lmi.LmiEdmdNuclearNormReg(alpha=1, ratio=1),
    # pykoop.lmi.LmiEdmdSpectralRadiusConstr(tol=100),  # Loosen tol
    # pykoop.lmi.LmiEdmdHinfReg(alpha=1, ratio=1, tol=100),  # Loosen tol
    pykoop.AnglePreprocessor(),
    pykoop.PolynomialLiftingFn(),
    pykoop.DelayLiftingFn(),
    pykoop.SkLearnLiftingFn(preprocessing.MaxAbsScaler()),
    pykoop.BilinearInputLiftingFn(),
    pykoop.KoopmanPipeline(regressor=pykoop.Edmd()),
    pykoop.KoopmanPipeline(
        lifting_functions=[
            pykoop.PolynomialLiftingFn(),
        ],
        regressor=pykoop.Edmd(),
    ),
    pykoop.koopman_pipeline.SplitPipeline(
        lifting_functions_state=None,
        lifting_functions_input=None,
    ),
    pykoop.koopman_pipeline.SplitPipeline(
        lifting_functions_state=[
            pykoop.PolynomialLiftingFn(),
        ],
        lifting_functions_input=None,
    ),
])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
