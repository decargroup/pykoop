"""Test scikit-learn compatibility using ``sklearn.utils.estimator_checks``."""

import pytest
import sklearn.utils.estimator_checks
from sklearn import preprocessing

import pykoop
import pykoop.lmi_regressors


class TestSklearn:
    """Test scikit-learn compatibility."""

    mosek_solver_params = {
        'solver': 'mosek',
        'dualize': True,
        '*_fsb_tol': 1e-6,
        '*_opt_tol': 1e-6,
    }

    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.Edmd(),
        pykoop.Dmdc(),
        pykoop.Dmd(),
        pykoop.AnglePreprocessor(),
        pykoop.PolynomialLiftingFn(),
        pykoop.DelayLiftingFn(),
        pykoop.SkLearnLiftingFn(preprocessing.MaxAbsScaler()),
        pykoop.BilinearInputLiftingFn(),
        pykoop.KoopmanPipeline(regressor=pykoop.Edmd()),
        pykoop.KoopmanPipeline(
            lifting_functions=[
                ('pl', pykoop.PolynomialLiftingFn()),
            ],
            regressor=pykoop.Edmd(),
        ),
        pykoop.SplitPipeline(
            lifting_functions_state=None,
            lifting_functions_input=None,
        ),
        pykoop.SplitPipeline(
            lifting_functions_state=[
                ('pl', pykoop.PolynomialLiftingFn()),
            ],
            lifting_functions_input=None,
        ),
        pykoop.Tsvd(),
    ])
    def test_compatible_estimator(self, estimator, check):
        """Test scikit-learn compatibility for standard estimators."""
        check(estimator)

    @pytest.mark.mosek
    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.lmi_regressors.LmiEdmd(
            alpha=1e-3,
            solver_params=mosek_solver_params,
        ),
        pykoop.lmi_regressors.LmiEdmdSpectralRadiusConstr(
            max_iter=1,
            solver_params=mosek_solver_params,
        ),
        pykoop.lmi_regressors.LmiEdmdHinfReg(
            alpha=1,
            ratio=1,
            max_iter=1,
            solver_params=mosek_solver_params,
        ),
        pykoop.lmi_regressors.LmiEdmdDissipativityConstr(
            max_iter=1,
            solver_params=mosek_solver_params,
        ),
        pykoop.lmi_regressors.LmiDmdc(
            alpha=1e-3,
            solver_params=mosek_solver_params,
        ),
        pykoop.lmi_regressors.LmiDmdcSpectralRadiusConstr(
            max_iter=1,
            solver_params=mosek_solver_params,
        ),
        pykoop.lmi_regressors.LmiDmdcHinfReg(
            alpha=1,
            ratio=1,
            max_iter=1,
            solver_params=mosek_solver_params,
        ),
        pykoop.lmi_regressors.LmiHinfZpkMeta(
            pykoop.lmi_regressors.LmiEdmdHinfReg(
                alpha=1,
                ratio=1,
                max_iter=1,
                solver_params=mosek_solver_params,
            )),
    ])
    def test_compatible_estimator_lmi(self, estimator, check, remote,
                                      remote_url):
        """Test scikit-learn compatibility for LMI-based regressors."""
        # Set MOSEK solver to remote server if needed
        if remote:
            if hasattr(estimator, 'hinf_regressor'):
                estimator.hinf_regressor.solver_params[
                    'mosek_server'] = remote_url
            else:
                estimator.solver_params['mosek_server'] = remote_url
        check(estimator)
