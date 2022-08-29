"""Test a selection of invalid estimator parameter."""

import numpy as np
import pytest
import sklearn.utils.estimator_checks

import pykoop
from pykoop import lmi_regressors


class TestSklearn:
    """Test scikit-learn compatibility."""

    mosek_solver_params = {
        'solver': 'mosek',
        'dualize': True,
        '*_fsb_tol': 1e-6,
        '*_opt_tol': 1e-6,
    }

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
    def test_compatible_estimator(self, estimator, check, remote, remote_url):
        """Test scikit-learn compatibility for LMI-based regressors."""
        # Set MOSEK solver to remote server if needed
        if remote:
            if hasattr(estimator, 'hinf_regressor'):
                estimator.hinf_regressor.solver_params[
                    'mosek_server'] = remote_url
            else:
                estimator.solver_params['mosek_server'] = remote_url
        check(estimator)


class TestExceptions:
    """Test a selection of invalid estimator parameter."""

    X = np.array([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
    ])

    @pytest.mark.parametrize('estimator', [
        lmi_regressors.LmiEdmd(alpha=-1),
        lmi_regressors.LmiEdmd(ratio=-1),
        lmi_regressors.LmiEdmd(ratio=0),
        lmi_regressors.LmiEdmd(reg_method='blah'),
        lmi_regressors.LmiEdmd(inv_method='blah'),
        lmi_regressors.LmiEdmd(picos_eps=-1),
        lmi_regressors.LmiDmdc(alpha=-1),
        lmi_regressors.LmiDmdc(ratio=-1),
        lmi_regressors.LmiDmdc(ratio=0),
        lmi_regressors.LmiDmdc(reg_method='blah'),
        lmi_regressors.LmiDmdc(picos_eps=-1),
        lmi_regressors.LmiEdmdSpectralRadiusConstr(spectral_radius=-1),
        lmi_regressors.LmiEdmdSpectralRadiusConstr(spectral_radius=0),
        lmi_regressors.LmiEdmdSpectralRadiusConstr(max_iter=-1),
        lmi_regressors.LmiEdmdSpectralRadiusConstr(iter_atol=-1),
        lmi_regressors.LmiEdmdSpectralRadiusConstr(iter_rtol=-1),
        lmi_regressors.LmiEdmdSpectralRadiusConstr(alpha=-1),
        lmi_regressors.LmiDmdcSpectralRadiusConstr(spectral_radius=-1),
        lmi_regressors.LmiDmdcSpectralRadiusConstr(spectral_radius=0),
        lmi_regressors.LmiDmdcSpectralRadiusConstr(max_iter=-1),
        lmi_regressors.LmiDmdcSpectralRadiusConstr(iter_atol=-1),
        lmi_regressors.LmiDmdcSpectralRadiusConstr(iter_rtol=-1),
        lmi_regressors.LmiDmdcSpectralRadiusConstr(alpha=-1),
        lmi_regressors.LmiEdmdHinfReg(alpha=-1),
        lmi_regressors.LmiEdmdHinfReg(alpha=0),
        lmi_regressors.LmiEdmdHinfReg(ratio=0),
        lmi_regressors.LmiEdmdHinfReg(weight=(
            'blah',
            np.eye(1),
            np.eye(1),
            np.eye(1),
            np.eye(1),
        )),
        lmi_regressors.LmiEdmdHinfReg(max_iter=-1),
        lmi_regressors.LmiEdmdHinfReg(iter_atol=-1),
        lmi_regressors.LmiEdmdHinfReg(iter_rtol=-1),
        lmi_regressors.LmiDmdcHinfReg(alpha=-1),
        lmi_regressors.LmiDmdcHinfReg(alpha=0),
        lmi_regressors.LmiDmdcHinfReg(ratio=0),
        lmi_regressors.LmiDmdcHinfReg(weight=(
            'blah',
            np.eye(1),
            np.eye(1),
            np.eye(1),
            np.eye(1),
        )),
        lmi_regressors.LmiDmdcHinfReg(max_iter=-1),
        lmi_regressors.LmiDmdcHinfReg(iter_atol=-1),
        lmi_regressors.LmiEdmdDissipativityConstr(alpha=-1),
        lmi_regressors.LmiEdmdDissipativityConstr(alpha=0),
        lmi_regressors.LmiEdmdDissipativityConstr(max_iter=-1),
        lmi_regressors.LmiEdmdDissipativityConstr(iter_atol=-1),
        lmi_regressors.LmiEdmdDissipativityConstr(iter_rtol=-1),
    ])
    def test_invalid_params(self, estimator):
        """Test a selection of invalid estimator parameter."""
        with pytest.raises(ValueError):
            estimator.fit(self.X)
