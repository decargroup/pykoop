"""Test a selection of invalid estimator parameter."""

import numpy as np
import pytest

import pykoop
from pykoop import lmi_regressors


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
    def test_bad_params(self, estimator):
        """Test a selection of invalid estimator parameter."""
        with pytest.raises(ValueError):
            estimator.fit(self.X)
