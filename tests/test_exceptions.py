import numpy as np
import pytest

import pykoop
from pykoop import lmi_regressors


@pytest.mark.parametrize(
    'class_, params',
    [
        # Edmd
        (pykoop.Edmd, {
            'alpha': -1
        }),
        # LmiEdmd
        (lmi_regressors.LmiEdmd, {
            'alpha': -1
        }),
        (lmi_regressors.LmiEdmd, {
            'ratio': -1
        }),
        (lmi_regressors.LmiEdmd, {
            'ratio': 0
        }),
        (lmi_regressors.LmiEdmd, {
            'reg_method': 'blahblah'
        }),
        (lmi_regressors.LmiEdmd, {
            'inv_method': 'blahblah'
        }),
        (lmi_regressors.LmiEdmd, {
            'tsvd_method': ('known_noise', )
        }),
        (lmi_regressors.LmiEdmd, {
            'tsvd_method': ('cutoff', )
        }),
        (lmi_regressors.LmiEdmd, {
            'picos_eps': -1
        }),
        # LmiDmdc
        (lmi_regressors.LmiDmdc, {
            'alpha': -1
        }),
        (lmi_regressors.LmiDmdc, {
            'ratio': -1
        }),
        (lmi_regressors.LmiDmdc, {
            'ratio': 0
        }),
        (lmi_regressors.LmiDmdc, {
            'tsvd_method': ('known_noise', )
        }),
        (lmi_regressors.LmiDmdc, {
            'tsvd_method': ('rank', 1)
        }),
        (lmi_regressors.LmiDmdc, {
            'picos_eps': -1
        }),
        # LmiEdmdSpectralRadiusConstr
        (lmi_regressors.LmiEdmdSpectralRadiusConstr, {
            'spectral_radius': -1
        }),
        (lmi_regressors.LmiEdmdSpectralRadiusConstr, {
            'spectral_radius': 0
        }),
        (lmi_regressors.LmiEdmdSpectralRadiusConstr, {
            'max_iter': -1
        }),
        (lmi_regressors.LmiEdmdSpectralRadiusConstr, {
            'iter_tol': -1
        }),
        (lmi_regressors.LmiEdmdSpectralRadiusConstr, {
            'alpha': -1
        }),
        # LmiDmdcSpectralRadiusConstr
        (lmi_regressors.LmiDmdcSpectralRadiusConstr, {
            'spectral_radius': -1
        }),
        (lmi_regressors.LmiDmdcSpectralRadiusConstr, {
            'max_iter': -1
        }),
        (lmi_regressors.LmiDmdcSpectralRadiusConstr, {
            'iter_tol': -1
        }),
        # LmiEdmdHinfReg
        (lmi_regressors.LmiEdmdHinfReg, {
            'alpha': -1
        }),
        (lmi_regressors.LmiEdmdHinfReg, {
            'alpha': 0
        }),
        (lmi_regressors.LmiEdmdHinfReg, {
            'ratio': 0
        }),
        (lmi_regressors.LmiEdmdHinfReg, {
            'weight': ('blah', np.eye(1), np.eye(1), np.eye(1), np.eye(1))
        }),
        (lmi_regressors.LmiEdmdHinfReg, {
            'max_iter': -1
        }),
        (lmi_regressors.LmiEdmdHinfReg, {
            'iter_tol': -1
        }),
        # LmiDmdcHinfReg
        (lmi_regressors.LmiDmdcHinfReg, {
            'alpha': -1
        }),
        (lmi_regressors.LmiDmdcHinfReg, {
            'alpha': 0
        }),
        (lmi_regressors.LmiDmdcHinfReg, {
            'ratio': 0
        }),
        (lmi_regressors.LmiDmdcHinfReg, {
            'weight': ('blah', np.eye(1), np.eye(1), np.eye(1), np.eye(1))
        }),
        (lmi_regressors.LmiDmdcHinfReg, {
            'max_iter': -1
        }),
        (lmi_regressors.LmiDmdcHinfReg, {
            'iter_tol': -1
        }),
        # LmiEdmdDissipativityConstr
        (lmi_regressors.LmiDmdcHinfReg, {
            'alpha': -1
        }),
        (lmi_regressors.LmiDmdcHinfReg, {
            'alpha': 0
        }),
        (lmi_regressors.LmiDmdcHinfReg, {
            'max_iter': -1
        }),
        (lmi_regressors.LmiDmdcHinfReg, {
            'iter_tol': -1
        }),
    ])
def test_bad_params(class_, params):
    X = np.array([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
    ])
    est = class_(**params)
    with pytest.raises(ValueError):
        est.fit(X)
