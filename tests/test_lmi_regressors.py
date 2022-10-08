"""Test :mod:`lmi_regressors`."""

import numpy as np
import pytest
import sklearn.linear_model
import sklearn.utils.estimator_checks
from scipy import signal

import pykoop
from pykoop import lmi_regressors


@pytest.fixture
def mosek_solver_params(remote, remote_url):
    """MOSEK solver parameters."""
    params = {
        'solver': 'mosek',
        'dualize': True,
    }
    # Set MOSEK solver to remote server if needed
    if remote:
        params['mosek_server'] = remote_url
    return params


@pytest.mark.mosek
@pytest.mark.parametrize(
    'regressor, mass_spring_damper, fit_tol, predict_tol',
    [
        (
            pykoop.lmi_regressors.LmiEdmd(alpha=0, inv_method='eig'),
            'mass_spring_damper_no_input',
            1e-4,
            1e-5,
        ),
        (
            pykoop.lmi_regressors.LmiEdmd(
                alpha=0,
                inv_method='inv',
                solver_params={'dualize': False},
            ),
            'mass_spring_damper_no_input',
            1e-3,
            1e-4,
        ),
        (
            pykoop.lmi_regressors.LmiEdmd(alpha=0, inv_method='ldl'),
            'mass_spring_damper_no_input',
            1e-4,
            1e-5,
        ),
        (
            pykoop.lmi_regressors.LmiEdmd(alpha=0, inv_method='chol'),
            'mass_spring_damper_no_input',
            1e-4,
            1e-5,
        ),
        (
            pykoop.lmi_regressors.LmiEdmd(alpha=0, inv_method='sqrt'),
            'mass_spring_damper_no_input',
            1e-4,
            1e-5,
        ),
        (
            pykoop.lmi_regressors.LmiEdmd(alpha=0, inv_method='svd'),
            'mass_spring_damper_no_input',
            1e-4,
            1e-5,
        ),
        (
            pykoop.lmi_regressors.LmiDmdc(alpha=0),
            'mass_spring_damper_no_input',
            1e-4,
            1e-5,
        ),
    ],
)
class TestLmiRegressorsExact:
    """Test LMI regressors with exact solutions."""

    def test_fit(
        self,
        request,
        regressor,
        mass_spring_damper,
        fit_tol,
        predict_tol,
        mosek_solver_params,
    ):
        """Test fit accuracy by comparing Koopman matrix."""
        # Get fixture from name
        msd = request.getfixturevalue(mass_spring_damper)
        # Update solver settings
        if regressor.solver_params is None:
            regressor.solver_params = mosek_solver_params
        else:
            mosek_solver_params.update(regressor.solver_params)
            regressor.solver_params = mosek_solver_params
        # Fit regressor
        regressor.fit(
            msd['X_train'],
            msd['Xp_train'],
            n_inputs=msd['n_inputs'],
            episode_feature=msd['episode_feature'],
        )
        # Test value of Koopman operator
        np.testing.assert_allclose(
            regressor.coef_.T,
            msd['U_valid'],
            atol=fit_tol,
            rtol=0,
        )

    def test_predict(
        self,
        request,
        regressor,
        mass_spring_damper,
        fit_tol,
        predict_tol,
        mosek_solver_params,
    ):
        """Test fit accuracy by comparing prediction."""
        # Get fixture from name
        msd = request.getfixturevalue(mass_spring_damper)
        # Update solver settings
        if regressor.solver_params is None:
            regressor.solver_params = mosek_solver_params
        else:
            mosek_solver_params.update(regressor.solver_params)
            regressor.solver_params = mosek_solver_params
        # Fit regressor
        regressor.fit(
            msd['X_train'],
            msd['Xp_train'],
            n_inputs=msd['n_inputs'],
            episode_feature=msd['episode_feature'],
        )
        # Test prediction
        prediction = regressor.predict(msd['X_valid'])
        np.testing.assert_allclose(
            prediction,
            msd['Xp_valid'],
            atol=predict_tol,
            rtol=0,
        )


@pytest.mark.mosek
@pytest.mark.parametrize(
    'regressor, mass_spring_damper, fit_tol',
    [
        (
            pykoop.lmi_regressors.LmiEdmd(alpha=1, inv_method='chol'),
            'mass_spring_damper_no_input',
            1e-4,
        ),
        (
            pykoop.lmi_regressors.LmiEdmd(alpha=0, inv_method='chol'),
            'mass_spring_damper_no_input',
            1e-4,
        ),
    ],
)
class TestLmiRegressorsTikhonov:
    """Test Tikhonov LMI regressors."""

    def test_fit(
        self,
        request,
        regressor,
        mass_spring_damper,
        fit_tol,
        mosek_solver_params,
    ):
        """Test fit accuracy by comparing Koopman matrix."""
        # Get fixture from name
        msd = request.getfixturevalue(mass_spring_damper)
        # Update solver settings
        if regressor.solver_params is None:
            regressor.solver_params = mosek_solver_params
        else:
            mosek_solver_params.update(regressor.solver_params)
            regressor.solver_params = mosek_solver_params
        # Fit regressor
        regressor.fit(
            msd['X_train'],
            msd['Xp_train'],
            n_inputs=msd['n_inputs'],
            episode_feature=msd['episode_feature'],
        )
        # Fit equivalent ``scikit-learn`` regressor
        clf = sklearn.linear_model.Ridge(
            alpha=regressor.alpha,
            fit_intercept=False,
            solver='cholesky',
            tol=1e-8,
        )
        clf.fit(msd['X_train'], msd['Xp_train'])
        U_valid = clf.coef_
        # Test value of Koopman operator
        np.testing.assert_allclose(
            regressor.coef_.T,
            U_valid,
            atol=fit_tol,
            rtol=0,
        )


@pytest.mark.mosek
@pytest.mark.parametrize(
    'regressor, mass_spring_damper',
    [
        (
            pykoop.lmi_regressors.LmiEdmd(
                alpha=1,
                reg_method='twonorm',
                inv_method='chol',
                solver_params={'dualize': False},
            ),
            'mass_spring_damper_no_input',
        ),
        (
            pykoop.lmi_regressors.LmiEdmd(
                alpha=1,
                reg_method='nuclear',
                inv_method='chol',
                solver_params={'dualize': False},
            ),
            'mass_spring_damper_no_input',
        ),
        (
            pykoop.lmi_regressors.LmiEdmdSpectralRadiusConstr(
                inv_method='chol',
                spectral_radius=1.1,
            ),
            'mass_spring_damper_no_input',
        ),
        (
            pykoop.lmi_regressors.LmiEdmdSpectralRadiusConstr(
                inv_method='chol',
                spectral_radius=0.8,
                solver_params={'dualize': False},
            ),
            'mass_spring_damper_no_input',
        ),
        (
            pykoop.lmi_regressors.LmiEdmdHinfReg(
                inv_method='eig',
                max_iter=100,
                iter_atol=1e-3,
                alpha=1,
                ratio=1,
                solver_params={'dualize': False}),
            'mass_spring_damper_sine_input',
        ),
    ],
)
class TestLmiRegressorsRegression:
    """Run regression tests for LMI regressors without easy-to-check solutions.

    Attributes
    ----------
    tol : float
        Tolerance for regression test.
    """

    tol = 1e-3

    def test_fit_predict(
        self,
        request,
        ndarrays_regression,
        regressor,
        mass_spring_damper,
        mosek_solver_params,
    ):
        """Test fit accuracy by comparing Koopman matrix and prediction."""
        # Get fixture from name
        msd = request.getfixturevalue(mass_spring_damper)
        # Update solver settings
        if regressor.solver_params is None:
            regressor.solver_params = mosek_solver_params
        else:
            mosek_solver_params.update(regressor.solver_params)
            regressor.solver_params = mosek_solver_params
        # Fit regressor
        regressor.fit(
            msd['X_train'],
            msd['Xp_train'],
            n_inputs=msd['n_inputs'],
            episode_feature=msd['episode_feature'],
        )
        # Compute prediction
        prediction = regressor.predict(msd['X_valid'])
        # Compare to prior results
        ndarrays_regression.check(
            {
                'regressor.coef_': regressor.coef_,
                'prediction': prediction,
            },
            default_tolerance=dict(atol=self.tol, rtol=0),
        )


@pytest.mark.mosek
@pytest.mark.parametrize(
    'regressor',
    [
        pykoop.lmi_regressors.LmiEdmdHinfReg,
        pykoop.lmi_regressors.LmiDmdcHinfReg,
    ],
)
class TestLmiHinfZpkMeta:
    """Test :class:`LmiHinfZpkMeta`.

    Attributes
    ----------
    tol : float
        Tolerance for regression test.
    """

    tol = 1e-6

    def test_hinf_weight(
        self,
        regressor,
        mosek_solver_params,
        mass_spring_damper_sine_input,
    ):
        """Test that :class:`LmiHinfZpkMeta` weight is correct.

        .. todo:: Break up multiple asserts.
        """
        # Compute state space matrices
        ss_ct = signal.ZerosPolesGain(
            [-0],
            [-5],
            1,
        ).to_ss()
        ss_dt = ss_ct.to_discrete(
            dt=mass_spring_damper_sine_input['t_step'],
            method='bilinear',
        )
        weight = (
            'post',
            ss_dt.A,
            ss_dt.B,
            ss_dt.C,
            ss_dt.D,
        )
        est_expected = regressor(
            weight=weight,
            solver_params=mosek_solver_params,
        )
        est_actual = pykoop.lmi_regressors.LmiHinfZpkMeta(
            hinf_regressor=regressor(solver_params=mosek_solver_params),
            type='post',
            zeros=-0,
            poles=-5,
            gain=1,
            discretization='bilinear',
            t_step=mass_spring_damper_sine_input['t_step'],
        )
        # Fit regressors
        est_expected.fit(
            mass_spring_damper_sine_input['X_train'],
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=mass_spring_damper_sine_input['episode_feature'],
        )
        est_actual.fit(
            mass_spring_damper_sine_input['X_train'],
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=mass_spring_damper_sine_input['episode_feature'],
        )
        # Check Koopman matrices
        U_expected = est_expected.coef_.T
        U_actual = est_actual.hinf_regressor_.coef_.T
        np.testing.assert_allclose(
            U_actual,
            U_expected,
            atol=self.tol,
            rtol=0,
        )
        # Check state space matrices
        assert est_expected.weight[0] == est_actual.hinf_regressor_.weight[0]
        for i in range(1, 5):
            np.testing.assert_allclose(
                est_expected.weight[i],
                est_actual.hinf_regressor_.weight[i],
                atol=self.tol,
                rtol=0,
            )

    def test_pole_zero_units(
        self,
        regressor,
        mosek_solver_params,
        mass_spring_damper_sine_input,
    ):
        """Test that :class:`LmiHinfZpkMeta` zero and pole units are correct.

        .. todo:: Break up multiple asserts.
        """
        est_1 = pykoop.lmi_regressors.LmiHinfZpkMeta(
            hinf_regressor=regressor(solver_params=mosek_solver_params),
            type='post',
            zeros=-0,
            poles=(-2 * np.pi / mass_spring_damper_sine_input['t_step']) / 2,
            gain=1,
            discretization='bilinear',
            t_step=mass_spring_damper_sine_input['t_step'],
            units='rad/s',
        )
        est_2 = pykoop.lmi_regressors.LmiHinfZpkMeta(
            hinf_regressor=regressor(solver_params=mosek_solver_params),
            type='post',
            zeros=-0,
            poles=(-1 / mass_spring_damper_sine_input['t_step']) / 2,
            gain=1,
            discretization='bilinear',
            t_step=mass_spring_damper_sine_input['t_step'],
            units='hz',
        )
        est_3 = pykoop.lmi_regressors.LmiHinfZpkMeta(
            hinf_regressor=regressor(solver_params=mosek_solver_params),
            type='post',
            zeros=-0,
            poles=-1,
            gain=1,
            discretization='bilinear',
            t_step=mass_spring_damper_sine_input['t_step'],
            units='normalized',
        )
        # Fit estimators
        est_1.fit(
            mass_spring_damper_sine_input['X_train'],
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=mass_spring_damper_sine_input['episode_feature'],
        )
        est_2.fit(
            mass_spring_damper_sine_input['X_train'],
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=mass_spring_damper_sine_input['episode_feature'],
        )
        est_3.fit(
            mass_spring_damper_sine_input['X_train'],
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=mass_spring_damper_sine_input['episode_feature'],
        )
        # Check poles
        np.testing.assert_allclose(
            est_1.ss_ct_.poles,
            est_2.ss_ct_.poles,
            atol=self.tol,
            rtol=0,
        )
        np.testing.assert_allclose(
            est_2.ss_ct_.poles,
            est_3.ss_ct_.poles,
            atol=self.tol,
            rtol=0,
        )
        np.testing.assert_allclose(
            est_3.ss_ct_.poles,
            est_1.ss_ct_.poles,
            atol=self.tol,
            rtol=0,
        )
        # Check zeros
        np.testing.assert_allclose(
            est_1.ss_ct_.zeros,
            est_2.ss_ct_.zeros,
            atol=self.tol,
            rtol=0,
        )
        np.testing.assert_allclose(
            est_2.ss_ct_.zeros,
            est_3.ss_ct_.zeros,
            atol=self.tol,
            rtol=0,
        )
        np.testing.assert_allclose(
            est_3.ss_ct_.zeros,
            est_1.ss_ct_.zeros,
            atol=self.tol,
            rtol=0,
        )
        # Check parameters
        assert est_1.n_features_in_ == est_1.hinf_regressor_.n_features_in_
        assert est_1.n_states_in_ == est_1.hinf_regressor_.n_states_in_
        assert est_1.n_inputs_in_ == est_1.hinf_regressor_.n_inputs_in_
        assert est_1.episode_feature_ == est_1.hinf_regressor_.episode_feature_
        np.testing.assert_allclose(est_1.coef_, est_1.hinf_regressor_.coef_)


@pytest.mark.mosek
class TestSkLearn:
    """Test ``scikit-learn`` compatibility."""

    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.lmi_regressors.LmiEdmd(alpha=1e-3, ),
        pykoop.lmi_regressors.LmiEdmdSpectralRadiusConstr(max_iter=1, ),
        pykoop.lmi_regressors.LmiEdmdHinfReg(alpha=1, ratio=1, max_iter=1),
        pykoop.lmi_regressors.LmiEdmdDissipativityConstr(max_iter=1, ),
        pykoop.lmi_regressors.LmiDmdc(alpha=1e-3, ),
        pykoop.lmi_regressors.LmiDmdcSpectralRadiusConstr(max_iter=1, ),
        pykoop.lmi_regressors.LmiDmdcHinfReg(alpha=1, ratio=1, max_iter=1),
        pykoop.lmi_regressors.LmiHinfZpkMeta(
            pykoop.lmi_regressors.LmiEdmdHinfReg(
                alpha=1,
                ratio=1,
                max_iter=1,
            )),
    ])
    def test_compatible_estimator(self, estimator, check, mosek_solver_params):
        """Test ``scikit-learn`` compatibility for LMI-based regressors."""
        if hasattr(estimator, 'hinf_regressor'):
            estimator.hinf_regressor.solver_params = mosek_solver_params
        else:
            estimator.solver_params = mosek_solver_params
        check(estimator)


@pytest.mark.mosek
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
