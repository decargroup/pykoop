import numpy as np
import pytest
from scipy import integrate, linalg, signal
from sklearn import linear_model

import pykoop
import pykoop.dynamic_models
import pykoop.lmi_regressors

# TODO This file is a nightmare


@pytest.fixture(
    params=[
        (pykoop.Edmd(), 'msd-no-input', 1e-5, 1e-5, 'exact'),
        (pykoop.Dmdc(), 'msd-no-input', 1e-5, 1e-5, 'exact'),
        (pykoop.Dmd(), 'msd-no-input', 1e-5, 1e-5, 'exact'),
        (
            pykoop.Dmdc(tsvd_unshifted=pykoop.Tsvd('known_noise', 1),
                        tsvd_shifted=pykoop.Tsvd('known_noise', 1)),
            'msd-no-input',
            None,
            None,
            'exact',
        ),
        (
            pykoop.Dmd(tsvd=pykoop.Tsvd('known_noise', 1)),
            'msd-no-input',
            None,
            None,
            'exact',
        ),
        (
            pykoop.lmi_regressors.LmiEdmd(alpha=0, inv_method='eig'),
            'msd-no-input',
            1e-4,
            1e-5,
            'exact',
        ),
        (
            pykoop.lmi_regressors.LmiEdmd(
                alpha=0,
                inv_method='inv',
                solver_params={'dualize': False},
            ),
            'msd-no-input',
            1e-3,
            1e-4,
            'exact',
        ),
        (pykoop.lmi_regressors.LmiEdmd(alpha=0, inv_method='ldl'),
         'msd-no-input', 1e-4, 1e-5, 'exact'),
        (pykoop.lmi_regressors.LmiEdmd(alpha=0, inv_method='chol'),
         'msd-no-input', 1e-4, 1e-5, 'exact'),
        (pykoop.lmi_regressors.LmiEdmd(alpha=0, inv_method='sqrt'),
         'msd-no-input', 1e-4, 1e-5, 'exact'),
        (pykoop.lmi_regressors.LmiEdmd(alpha=0, inv_method='svd'),
         'msd-no-input', 1e-4, 1e-5, 'exact'),
        (
            pykoop.lmi_regressors.LmiDmdc(alpha=0),
            'msd-no-input',
            1e-4,
            1e-5,
            'exact',
        ),
        (
            pykoop.lmi_regressors.LmiEdmd(alpha=1, inv_method='chol'),
            'msd-no-input',
            1e-4,
            None,
            'sklearn-ridge',
        ),
        # (pykoop.lmi_regressors.LmiDmdc(alpha=2.29), 'msd-no-input', 1e-4,
        #  None, 'sklearn-ridge-1'),  # I don't think this is the same
        (
            pykoop.lmi_regressors.LmiEdmd(
                alpha=1,
                reg_method='twonorm',
                inv_method='chol',
                solver_params={'dualize': False},
            ),
            'msd-no-input',
            1e-3,
            None,
            # Regression test! Not unit test!
            np.array([
                [0.929949, 0.065987],
                [-0.102050, 0.893498],
            ])),
        (
            pykoop.lmi_regressors.LmiEdmd(
                alpha=1,
                reg_method='nuclear',
                inv_method='chol',
                solver_params={'dualize': False},
            ),
            'msd-no-input',
            1e-3,
            None,
            # Regression test! Not unit test!
            np.array([
                [0.875848, -0.017190],
                [-0.210071, 0.727786],
            ])),
        pytest.param(
            (
                pykoop.lmi_regressors.LmiEdmdSpectralRadiusConstr(
                    inv_method='chol', spectral_radius=1.1),
                'msd-no-input',
                2e-4,
                1e-5,
                # Since the constraint is larger than the actual eigenvalue
                # magnitudes it will have no effect and we can compare to the
                # exact solution.
                'exact'),
            marks=pytest.mark.slow),
        pytest.param(
            (
                pykoop.lmi_regressors.LmiEdmdSpectralRadiusConstr(
                    inv_method='chol',
                    spectral_radius=0.8,
                    solver_params={'dualize': False}),
                'msd-no-input',
                1e-3,
                None,
                # Regression test generated from this code. Result was manually
                # checked (eigenvalue magnitudes are less than 0.8) but
                # strictly speaking, it hasn't been checked against other code.
                np.array([
                    [0.797335, 0.065241],
                    [-0.065242, 0.797335],
                ])),
            marks=pytest.mark.slow),
        pytest.param(
            (
                pykoop.lmi_regressors.LmiEdmdHinfReg(
                    inv_method='eig',
                    max_iter=100,
                    iter_atol=1e-3,
                    alpha=1,
                    ratio=1,
                    solver_params={'dualize': False}),
                'msd-sin-input',
                1e-3,
                None,
                # Regression test! Not unit test!
                np.array([
                    [0.509525, -0.279917, 0.471535],
                    [-0.296477, 0.162875, 0.831542],
                ])),
            marks=pytest.mark.slow),
    ],
    ids=lambda value: f'{value[0]}-{value[1]}')  # Formatting for test IDs
def scenario(request, remote, remote_url):
    regressor, system, fit_tol, predict_tol, soln = request.param
    # Set MOSEK solver to remote server if needed
    if remote and hasattr(regressor, 'solver_params'):
        if regressor.solver_params is None:
            regressor.solver_params = {'mosek_server': remote_url}
        else:
            regressor.solver_params['mosek_server'] = remote_url
    # Simulate or load data
    # Not all systems and solutions are compatible.
    # For `exact` to work, `t_step`, `A`, and `y` must be defined.
    # For `ridge` to work, only `y` is needed
    if system == 'msd-no-input':
        # Set up problem
        t_range = (0, 10)
        t_step = 0.1
        msd = pykoop.dynamic_models.MassSpringDamper(0.5, 0.7, 0.6)
        # Solve ODE for training data
        x0 = np.array([1, 0])
        t, x = msd.simulate(
            t_range,
            t_step,
            x0,
            lambda t: 0,
            rtol=1e-8,
            atol=1e-8,
        )
        A = msd.A
        # Split the data
        y_train, y_valid = np.split(x.T, 2, axis=1)
        X_train = y_train[:, :-1]
        Xp_train = y_train[:, 1:]
        X_valid = y_valid[:, :-1]
        Xp_valid = y_valid[:, 1:]
    elif system == 'msd-sin-input':
        # Set up problem
        t_range = (0, 10)
        t_step = 0.1
        msd = pykoop.dynamic_models.MassSpringDamper(0.5, 0.7, 0.6)

        def u(t):
            return 0.1 * np.sin(t)

        # Solve ODE for training data
        x0 = np.array([0, 0])
        t, x = msd.simulate(
            t_range,
            t_step,
            x0,
            u,
            rtol=1e-8,
            atol=1e-8,
        )
        A = msd.A
        # Split the data
        y_train, y_valid = np.split(x.T, 2, axis=1)
        u_train, u_valid = np.split(np.reshape(u(t), (1, -1)), 2, axis=1)
        X_train = np.vstack((y_train[:, :-1], u_train[:, :-1]))
        Xp_train = y_train[:, 1:]
        X_valid = np.vstack((y_valid[:, :-1], u_valid[:, :-1]))
        Xp_valid = y_valid[:, 1:]
    # Approximate the Koopman operator
    if type(soln) == np.ndarray:
        U_valid = soln
    elif soln == 'exact':
        U_valid = linalg.expm(A * t_step)
    elif soln == 'sklearn-ridge':
        clf = linear_model.Ridge(alpha=regressor.alpha,
                                 fit_intercept=False,
                                 solver='cholesky',
                                 tol=1e-8)
        clf.fit(X_train.T, Xp_train.T)
        U_valid = clf.coef_
    # Return fixture dictionary
    return {
        'X_train': X_train,
        'Xp_train': Xp_train,
        'X_valid': X_valid,
        'Xp_valid': Xp_valid,
        'U_valid': U_valid,
        'regressor': regressor,
        'fit_tol': fit_tol,
        'predict_tol': predict_tol,
    }


def test_scenario_data(scenario):
    # Make sure training and validation sets are not the same
    assert not np.allclose(scenario['X_train'], scenario['X_valid'])
    assert not np.allclose(scenario['Xp_train'], scenario['Xp_valid'])
    # Make sure Xp is time-shifted version of X
    p_theta = scenario['Xp_train'].shape[0]
    np.testing.assert_allclose(scenario['X_train'][:p_theta, 1:],
                               scenario['Xp_train'][:, :-1])
    np.testing.assert_allclose(scenario['X_valid'][:p_theta, 1:],
                               scenario['Xp_valid'][:, :-1])


def test_fit(scenario):
    # Fit regressor
    scenario['regressor'].fit(scenario['X_train'].T, scenario['Xp_train'].T)
    if scenario['fit_tol'] is None:
        pytest.skip()
    # Test value of Koopman operator
    np.testing.assert_allclose(
        scenario['regressor'].coef_.T,
        scenario['U_valid'],
        atol=scenario['fit_tol'],
        rtol=0,
    )


def test_predict(scenario):
    # Fit regressor
    scenario['regressor'].fit(scenario['X_train'].T, scenario['Xp_train'].T)
    if scenario['predict_tol'] is None:
        pytest.skip()
    # Test prediction
    np.testing.assert_allclose(
        scenario['regressor'].predict(scenario['X_valid'].T).T,
        scenario['Xp_valid'],
        atol=scenario['predict_tol'],
        rtol=0,
    )


@pytest.mark.slow
@pytest.mark.parametrize('cls', [
    pykoop.lmi_regressors.LmiEdmdHinfReg,
    pykoop.lmi_regressors.LmiDmdcHinfReg,
])
def test_hinf_zpk_meta(cls, remote, remote_url):
    # Specify duration
    t_range = (0, 10)
    t_step = 0.1
    # Create object
    msd = pykoop.dynamic_models.MassSpringDamper(
        mass=0.5,
        stiffness=0.7,
        damping=0.6,
    )

    # Specify input
    def u(t):
        """Sinusoidal input."""
        return 0.1 * np.sin(t)

    # Specify initial conditions
    x0 = msd.x0(np.array([0, 0]))
    # Simulate ODE
    t, x = msd.simulate(t_range, t_step, x0, u, rtol=1e-8, atol=1e-8)
    # Format the data
    X_msd = np.hstack((
        np.zeros((t.shape[0], 1)),  # episode feature
        x,
        np.reshape(u(t), (-1, 1)),
    ))
    # Compute state space matrices
    ss_ct = signal.ZerosPolesGain(
        [-0],
        [-5],
        1,
    ).to_ss()
    ss_dt = ss_ct.to_discrete(
        dt=t_step,
        method='bilinear',
    )
    weight = (
        'post',
        ss_dt.A,
        ss_dt.B,
        ss_dt.C,
        ss_dt.D,
    )
    est_expected = cls(weight=weight)
    est_actual = pykoop.lmi_regressors.LmiHinfZpkMeta(
        hinf_regressor=cls(),
        type='post',
        zeros=-0,
        poles=-5,
        gain=1,
        discretization='bilinear',
        t_step=t_step,
    )
    # Set MOSEK solver to remote server if needed
    if remote:
        est_expected.solver_params = {'mosek_server': remote_url}
        est_actual.solver_params = {'mosek_server': remote_url}
    # Fit regressors
    est_expected.fit(X_msd, n_inputs=1, episode_feature=True)
    est_actual.fit(X_msd, n_inputs=1, episode_feature=True)
    # Check Koopman matrices
    U_expected = est_expected.coef_.T
    U_actual = est_actual.hinf_regressor_.coef_.T
    np.testing.assert_allclose(U_actual, U_expected)
    # Check state space matrices
    assert est_expected.weight[0] == est_actual.hinf_regressor_.weight[0]
    for i in range(1, 5):
        np.testing.assert_allclose(est_expected.weight[i],
                                   est_actual.hinf_regressor_.weight[i])


@pytest.mark.slow
def test_hinf_zpk_units(remote, remote_url):
    # Specify duration
    t_range = (0, 10)
    t_step = 0.1
    # Create object
    msd = pykoop.dynamic_models.MassSpringDamper(
        mass=0.5,
        stiffness=0.7,
        damping=0.6,
    )

    # Specify input
    def u(t):
        """Sinusoidal input."""
        return 0.1 * np.sin(t)

    # Specify initial conditions
    x0 = msd.x0(np.array([0, 0]))
    # Simulate ODE
    t, x = msd.simulate(t_range, t_step, x0, u, rtol=1e-8, atol=1e-8)
    # Format the data
    X_msd = np.hstack((
        np.zeros((t.shape[0], 1)),  # episode feature
        x,
        np.reshape(u(t), (-1, 1)),
    ))

    est_1 = pykoop.lmi_regressors.LmiHinfZpkMeta(
        hinf_regressor=pykoop.lmi_regressors.LmiEdmdHinfReg(),
        type='post',
        zeros=-0,
        poles=(-2 * np.pi / t_step) / 2,
        gain=1,
        discretization='bilinear',
        t_step=t_step,
        units='rad/s',
    )
    est_2 = pykoop.lmi_regressors.LmiHinfZpkMeta(
        hinf_regressor=pykoop.lmi_regressors.LmiEdmdHinfReg(),
        type='post',
        zeros=-0,
        poles=(-1 / t_step) / 2,
        gain=1,
        discretization='bilinear',
        t_step=t_step,
        units='hz',
    )
    est_3 = pykoop.lmi_regressors.LmiHinfZpkMeta(
        hinf_regressor=pykoop.lmi_regressors.LmiEdmdHinfReg(),
        type='post',
        zeros=-0,
        poles=-1,
        gain=1,
        discretization='bilinear',
        t_step=t_step,
        units='normalized',
    )
    # Set MOSEK solver to remote server if needed
    if remote:
        est_1.solver_params = {'mosek_server': remote_url}
        est_2.solver_params = {'mosek_server': remote_url}
        est_3.solver_params = {'mosek_server': remote_url}
    # Fit estimators
    est_1.fit(X_msd, n_inputs=1, episode_feature=True)
    est_2.fit(X_msd, n_inputs=1, episode_feature=True)
    est_3.fit(X_msd, n_inputs=1, episode_feature=True)
    # Check poles
    np.testing.assert_allclose(est_1.ss_ct_.poles, est_2.ss_ct_.poles)
    np.testing.assert_allclose(est_2.ss_ct_.poles, est_3.ss_ct_.poles)
    np.testing.assert_allclose(est_3.ss_ct_.poles, est_1.ss_ct_.poles)
    # Check zeros
    np.testing.assert_allclose(est_1.ss_ct_.zeros, est_2.ss_ct_.zeros)
    np.testing.assert_allclose(est_2.ss_ct_.zeros, est_3.ss_ct_.zeros)
    np.testing.assert_allclose(est_3.ss_ct_.zeros, est_1.ss_ct_.zeros)
    # Check parameters
    assert est_1.n_features_in_ == est_1.hinf_regressor_.n_features_in_
    assert est_1.n_states_in_ == est_1.hinf_regressor_.n_states_in_
    assert est_1.n_inputs_in_ == est_1.hinf_regressor_.n_inputs_in_
    assert est_1.episode_feature_ == est_1.hinf_regressor_.episode_feature_
    np.testing.assert_allclose(est_1.coef_, est_1.hinf_regressor_.coef_)
