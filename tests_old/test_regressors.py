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
        (
            pykoop.Edmd(),
            'msd-no-input',
            1e-5,
            1e-5,
            'exact',
        ),
        (
            pykoop.Dmdc(),
            'msd-no-input',
            1e-5,
            1e-5,
            'exact',
        ),
        (
            pykoop.Dmd(),
            'msd-no-input',
            1e-5,
            1e-5,
            'exact',
        ),
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
