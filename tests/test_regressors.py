import pytest
import numpy as np
from scipy import integrate, linalg
from pykoop import dmd, lmi
from dynamics import mass_spring_damper
from sklearn import linear_model


@pytest.fixture(params=[
    ('msd', dmd.Edmd(), 1e-5, 1e-5, 'exact'),
    ('msd', lmi.LmiEdmd(inv_method='eig'), 1e-4, 1e-5, 'exact'),
    ('msd', lmi.LmiEdmd(inv_method='inv'), 1e-4, 1e-5, 'exact'),
    ('msd', lmi.LmiEdmd(inv_method='ldl'), 1e-4, 1e-5, 'exact'),
    ('msd', lmi.LmiEdmd(inv_method='chol'), 1e-4, 1e-5, 'exact'),
    ('msd', lmi.LmiEdmd(inv_method='sqrt'), 1e-4, 1e-5, 'exact'),
    (
        'msd',
        lmi.LmiEdmdTikhonovReg(inv_method='chol', alpha=1),
        1e-4,
        None,
        'sklearn-ridge'
    ),
    (
        'msd',
        lmi.LmiEdmdTwoNormReg(inv_method='chol', alpha=1),
        1e-4,
        None,
        # Test vector generated from old code. More of a regression test than
        # anything. If the same error is present in that old code and this
        # code, this test is meaningless!
        np.array([
            [ 0.89995985, 0.07048035],  # noqa: E201
            [-0.07904385, 0.89377084]
        ])
    ),
    (
        'msd',
        lmi.LmiEdmdNuclearNormReg(inv_method='chol', alpha=1),
        1e-4,
        None,
        # Test vector generated from old code. More of a regression test than
        # anything. If the same error is present in that old code and this
        # code, this test is meaningless!
        np.array([
            [ 0.70623152, -0.17749238],  # noqa: E201
            [-0.32354638,  0.50687639]
        ])
    ),
    # Test cases marked as slow can be deselected easily with `pytest -k-slow`
    pytest.param((
        'msd',
        lmi.LmiEdmdSpectralRadiusConstr(inv_method='chol', rho_bar=1.1),
        1e-4,
        None,
        # Since the constraint is larger than the actual eigenvalue magnitudes,
        # it will have no effect and we can compare to the exact solution.
        'exact'
    ), marks=pytest.mark.slow),
    pytest.param((
        'msd',
        lmi.LmiEdmdSpectralRadiusConstr(inv_method='chol', rho_bar=0.8),
        1e-4,
        None,
        # Regression test generated from this code. Result was manually
        # checked (eigenvalue magnitudes are less than 0.8) but strictly
        # speaking, it hasn't been checked against other code.
        np.array([
            [ 0.88994802, 0.04260765],  # noqa: E201
            [-0.22883601, 0.70816555]
        ])
    ), marks=pytest.mark.slow),
], ids=[
    "msd-dmd.Edmd()",
    "msd-lmi.LmiEdmd(inv_method='eig')",
    "msd-lmi.LmiEdmd(inv_method='inv')",
    "msd-lmi.LmiEdmd(inv_method='ldl')",
    "msd-lmi.LmiEdmd(inv_method='chol')",
    "msd-lmi.LmiEdmd(inv_method='sqrt')",
    "msd-lmi.LmiEdmdTikhonovReg(inv_method='chol', alpha=1)",
    "msd-lmi.LmiEdmdTwoNormReg(inv_method='chol', alpha=1)",
    "msd-lmi.LmiEdmdNuclearNormReg(inv_method='chol', alpha=1)",
    "msd-lmi.LmiEdmdSpectralRadiusConstr(inv_method='chol', rho_bar=1.1)",
    "msd-lmi.LmiEdmdSpectralRadiusConstr(inv_method='chol', rho_bar=0.8)",
])
def scenario(request):
    system, regressor, fit_tol, predict_tol, soln = request.param
    # Simulate or load data
    # Not all systems and solutions are compatible.
    # For `exact` to work, `t_step`, `A`, and `y` must be defined.
    # For `ridge` to work, only `y` is needed
    if system == 'msd':
        # Set up problem
        t_range = (0, 10)
        t_step = 0.1
        msd = mass_spring_damper.MassSpringDamper(0.5, 0.7, 0.6)
        # Solve ODE for training data
        x0 = msd.x0(np.array([1, 0]))
        sol = integrate.solve_ivp(lambda t, x: msd.f(t, x, 0), t_range, x0,
                                  t_eval=np.arange(*t_range, t_step),
                                  rtol=1e-8, atol=1e-8)
        A = msd._A
        y = sol.y
    # Split the data
    y_train, y_valid = np.split(y, 2, axis=1)
    X_train = y_train[:, :-1]
    Xp_train = y_train[:, 1:]
    X_valid = y_valid[:, :-1]
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
    else:
        U_valid = None
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
    np.testing.assert_allclose(scenario['X_train'][:, 1:],
                               scenario['Xp_train'][:, :-1])
    np.testing.assert_allclose(scenario['X_valid'][:, 1:],
                               scenario['Xp_valid'][:, :-1])


def test_fit(scenario):
    if scenario['fit_tol'] is None:
        pytest.skip()
    # Fit regressor
    scenario['regressor'].fit(scenario['X_train'].T, scenario['Xp_train'].T)
    # Test value of Koopman operator
    np.testing.assert_allclose(
        scenario['regressor'].coef_.T,
        scenario['U_valid'],
        atol=scenario['fit_tol'],
        rtol=0
    )


def test_predict(scenario):
    if scenario['predict_tol'] is None:
        pytest.skip()
    # Fit regressor
    scenario['regressor'].fit(scenario['X_train'].T, scenario['Xp_train'].T)
    # Test prediction
    np.testing.assert_allclose(
        scenario['regressor'].predict(scenario['X_valid'].T).T,
        scenario['Xp_valid'],
        atol=scenario['predict_tol'],
        rtol=0
    )
