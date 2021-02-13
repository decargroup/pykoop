import pytest
import numpy as np
from scipy import integrate, linalg
import pykoop.dmd
import pykoop.lmi
from dynamics import mass_spring_damper


@pytest.fixture
def msd():
    # Set up problem
    t_range = (0, 10)
    t_step = 0.1
    msd = mass_spring_damper.MassSpringDamper(0.5, 0.7, 0.6)
    # Solve ODE for training data
    x0 = msd.x0(np.array([1, 0]))
    sol = integrate.solve_ivp(lambda t, x: msd.f(t, x, 0), t_range, x0,
                              t_eval=np.arange(*t_range, t_step),
                              rtol=1e-8, atol=1e-8)
    # Calculate discrete-time A matrix
    Ad = linalg.expm(msd._A * t_step)
    # Split the data
    y_train, y_valid = np.split(sol.y, 2, axis=1)
    X_train = y_train[:, :-1]
    Xp_train = y_train[:, 1:]
    X_valid = y_valid[:, :-1]
    Xp_valid = y_valid[:, 1:]
    return {
        'X_train':  X_train,
        'Xp_train': Xp_train,
        'X_valid':  X_valid,
        'Xp_valid': Xp_valid,
        'Ad':       Ad,
    }


def test_msd_data(msd):
    # Make sure training and validation sets are not the same
    assert not np.allclose(msd['X_train'], msd['X_valid'])
    assert not np.allclose(msd['Xp_train'], msd['Xp_valid'])
    # Make sure Xp is time-shifted version of X
    assert np.allclose(msd['X_train'][:, 1:], msd['Xp_train'][:, :-1])
    assert np.allclose(msd['X_valid'][:, 1:], msd['Xp_valid'][:, :-1])


@pytest.mark.parametrize('reg,rtol,atol', [
    (pykoop.dmd.Edmd(), 1e-5, 1e-8),
    (pykoop.lmi.LmiEdmd(inv_method='eig'), 1e-4, 1e-8),
    (pykoop.lmi.LmiEdmd(inv_method='inv'), 1e-3, 1e-8),
    (pykoop.lmi.LmiEdmd(inv_method='ldl'), 1e-4, 1e-8),
    (pykoop.lmi.LmiEdmd(inv_method='chol'), 1e-4, 1e-8),
    (pykoop.lmi.LmiEdmd(inv_method='sqrt'), 1e-4, 1e-8),
], ids=[
    "Edmd()",
    "LmiEdmd(inv_method='eig')",
    "LmiEdmd(inv_method='inv')",
    "LmiEdmd(inv_method='ldl')",
    "LmiEdmd(inv_method='chol')",
    "LmiEdmd(inv_method='sqrt')",
])
def test_msd_fit(msd, reg, rtol, atol):
    # Fit regressor
    reg.fit(msd['X_train'].T, msd['Xp_train'].T)
    # Test value of Koopman operator
    U_fit = reg.U_.T
    assert np.allclose(msd['Ad'], U_fit, rtol, atol)


@pytest.mark.parametrize('reg,rtol,atol', [
    (pykoop.dmd.Edmd(), 1e-5, 1e-8),
    (pykoop.lmi.LmiEdmd(inv_method='eig'), 1e-3, 1e-8),
    (pykoop.lmi.LmiEdmd(inv_method='inv'), 1e-2, 1e-8),
    (pykoop.lmi.LmiEdmd(inv_method='ldl'), 1e-3, 1e-8),
    (pykoop.lmi.LmiEdmd(inv_method='chol'), 1e-3, 1e-8),
    (pykoop.lmi.LmiEdmd(inv_method='sqrt'), 1e-3, 1e-8),
], ids=[
    "Edmd()",
    "LmiEdmd(inv_method='eig')",
    "LmiEdmd(inv_method='inv')",
    "LmiEdmd(inv_method='ldl')",
    "LmiEdmd(inv_method='chol')",
    "LmiEdmd(inv_method='sqrt')",
])
def test_msd_predict(msd, reg, rtol, atol):
    # Fit regressor
    reg.fit(msd['X_train'].T, msd['Xp_train'].T)
    # Test prediction
    Xp_pred = reg.predict(msd['X_valid'].T).T
    assert np.allclose(msd['Xp_valid'], Xp_pred, rtol, atol)
