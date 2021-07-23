import logging

import numpy as np
from dynamics import mass_spring_damper
from matplotlib import pyplot as plt
from pykoop import lmi_regressors
from scipy import integrate, linalg

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


def main():
    # logger = logging.getLogger('lmi')
    # logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)

    # Set up problem
    t_range = (0, 5)
    t_step = 0.1
    msd = mass_spring_damper.MassSpringDamper(mass=0.5,
                                              stiffness=0.7,
                                              damping=0.6)

    def u(t):
        return 0.1 * np.sin(t)

    def ivp(t, x):
        return msd.f(t, x, u(t))

    # Solve ODE for training data
    x0 = msd.x0(np.array([0, 0]))
    sol = integrate.solve_ivp(ivp,
                              t_range,
                              x0,
                              t_eval=np.arange(*t_range, t_step),
                              rtol=1e-8,
                              atol=1e-8)

    # Split the data
    X = np.vstack((sol.y[:, :-1], np.reshape(u(sol.t), (1, -1))[:, :-1]))
    Xp = sol.y[:, 1:]

    # Regressor with no constraint
    reg_no_const = lmi_regressors.LmiEdmd(alpha=0)
    reg_no_const.fit(X.T, Xp.T)
    U_no_const = reg_no_const.coef_.T

    # Regressor with Hinf regularization
    reg_hinf_old = lmi_regressors.LmiEdmdHinfReg(alpha=0.1)
    reg_hinf_old.fit(X.T, Xp.T)
    U_hinf_old = reg_hinf_old.coef_.T

    # Regressor with Hinf regularization
    reg_hinf = lmi_regressors.LmiDmdcHinfReg(alpha=0.1,
                                             tsvd_method=('manual', 2, 2))
    reg_hinf.fit(X.T, Xp.T)
    U_hinf = reg_hinf.coef_.T

    # Plot results
    fig = plt.figure()  # noqa: F841
    ax = plt.subplot(projection='polar')
    ax.set_xlabel(r'$\mathrm{Re}(\lambda)$')
    ax.set_ylabel(r'$\mathrm{Im}(\lambda)$')
    plt_eig(U_no_const[:2, :2], ax, 'True system', marker='o')
    plt_eig(U_hinf_old[:2, :2], ax, r'Hinf-regularized EDMD')
    plt_eig(U_hinf[:2, :2], ax, r'Hinf-regularized DMDc')
    ax.set_rmax(1.1)
    ax.legend()
    plt.show()


def plt_eig(A, ax, label='', marker='x'):
    """Eigenvalue plotting helper function."""
    eigv = linalg.eig(A)[0]
    ax.scatter(np.angle(eigv), np.absolute(eigv), marker=marker, label=label)


if __name__ == '__main__':
    main()
