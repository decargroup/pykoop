import numpy as np
from scipy import integrate, linalg
from pykoop import lmi
from dynamics import mass_spring_damper
from matplotlib import pyplot as plt

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


def main():
    # Set up problem
    t_range = (0, 5)
    t_step = 0.1
    msd = mass_spring_damper.MassSpringDamper(
        mass=0.5,
        stiffness=0.7,
        damping=0.6
    )
    # Solve ODE for training data
    x0 = msd.x0(np.array([1, 0]))
    sol = integrate.solve_ivp(lambda t, x: msd.f(t, x, 0), t_range, x0,
                              t_eval=np.arange(*t_range, t_step),
                              rtol=1e-8, atol=1e-8)
    # Split the data
    X = sol.y[:, :-1]
    Xp = sol.y[:, 1:]

    # Regressor with no constraint
    reg_no_const = lmi.LmiEdmdTikhonovReg(alpha=0)

    # Can call _calc_G_H to preview G and H and their condition numbers.
    # Might be helpful to make sure your data isn't "bad"
    a = lmi._calc_G_H(X.T, Xp.T, 1)
    print(a)

    reg_no_const.fit(X.T, Xp.T)
    U_no_const = reg_no_const.coef_.T
    # Regressor with constraint larger than actual spectral radius.
    # Should not have any effect on the problem.
    reg_big_const = lmi.LmiEdmdSpectralRadiusConstr(rho_bar=1.1)
    reg_big_const.fit(X.T, Xp.T)
    U_big_const = reg_big_const.coef_.T
    # Regressor with significant constraint on spectral radius.
    # Will push eigenvalues toward centre of unit circle.
    reg_small_const = lmi.LmiEdmdSpectralRadiusConstr(rho_bar=0.8)
    reg_small_const.fit(X.T, Xp.T)
    U_small_const = reg_small_const.coef_.T

    # Plot results
    fig = plt.figure()  # noqa: F841
    ax = plt.subplot(projection='polar')
    ax.set_xlabel(r'$\mathrm{Re}(\lambda)$')
    ax.set_ylabel(r'$\mathrm{Im}(\lambda)$')
    plt_eig(U_no_const, ax, 'True system', marker='o')
    plt_eig(U_big_const, ax, r'Constrained, $\bar{\rho}=1.1$')
    plt_eig(U_small_const, ax, r'Constrained, $\bar{\rho}=0.8$')
    ax.set_rmax(1.1)
    ax.legend()
    plt.show()


def plt_eig(A, ax, label='', marker='x'):
    """Eigenvalue plotting helper function."""
    eigv = linalg.eig(A)[0]
    ax.scatter(np.angle(eigv), np.absolute(eigv), marker=marker, label=label)


if __name__ == '__main__':
    main()
