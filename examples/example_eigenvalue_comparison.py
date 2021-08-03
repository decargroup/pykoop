"""Example comparing eigenvalues of different versions of EDMD."""

import numpy as np
import pykoop
import pykoop.lmi_regressors
from matplotlib import pyplot as plt
from scipy import integrate, linalg

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


def main() -> None:
    """Compare eigenvalues of different versions of EDMD."""
    # Get example data
    X = pykoop.example_data_msd()

    # Regressor with no constraint
    reg_no_const = pykoop.KoopmanPipeline(
        regressor=pykoop.lmi_regressors.LmiEdmd())
    reg_no_const.fit(X, n_inputs=1, episode_feature=True)
    U_no_const = reg_no_const.regressor_.coef_.T

    # Regressor H-infinity norm constraint of 1.5
    gamma = 1.5
    Xi = np.block([
        [1 / gamma * np.eye(2), np.zeros((2, 1))],
        [np.zeros((1, 2)), -gamma * np.eye(1)],
    ])  # yapf: disable
    reg_diss_const = pykoop.KoopmanPipeline(
        regressor=pykoop.lmi_regressors.LmiEdmdDissipativityConstr(
            supply_rate=Xi))
    reg_diss_const.fit(X, n_inputs=1, episode_feature=True)
    U_diss_const = reg_diss_const.regressor_.coef_.T

    # Regressor with H-infinity regularization
    reg_hinf_reg = pykoop.KoopmanPipeline(
        regressor=pykoop.lmi_regressors.LmiEdmdHinfReg(alpha=5))
    reg_hinf_reg.fit(X, n_inputs=1, episode_feature=True)
    U_hinf_reg = reg_hinf_reg.regressor_.coef_.T

    # Regressor with spectral radius constraint
    reg_sr_const = pykoop.KoopmanPipeline(
        regressor=pykoop.lmi_regressors.LmiEdmdSpectralRadiusConstr(
            spectral_radius=0.8))
    reg_sr_const.fit(X, n_inputs=1, episode_feature=True)
    U_sr_const = reg_sr_const.regressor_.coef_.T

    # Plot results
    fig = plt.figure()  # noqa: F841
    ax = plt.subplot(projection='polar')
    ax.set_xlabel(r'$\mathrm{Re}(\lambda)$')
    ax.set_ylabel(r'$\mathrm{Im}(\lambda)$', labelpad=30)
    plt_eig(U_no_const[:2, :2], ax, 'Unregularized system', marker='o')
    plt_eig(U_diss_const[:2, :2], ax, r'Dissipativity-constrained system')
    plt_eig(U_hinf_reg[:2, :2], ax, r'H-infinity-regularized system')
    plt_eig(U_sr_const[:2, :2], ax,
            r'Spectral-radius-constrained system ($\bar{\rho}=0.8$)')
    ax.set_rmax(1.1)
    ax.legend()
    ax.set_title(r'Eigenvalues of $\bf{A}$')
    plt.show()


def plt_eig(A: np.ndarray,
            ax: plt.Axes,
            label: str = '',
            marker: str = 'x') -> None:
    """Eigenvalue plotting helper function."""
    eigv = linalg.eig(A)[0]
    ax.scatter(np.angle(eigv), np.absolute(eigv), marker=marker, label=label)


if __name__ == '__main__':
    main()
