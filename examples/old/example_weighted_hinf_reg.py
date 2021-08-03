import numpy as np
from scipy import integrate, linalg, signal, stats
from pykoop import lmi, util, koopman_pipeline, lifting_functions, dmd
from dynamics import mass_spring_damper
from matplotlib import pyplot as plt
import logging
import control
from sklearn import preprocessing

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


def main():
    # logging.basicConfig(level=logging.WARNING)
    logging.basicConfig(level=logging.INFO)

    n_ep = 10
    t_range = (0, 5)
    t_step = 1e-2
    cov = 0.01
    msd = mass_spring_damper.MassSpringDamper(
        mass=0.5,
        stiffness=2,
        damping=0.6,
    )
    rng = np.random.default_rng(1234)
    n_u = 1

    X_ep = []
    X_ep_noise = []
    for i in range(n_ep):
        # Create input
        u = util.random_input(
            t_range,
            t_step,
            np.array([-10]),
            np.array([10]),
            0.1,
            rng=rng,
        )
        # Set initial condition
        x0 = msd.x0(
            util.random_state(
                -np.ones((2, )),
                np.ones((2, )),
                rng=rng,
            ))
        sol = integrate.solve_ivp(
            lambda t, x: msd.f(t, x, u(t)),
            t_range,
            x0,
            t_eval=np.arange(*t_range, t_step),
            rtol=1e-8,
            atol=1e-8,
        )
        X_ep.append(np.vstack((
            i * np.ones_like(sol.t),
            sol.y,
            u(sol.t),
        )))
    X = np.hstack(X_ep)

    hp = {
        'max_iter': 100,
        'picos_eps': 1e-3,
        'tol': 1e-12,
        'alpha': 1,
        'ratio': 1,
        'inv_method': 'svd',
        'solver_params': {
            'solver': 'mosek',
            'dualize': True,
            'verbosity': 0,
            'mosek_params': {
                'MSK_IPAR_NUM_THREADS': 4,
            },
            '*_fsb_tol': 1e-4,
            '*_opt_tol': 1e-4,
        }
    }

    dl = 0
    ord = 1

    kp_edmd = koopman_pipeline.KoopmanPipeline(
        preprocessing=preprocessing.StandardScaler(),
        delay=lifting_functions.Delay(n_delay_x=dl, n_delay_u=dl),
        lifting_function=lifting_functions.PolynomialLiftingFn(order=ord),
        estimator=dmd.Edmd(),
    )
    kp_hinf = koopman_pipeline.KoopmanPipeline(
        preprocessing=preprocessing.StandardScaler(),
        delay=lifting_functions.Delay(n_delay_x=dl, n_delay_u=dl),
        lifting_function=lifting_functions.PolynomialLiftingFn(order=ord),
        estimator=lmi.LmiEdmdHinfReg(**hp),
    )
    kp_hinfw = koopman_pipeline.KoopmanPipeline(
        preprocessing=preprocessing.StandardScaler(),
        delay=lifting_functions.Delay(n_delay_x=dl, n_delay_u=dl),
        lifting_function=lifting_functions.PolynomialLiftingFn(order=ord),
        estimator=lmi.LmiEdmdHinfReg(**hp),
    )

    kp_edmd.fit(X.T, n_u=n_u)

    # Set up weights
    fs = 1 / t_step
    n = msd._B.shape[0]
    # b, a = signal.butter(
    #     N=2,
    #     Wn=10,
    #     btype='lowpass',
    #     analog=True,
    #     output='ba',
    # )
    # sys = signal.ZerosPolesGain([], [-45], 1).to_tf()
    # b = sys.num
    # a = sys.den
    # fig, ax = plt.subplots(2, 1)
    # w, mag, phase = signal.bode(sys, np.logspace(-3, 3, 100))
    # ax[0].semilogx(w, mag)
    # ax[1].semilogx(w, phase)
    # plt.show()
    # exit()
    # b0 = np.array([0])
    # a0 = np.array([1])
    # num = [[b for i_u in range(n)] for i_y in range(n)]
    # den = [[a for i_u in range(n)] for i_y in range(n)]
    # num = [
    #     [b, b0],
    #     [b0, b]
    # ]
    # den = [
    #     [a, a0],
    #     [a0, a]
    # ]
    # tf = control.TransferFunction(num, den)
    # ss = control.tf2ss(tf).sample(t_step, method='bilinear')
    # weights = (ss.A, ss.B, ss.C, ss.D)
    # b = np.array([1])
    # a = np.array([1, 45])
    # tf = control.TransferFunction(b, a)
    # ss = control.tf2ss(tf).sample(t_step)
    zpk = signal.ZerosPolesGain([0], [0.99], 100, dt=t_step)
    ss = zpk.to_ss()
    # A = linalg.block_diag(ss.A, ss.A)
    # B = linalg.block_diag(ss.B, ss.B)
    # C = linalg.block_diag(ss.C, ss.C)
    # D = linalg.block_diag(ss.D, ss.D)
    # weights = (A, B, C, D)
    weights = (ss.A, ss.B, ss.C, ss.D)

    kp_hinfw.fit(X.T, n_u=n_u, weights=weights)
    if 'Unable' in kp_hinfw.estimator_.stop_reason_:
        exit()

    kp_hinf.fit(X.T, n_u=n_u)

    # Plot results
    fig = plt.figure()  # noqa: F841
    ax = plt.subplot(projection='polar')
    # ax = plt.subplot()
    ax.set_xlabel(r'$\mathrm{Re}(\lambda)$')
    ax.set_ylabel(r'$\mathrm{Im}(\lambda)$')
    sh = kp_edmd.estimator_.coef_.T.shape[0]
    plt_eig(kp_edmd.estimator_.coef_.T[:, :sh], ax, 'True', marker='o')
    # plt_eig(kp_edmd_n.estimator_.coef_.T[:, :sh], ax, 'True, noise', marker='o')
    plt_eig(kp_hinfw.estimator_.coef_.T[:, :sh], ax, 'Weighted Hinf', marker='x')
    plt_eig(kp_hinf.estimator_.coef_.T[:, :sh], ax, 'Hinf', marker='.')
    ax.set_rmax(1.1)
    ax.legend()


    comb = np.hstack((
        kp_hinfw.estimator_.coef_.T,
        kp_hinf.estimator_.coef_.T,
    ))
    maxx = np.max(np.abs(comb))
    param = {
        'vmax': maxx,
        'vmin': -maxx,
        'cmap': 'seismic',
    }

    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(kp_hinfw.estimator_.coef_.T, **param)
    ax[1].imshow(kp_hinf.estimator_.coef_.T, **param)

    plt.show()


def plt_eig(A, ax, label='', marker='x'):
    """Eigenvalue plotting helper function."""
    eigv = linalg.eig(A)[0]
    ax.scatter(np.angle(eigv), np.absolute(eigv), marker=marker, label=label)


if __name__ == '__main__':
    main()
