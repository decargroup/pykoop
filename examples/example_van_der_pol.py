import numpy as np
from pykoop import dmd, koopman_pipeline, lifting_functions
from dynamics import van_der_pol
from matplotlib import pyplot as plt

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


def main():
    # Set up problem
    t_range = (0, 25)
    t_step = 0.1
    vdp = van_der_pol.DiscreteVanDerPol(mu=2, t_step=t_step)

    def u(k):
        return 0.0 * k

    # Solve ODE for training data
    t = np.arange(*t_range, t_step)
    x0 = vdp.x0(np.array([-2, 2]))
    X_train = np.empty((x0.shape[0], t.shape[0]))
    X_train[:, 0] = x0
    for k in range(1, t.shape[0]):
        X_train[:, k] = vdp.f(k, X_train[:, k-1], u(k))

    # Split the data
    u_sim = np.reshape(u(t), (1, -1))
    X = np.vstack((
        X_train[:, :-1],
        u_sim[:, :-1]
    ))

    kp = koopman_pipeline.KoopmanPipeline(
        delay=lifting_functions.Delay(n_delay_x=10, n_delay_u=10),
        estimator=dmd.Edmd(),
    )
    kp.fit(X.T, n_u=1)

    n_samp = kp.delay_.n_samples_needed_

    X_sim = np.empty(X_train.shape)
    X_sim[:, :n_samp] = X_train[:, :n_samp]
    for k in range(n_samp, t.shape[0]):
        Xk = np.vstack((
            X_sim[:, (k-n_samp):k],
            u_sim[:, (k-n_samp):k]
        ))
        Xp = kp.predict(Xk.T).T
        X_sim[:, [k]] = Xp

    fig, ax = plt.subplots(1, 1)
    ax.plot(X_train[0, :], X_train[1, :], label='Validation')
    ax.plot(X_sim[0, :], X_sim[1, :], label='Koopman')
    ax.set_xlabel(r'$x_1[k]$')
    ax.set_ylabel(r'$x_2[k]$')

    fig, ax = plt.subplots(2, 2, squeeze=False)
    for k, a in enumerate(np.ravel(ax[:, 0])):
        a.plot(X_train[k, :], label='Validation')
        a.plot(X_sim[k, :], label='Koopman')
        a.scatter(n_samp, X_train[k, n_samp], color='C2')
    for k, a in enumerate(np.ravel(ax[:, 1])):
        a.plot(X_train[k, :] - X_sim[k, :])
        a.scatter(n_samp, X_train[k, n_samp] - X_sim[k, n_samp], color='C1')
    ax[0, 0].set_xlabel(r'k')
    ax[0, 1].set_xlabel(r'k')
    ax[1, 0].set_xlabel(r'k')
    ax[1, 1].set_xlabel(r'k')
    ax[0, 0].set_ylabel(r'$x_1[k]$')
    ax[0, 1].set_ylabel(r'$e_1[k]$')
    ax[1, 0].set_ylabel(r'$x_2[k]$')
    ax[1, 1].set_ylabel(r'$e_2[k]$')
    ax[0, 0].legend()
    plt.show()


if __name__ == '__main__':
    main()
