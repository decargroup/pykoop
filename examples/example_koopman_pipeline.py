import numpy as np
from scipy import integrate, linalg
from pykoop import dmd, koopman_pipeline, lifting_functions
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

    def u(t):
        return 0.1 * np.sin(t)

    def ivp(t, x):
        return msd.f(t, x, u(t))

    # Solve ODE for training data
    x0 = msd.x0(np.array([0, 0]))
    sol = integrate.solve_ivp(ivp, t_range, x0,
                              t_eval=np.arange(*t_range, t_step),
                              rtol=1e-8, atol=1e-8)

    u_sim = np.reshape(u(sol.t), (1, -1))
    # Split the data
    X = np.vstack((
        sol.y[:, :-1],
        u_sim[:, :-1]
    ))

    kp = koopman_pipeline.KoopmanPipeline(
        delay=lifting_functions.Delay(n_delay_x=1, n_delay_u=1),
        estimator=dmd.Edmd(),
    )
    kp.fit(X.T, n_u=1)

    n_samp = kp.delay_.n_samples_needed_

    X_sim = np.empty(sol.y.shape)
    X_sim[:, :n_samp] = sol.y[:, :n_samp]
    for k in range(n_samp, sol.t.shape[0]):
        X = np.vstack((
            X_sim[:, (k-n_samp):k],
            u_sim[:, (k-n_samp):k]
        ))
        Xp = kp.predict(X.T).T
        X_sim[:, [k]] = Xp

    fig, ax = plt.subplots(2, 2, squeeze=False)
    for k, a in enumerate(np.ravel(ax[:, 0])):
        a.plot(sol.y[k, :], label='Validation')
        a.plot(X_sim[k, :], label='Koopman')
    ax[0, 0].legend()
    for k, a in enumerate(np.ravel(ax[:, 1])):
        a.plot(sol.y[k, :] - X_sim[k, :])
    ax[0, 0].legend()
    plt.show()


if __name__ == '__main__':
    main()
