import numpy as np
import pykoop
from dynamics import mass_spring_damper
from matplotlib import pyplot as plt
from pykoop import lmi_regressors
from scipy import integrate

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


def main():
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

    u_sim = np.reshape(u(sol.t), (1, -1))
    # Split the data
    X = np.vstack((
        np.zeros((1, sol.t.shape[0] - 1)),
        sol.y[:, :-1],
        u_sim[:, :-1],
    ))

    kp = pykoop.KoopmanPipeline(
        preprocessors=None,
        # lifting_functions=[
        #     pykoop.DelayLiftingFn(n_delays_state=4, n_delays_input=4),
        # ],
        lifting_functions=None,
        # regressor=lmi_regressors.LmiEdmd(tsvd_method=('manual', 4)),
        # regressor=lmi_regressors.LmiDmdc(tsvd_method=('known_noise', 0.1)),
        regressor=lmi_regressors.LmiDmdc(alpha=1e-3, reg_method='tikhonov'),
    )
    kp.fit(X.T, n_inputs=1, episode_feature=True)

    ns = None
    sc = 'neg_mean_squared_error'
    ms = True
    s1 = pykoop.KoopmanPipeline.make_scorer(
        discount_factor=1,
        n_steps=ns,
        regression_metric=sc,
        multistep=ms,
    )
    s2 = pykoop.KoopmanPipeline.make_scorer(
        discount_factor=0.99,
        n_steps=ns,
        regression_metric=sc,
        multistep=ms,
    )
    s3 = pykoop.KoopmanPipeline.make_scorer(
        discount_factor=0,
        n_steps=ns,
        regression_metric=sc,
        multistep=ms,
    )
    print(s1(kp, X.T))
    print(s2(kp, X.T))
    print(s3(kp, X.T))
    print(kp.score(X.T))

    n_samp = kp.min_samples_

    Xx = np.empty(sol.y.shape)
    Xx[:, :n_samp] = sol.y[:, :n_samp]
    Xxx = np.vstack((
        np.zeros_like(u_sim),
        Xx,
        u_sim,
    ))
    X_sim2 = kp.predict_multistep(Xxx.T).T[1:, :]

    X_sim = np.empty(sol.y.shape)
    X_sim[:, :n_samp] = sol.y[:, :n_samp]
    for k in range(n_samp, sol.t.shape[0]):
        X = np.vstack((
            np.zeros((1, n_samp)),
            X_sim[:, (k - n_samp):k],
            u_sim[:, (k - n_samp):k],
        ))
        Xp = kp.predict(X.T).T
        X_sim[:, [k]] = Xp[1:, [-1]]

    fig, ax = plt.subplots(2, 2, squeeze=False)
    for k, a in enumerate(np.ravel(ax[:, 0])):
        a.plot(sol.y[k, :], label='Validation')
        a.plot(X_sim[k, :], label='Koopman')
        a.plot(X_sim2[k, :], label='Koopman2')
    for k, a in enumerate(np.ravel(ax[:, 1])):
        a.plot(sol.y[k, :] - X_sim[k, :])
        a.plot(sol.y[k, :] - X_sim2[k, :])
    ax[0, 0].legend()
    plt.show()


if __name__ == '__main__':
    main()
