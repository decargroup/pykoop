import numpy as np
from scipy import stats, signal
from pykoop import lmi, dmd, koopman_pipeline, lifting_functions
from dynamics import van_der_pol
from matplotlib import pyplot as plt
from sklearn import model_selection
import pandas
import shelve

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


def main():

    ics = [
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([-1, 0]),
        np.array([0, -1]),
        np.array([-1, -1]),
        np.array([1, -1]),
        np.array([-1, 1]),
    ]
    Xs = []
    for (ep, ic) in enumerate(ics):
        Xs.append(sim_vdp(ep, ic))
    Xs = np.hstack(Xs).T

    groups = Xs[:, 0]
    cv = model_selection.GroupShuffleSplit(
        random_state=2233).split(Xs, groups=groups)

    kp = koopman_pipeline.KoopmanPipeline(
        delay=lifting_functions.Delay(n_delay_x=50, n_delay_u=0),
        estimator=lmi.LmiEdmdTikhonovReg(alpha=0),
    )

    params = {
        'estimator__alpha': [10**i for i in range(-9, 3)]
    }
    gs = model_selection.GridSearchCV(kp, params, cv=cv, n_jobs=None, verbose=2)
    gs.fit(Xs, n_u=0)

    results = pandas.DataFrame(gs.cv_results_).sort_values(
        by=['param_estimator__alpha'])
    # col_alpha = results.pop('param_regressor__lmiedmdtikhonovreg__alpha')
    # results.insert(0, 'alpha', col_alpha)
    print(results)


    X_final_valid = sim_vdp(999, np.array([-0.75, 0.5]))[1:, :]
    est = gs.best_estimator_
    n_samp = est.delay_.n_samples_needed_

    with shelve.open('saved-results') as db:
        db['best_estimator'] = est
        db['cv_results'] = results

    X_sim = np.empty(X_final_valid.shape)
    X_sim[:, :n_samp] = X_final_valid[:, :n_samp]
    for k in range(n_samp, X_final_valid.shape[1]):
        Xk = np.vstack((
            np.zeros((1, n_samp)),
            X_sim[:, (k-n_samp):k],
            # u_sim[:, (k-n_samp):k]
        ))
        Xp = est.predict(Xk.T).T
        X_sim[:, [k]] = Xp[1:, [-1]]

    fig, ax = plt.subplots(1, 1)
    ax.plot(X_final_valid[0, :], X_final_valid[1, :], label='Validation')
    ax.plot(X_sim[0, :], X_sim[1, :], label='Koopman')
    ax.set_xlabel(r'$x_1[k]$')
    ax.set_ylabel(r'$x_2[k]$')

    fig, ax = plt.subplots(2, 2, squeeze=False)
    for k, a in enumerate(np.ravel(ax[:, 0])):
        a.plot(X_final_valid[k, :], label='Validation')
        a.plot(X_sim[k, :], label='Koopman')
        a.scatter(n_samp, X_final_valid[k, n_samp], color='C2')
    for k, a in enumerate(np.ravel(ax[:, 1])):
        a.plot(X_final_valid[k, :] - X_sim[k, :])
        a.scatter(n_samp, X_final_valid[k, n_samp] - X_sim[k, n_samp],
                  color='C1')
    ax[0, 0].set_xlabel(r'k')
    ax[0, 1].set_xlabel(r'k')
    ax[1, 0].set_xlabel(r'k')
    ax[1, 1].set_xlabel(r'k')
    ax[0, 0].set_ylabel(r'$x_1[k]$')
    ax[0, 1].set_ylabel(r'$e_1[k]$')
    ax[1, 0].set_ylabel(r'$x_2[k]$')
    ax[1, 1].set_ylabel(r'$e_2[k]$')
    ax[0, 0].legend()

    # fig, ax = plt.subplots(2, 1)
    # for k, a in enumerate(np.ravel(ax)):
    #     a.plot(X_train[k, :] + noise[k, :], label='Training Data')
    # ax[0].set_xlabel(r'k')
    # ax[1].set_xlabel(r'k')
    # ax[0].set_ylabel(r'$x_1[k]$')
    # ax[1].set_ylabel(r'$x_2[k]$')
    # ax[0].legend()

    plt.show()


def sim_vdp(ep, ic):
    # Set up problem
    t_range = (0, 25)
    t_step = 0.1
    vdp = van_der_pol.DiscreteVanDerPol(mu=2, t_step=t_step)

    def u(t):
        return 0 * np.sin(2 * np.pi * t)

    t = np.arange(*t_range, t_step)
    x0 = vdp.x0(ic)
    X_train = np.empty((x0.shape[0], t.shape[0]))
    X_train[:, 0] = x0
    for k in range(1, t.shape[0]):
        X_train[:, k] = vdp.f(t_step * k, X_train[:, k-1], u(t_step * k))

    cov = 0.5
    dist = stats.multivariate_normal(mean=np.zeros((X_train.shape[0],)),
                                     cov=cov*np.eye(X_train.shape[0]),
                                     seed=4321)
    noise = dist.rvs(size=X_train.shape[1]).T
    # noise = np.zeros(X_train.shape)

    # Split the data
    # u_sim = np.reshape(u(t), (1, -1))
    X = np.vstack((
        ep * np.ones((1, t.shape[0]-1)),
        X_train[:, :-1] + noise[:, :-1],
        # u_sim[:, :-1]
    ))

    return X


if __name__ == '__main__':
    main()
