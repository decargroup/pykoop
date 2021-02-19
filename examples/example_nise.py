import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from pykoop import lmi
from sklearn import compose, pipeline, preprocessing, model_selection


def main():
    """Example from Chapter 3, Problem 26 (p. 154) of
    @book{nise_control_2011,
        author = {Nise, Norman S.},
        title = {Control Systems Engineering},
        year = {2011},
        publisher = {John Wiley & Sons, Inc.},
        edition = {6th},
    }

    x_dot = Ax + Bu,
    y = Cx

    x = [w; q; z; theta],
    y = [z; theta],
    u = [delta_B; delta_S]

    w:          heave velocity
    q:          pitch rate
    z:          submarine depth
    theta:      pitch angle
    delta_B:    bow hydroplane angle
    delta_S:    stren hydroplane angle
    """

    A = np.array([
        [-0.038,  0.896, 0,  0.0015],  # noqa: E201
        [0.0017, -0.092, 0, -0.0056],
        [     1,      0, 0,  -3.086],  # noqa: E201
        [     0,      1, 0,       0],  # noqa: E201
    ])

    B = np.array([
        [-0.0075,  -0.023],
        [ 0.0017, -0.0022],  # noqa: E201
        [      0,       0],  # noqa: E201
        [      0,       0],  # noqa: E201
    ])

    # C = np.array([
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1],
    # ])

    def u(t):
        return np.array([
            np.sin(2 * np.pi * t),
            np.cos(2 * np.pi * t)
        ])

    def ivp_train(t, x):
        return np.ravel(A @ np.reshape(x, (-1, 1))
                        + B @ np.reshape(u(t), (-1, 1)))

    # Simulate dynamics
    t_range = (0, 10)
    t_step = 1e-2
    t = np.arange(*t_range, t_step)
    x0 = np.zeros((4,))
    sol = integrate.solve_ivp(ivp_train, t_range, x0, t_eval=t)

    X_sim = np.vstack((
        sol.y[:, :-1],
        u(sol.t)[:, :-1]
    ))
    Xp_sim = sol.y[:, 1:]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_sim.T, Xp_sim.T, test_size=0.4
    )

    edmd = lmi.LmiEdmd()
    scalerX = preprocessing.StandardScaler()
    scalery = preprocessing.StandardScaler()
    pipeX = pipeline.make_pipeline(scalerX, edmd)
    pipeXy = compose.TransformedTargetRegressor(regressor=pipeX,
                                                transformer=scalery)

    pipeXy.fit(X_train, y_train)

    print(f'Score: {pipeXy.score(X_test, y_test)}')

    cv_scores = model_selection.cross_val_score(pipeXy, X_train, y_train, cv=5)
    print(f'CV Scores: {cv_scores}')

    x = [sol.y[:, 0]]
    u_sim = u(sol.t)
    for k in range(1, sol.t.shape[0]):
        X = np.vstack((
            np.reshape(x[-1], (-1, 1)),
            u_sim[:, [k-1]]
        ))
        x.append(np.ravel(pipeXy.predict(X.T)).T)
    x = np.array(x).T

    fig, ax = plt.subplots(4, 1)
    for k, a in enumerate(np.ravel(ax)):
        a.plot(sol.y[k, :], label='Validation')
        a.plot(x[k, :], label='Koopman')
    ax[0].legend()

    plt.show()


if __name__ == '__main__':
    main()
