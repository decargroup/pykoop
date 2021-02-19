import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from pykoop import lmi
from sklearn import compose, pipeline, preprocessing


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

    n_x = A.shape[0]
    # n_u = B.shape[1]
    # n_y = C.shape[0]

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

    # Split data
    y_train, y_valid = np.split(sol.y, 2, axis=1)
    u_train, u_valid = np.split(u(sol.t), 2, axis=1)
    X_train = np.vstack((
        y_train[:, :-1],
        u_train[:, :-1]
    ))
    Xp_train = y_train[:, 1:]
    X_valid = np.vstack((
        y_valid[:, :-1],
        u_valid[:, :-1]
    ))
    # Xp_valid = y_valid[:, 1:]

    edmd = lmi.LmiEdmd()
    scalerX = preprocessing.StandardScaler()
    scalery = preprocessing.StandardScaler()
    pipeX = pipeline.make_pipeline(scalerX, edmd)
    pipeXy = compose.TransformedTargetRegressor(regressor=pipeX,
                                                transformer=scalery)

    pipeXy.fit(X_train.T, Xp_train.T)
    # U = reg.coef_.T
    U = pipeXy.regressor_['lmiedmd'].coef_.T
    # Ad = linalg.expm(A * t_step)

    X_sel = X_valid

    x = [X_sel[:n_x, 0]]
    for k in range(1, X_sel.shape[1]):
        # x.append(np.ravel(
        #     U[:, :n_x] @ np.reshape(x[-1], (-1, 1))
        #     + U[:, n_x:] @ X_sel[n_x:, [k - 1]]
        # ))
        # X = np.vstack((
        #     np.reshape(x[-1], (-1, 1)),
        #     X_sel[n_x:, [k-1]]
        # ))
        # x.append(np.ravel(pipeXy.predict(X.T)).T)
        X = np.vstack((
            np.reshape(x[-1], (-1, 1)),
            X_sel[n_x:, [k-1]]
        ))
        X_scaled = pipeXy.regressor_['standardscaler'].transform(X.T).T
        Xp_scaled = U @ X_scaled
        Xp = pipeXy.transformer_.inverse_transform(Xp_scaled.T).T
        x.append(np.ravel(Xp[:Xp_scaled.shape[0], :]))
    x = np.array(x).T

    fig, ax = plt.subplots(4, 1)
    for k, a in enumerate(np.ravel(ax)):
        a.plot(X_sel[k, :], label='Validation')
        a.plot(x[k, :], label='Koopman')
    ax[0].legend()

    plt.show()


if __name__ == '__main__':
    main()
