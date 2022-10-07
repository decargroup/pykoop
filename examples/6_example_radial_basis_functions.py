"""Example of radial basis functions on a pendulum."""
import scipy.stats
from matplotlib import pyplot as plt

import pykoop


def example_radial_basis_functions() -> None:
    """Demonstrate radial basis functions on a pendulum."""
    # Get example pendulum data
    eg = pykoop.example_data_pendulum()

    # Create pipeline
    kp = pykoop.KoopmanPipeline(
        lifting_functions=[(
            'rbf',
            pykoop.RbfLiftingFn(
                rbf='thin_plate',
                centers=pykoop.QmcCenters(
                    n_centers=100,
                    qmc=scipy.stats.qmc.LatinHypercube,
                    random_state=666,
                ),
            ),
        )],
        regressor=pykoop.Edmd(),
    )

    # Fit the pipeline
    kp.fit(
        eg['X_train'],
        n_inputs=eg['n_inputs'],
        episode_feature=eg['episode_feature'],
    )

    fig, ax = plt.subplots()
    ax.plot(eg['X_train'][:, 1], eg['X_train'][:, 2])
    ax.plot(eg['X_valid'][:, 1], eg['X_valid'][:, 2])
    c = kp.lifting_functions_[0][1].centers_.centers_
    for i in range(c.shape[0]):
        ax.scatter(c[i, 0], c[i, 1], marker='X', color='red', s=100, zorder=10)

    X_pred = kp.predict_trajectory(
        eg['x0_valid'],
        eg['u_valid'],
    )

    fig, ax = plt.subplots()
    ax.plot(eg['X_valid'][:, 1], eg['X_valid'][:, 2])
    ax.plot(X_pred[X_pred[:, 0] == 5, 1], X_pred[X_pred[:, 0] == 5, 2])
    ax.plot(X_pred[X_pred[:, 0] == 25, 1], X_pred[X_pred[:, 0] == 25, 2])
    ax.plot(X_pred[X_pred[:, 0] == 45, 1], X_pred[X_pred[:, 0] == 45, 2])


if __name__ == '__main__':
    example_radial_basis_functions()
    plt.show()
