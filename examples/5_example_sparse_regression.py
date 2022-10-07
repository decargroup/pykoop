"""Example of sparse regression with :class:`EdmdMeta`."""

from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso

import pykoop

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


def example_sparse_regression() -> None:
    """Demonstrate sparse regression with :class:`EdmdMeta`."""
    # Get example mass-spring-damper data
    eg = pykoop.example_data_msd()

    # Choose lifting functions
    poly3 = [('pl', pykoop.PolynomialLiftingFn(order=3))]

    # Create dense pipeline
    kp_dense = pykoop.KoopmanPipeline(
        lifting_functions=poly3,
        regressor=pykoop.Edmd(alpha=1e-9),
    )
    # Create sparse pipeline
    kp_sparse = pykoop.KoopmanPipeline(
        lifting_functions=poly3,
        # Also try out :class:`sklearn.linear_model.OrthogonalMatchingPursuit`!
        regressor=pykoop.EdmdMeta(regressor=Lasso(alpha=1e-9)),
    )

    # Fit the pipelines
    kp_dense.fit(
        eg['X_train'],
        n_inputs=eg['n_inputs'],
        episode_feature=eg['episode_feature'],
    )
    kp_sparse.fit(
        eg['X_train'],
        n_inputs=eg['n_inputs'],
        episode_feature=eg['episode_feature'],
    )

    # Plot dense prediction and Koopman matrix
    fig, ax = kp_dense.plot_predicted_trajectory(
        eg['X_valid'],
        plot_input=True,
    )
    ax[0, 0].set_title('Dense regression')
    fig, ax = kp_dense.plot_koopman_matrix()
    ax.set_title('Dense regression')

    # Plot sparse prediction and Koopman matrix
    fig, ax = kp_sparse.plot_predicted_trajectory(
        eg['X_valid'],
        plot_input=True,
    )
    ax[0, 0].set_title('Sparse regression')
    fig, ax = kp_sparse.plot_koopman_matrix()
    ax.set_title('Sparse regression')


if __name__ == '__main__':
    example_sparse_regression()
    plt.show()
