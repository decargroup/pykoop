"""Example of how to use the Koopman pipeline."""

from matplotlib import pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

import pykoop

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


def example_pipeline_simple() -> None:
    """Demonstrate how to use the Koopman pipeline."""
    # Get example mass-spring-damper data
    eg = pykoop.example_data_msd()

    # Create pipeline
    kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            ('ma', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
            ('pl', pykoop.PolynomialLiftingFn(order=2)),
            ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ],
        regressor=pykoop.Edmd(alpha=1),
    )

    # Fit the pipeline
    kp.fit(
        eg['X_train'],
        n_inputs=eg['n_inputs'],
        episode_feature=eg['episode_feature'],
    )

    # Predict using the pipeline
    X_pred = kp.predict_trajectory(eg['x0_valid'], eg['u_valid'])

    # Score using the pipeline
    score = kp.score(eg['X_valid'])

    # Plot results
    fig, ax = plt.subplots(
        kp.n_states_in_ + kp.n_inputs_in_,
        1,
        constrained_layout=True,
        sharex=True,
        figsize=(6, 6),
    )
    # Plot true trajectory
    ax[0].plot(eg['t'], eg['X_valid'][:, 1], label='True trajectory')
    ax[1].plot(eg['t'], eg['X_valid'][:, 2])
    ax[2].plot(eg['t'], eg['X_valid'][:, 3])
    # Plot predicted trajectory
    ax[0].plot(eg['t'], X_pred[:, 1], '--', label='Predicted trajectory')
    ax[1].plot(eg['t'], X_pred[:, 2], '--')
    # Add labels
    ax[-1].set_xlabel('$t$')
    ax[0].set_ylabel('$x(t)$')
    ax[1].set_ylabel(r'$\dot{x}(t)$')
    ax[2].set_ylabel('$u$')
    ax[0].set_title(f'True and predicted states; MSE={-1 * score:.2e}')
    ax[0].legend(loc='upper right')


if __name__ == '__main__':
    example_pipeline_simple()
    plt.show()
