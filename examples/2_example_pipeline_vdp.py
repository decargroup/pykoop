"""Example of how to use the Koopman pipeline."""

import numpy as np
from matplotlib import pyplot as plt

import pykoop

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


def example_pipeline_vdp() -> None:
    """Demonstrate how to use the Koopman pipeline."""
    # Get example Van der Pol data
    eg = pykoop.example_data_vdp()

    # Create pipeline
    kp = pykoop.KoopmanPipeline(
        lifting_functions=[(
            'sp',
            pykoop.SplitPipeline(
                lifting_functions_state=[
                    ('pl', pykoop.PolynomialLiftingFn(order=3))
                ],
                lifting_functions_input=None,
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

    # Predict with re-lifting between timesteps (default)
    X_pred_local = kp.predict_trajectory(
        eg['x0_valid'],
        eg['u_valid'],
        relift_state=True,
    )

    # Predict without re-lifting between timesteps
    X_pred_global = kp.predict_trajectory(
        eg['x0_valid'],
        eg['u_valid'],
        relift_state=False,
    )

    # Plot trajectories in phase space
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 6))
    ax.plot(
        eg['X_valid'][:, 1],
        eg['X_valid'][:, 2],
        label='True trajectory',
    )
    ax.plot(
        X_pred_local[:, 1],
        X_pred_local[:, 2],
        '--',
        label='Local prediction',
    )
    ax.plot(
        X_pred_global[:, 1],
        X_pred_global[:, 2],
        '--',
        label='Global prediction',
    )
    ax.set_xlabel('$x_1[k]$')
    ax.set_ylabel('$x_2[k]$')
    ax.legend()
    ax.set_title('True and predicted phase-space trajectories')

    # Lift validation set
    Psi_valid = kp.lift(eg['X_valid'])

    # Predict lifted state with re-lifting between timesteps (default)
    Psi_pred_local = kp.predict_trajectory(
        eg['x0_valid'],
        eg['u_valid'],
        relift_state=True,
        return_lifted=True,
        return_input=True,
    )

    # Predict lifted state without re-lifting between timesteps
    Psi_pred_global = kp.predict_trajectory(
        eg['x0_valid'],
        eg['u_valid'],
        relift_state=False,
        return_lifted=True,
        return_input=True,
    )

    # Get feature names
    names = kp.get_feature_names_out(format='latex')

    fig, ax = plt.subplots(
        kp.n_states_out_,
        1,
        constrained_layout=True,
        sharex=True,
        squeeze=False,
        figsize=(6, 12),
    )
    for i in range(ax.shape[0]):
        ax[i, 0].plot(
            Psi_valid[:, i + 1],
            label='True trajectory',
        )
        ax[i, 0].plot(
            Psi_pred_local[:, i + 1],
            '--',
            label='Local prediction',
        )
        ax[i, 0].plot(
            Psi_pred_global[:, i + 1],
            '--',
            label='Global prediction',
        )
        ax[i, 0].set_ylabel(rf'$\vartheta_{i + 1} = {names[i + 1]}$')

    ax[-1, 0].set_xlabel('$k$')
    ax[0, 0].set_title('True and predicted lifted states')
    ax[-1, -1].legend(loc='lower right')
    fig.align_ylabels()


if __name__ == '__main__':
    example_pipeline_vdp()
    plt.show()
