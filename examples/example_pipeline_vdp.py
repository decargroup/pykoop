"""Example of how to use the Koopman pipeline."""

import numpy as np
from matplotlib import pyplot as plt

import pykoop
import pykoop.dynamic_models


def main() -> None:
    """Demonstrate how to use the Koopman pipeline."""
    # Get sample data
    X_vdp = pykoop.example_data_vdp()

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

    # Take last episode for validation
    X_train = X_vdp[X_vdp[:, 0] < 4]
    X_valid = X_vdp[X_vdp[:, 0] == 4]

    # Fit the pipeline
    kp.fit(X_train, n_inputs=1, episode_feature=True)

    # Extract initial conditions and input from validation episode
    x0 = X_valid[[0], 1:3]
    u = X_valid[:, 3:]

    # Predict with re-lifting between timesteps (default)
    X_pred_local = kp.predict_state(
        x0,
        u,
        relift_state=True,
        episode_feature=False,
    )

    # Predict without re-lifting between timesteps
    X_pred_global = kp.predict_state(
        x0,
        u,
        relift_state=False,
        episode_feature=False,
    )

    # Plot trajectories in phase space
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(
        X_valid[:, 1],
        X_valid[:, 2],
        label='True trajectory',
    )
    ax.plot(
        X_pred_local[:, 0],
        X_pred_local[:, 1],
        '--',
        label='Local prediction',
    )
    ax.plot(
        X_pred_global[:, 0],
        X_pred_global[:, 1],
        '--',
        label='Global prediction',
    )
    ax.set_xlabel('$x_1[k]$')
    ax.set_ylabel('$x_2[k]$')
    ax.legend()
    ax.grid(linestyle='--')

    # Lift validation set
    Psi_valid = kp.lift(X_valid[:, 1:], episode_feature=False)

    # Predict lifted state with re-lifting between timesteps (default)
    Psi_pred_local = kp.predict_state(
        x0,
        u,
        relift_state=True,
        return_lifted=True,
        return_input=True,
        episode_feature=False,
    )

    # Predict lifted state without re-lifting between timesteps
    Psi_pred_global = kp.predict_state(
        x0,
        u,
        relift_state=False,
        return_lifted=True,
        return_input=True,
        episode_feature=False,
    )

    fig, ax = plt.subplots(
        kp.n_states_out_,
        1,
        constrained_layout=True,
        sharex=True,
        squeeze=False,
    )
    for i in range(ax.shape[0]):
        ax[i, 0].plot(Psi_valid[:, i], label='True trajectory')
        ax[i, 0].plot(Psi_pred_local[:, i], '--', label='Local prediction')
        ax[i, 0].plot(Psi_pred_global[:, i], '--', label='Global prediction')
        ax[i, 0].grid(linestyle='--')
        ax[i, 0].set_ylabel(rf'$\vartheta_{i + 1}[k]$')

    ax[-1, 0].set_xlabel('$k$')
    ax[0, 0].legend()

    fig, ax = plt.subplots(
        kp.n_inputs_out_,
        1,
        constrained_layout=True,
        sharex=True,
        squeeze=False,
    )
    for i in range(ax.shape[0]):
        j = kp.n_states_out_ + i
        ax[i, 0].plot(Psi_valid[:, j], label='True trajectory')
        ax[i, 0].plot(Psi_pred_local[:, j], '--', label='Local prediction')
        ax[i, 0].plot(Psi_pred_global[:, j], '--', label='Global prediction')
        ax[i, 0].grid(linestyle='--')

    ax[-1, 0].set_xlabel('$k$')
    ax[0, 0].legend()
    ax[0, 0].set_ylabel('$u[k]$')

    plt.show()


if __name__ == '__main__':
    main()
