"""Create a Koopman pipeline diagram.

Generates figures needed to compile ``pykoop_diagram.tex``.
"""

import pathlib

import pykoop
from matplotlib import pyplot as plt

# Okabe-Ito colorscheme: https://jfly.uni-koeln.de/color/
OI = {
    'black': (0.00, 0.00, 0.00),
    'orange': (0.90, 0.60, 0.00),
    'sky blue': (0.35, 0.70, 0.90),
    'bluish green': (0.00, 0.60, 0.50),
    'yellow': (0.95, 0.90, 0.25),
    'blue': (0.00, 0.45, 0.70),
    'vermillion': (0.80, 0.40, 0.00),
    'reddish purple': (0.80, 0.60, 0.70),
    'grey': (0.60, 0.60, 0.60),
}


def main():
    """Create a Koopman pipeline diagram."""
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
        regressor=pykoop.Edmd(alpha=2),
    )
    # Fit the pipeline
    kp.fit(
        eg['X_train'],
        n_inputs=eg['n_inputs'],
        episode_feature=eg['episode_feature'],
    )
    # Predict unseen trajectory
    X_pred = kp.predict_trajectory(eg['x0_valid'], eg['u_valid'])
    # Make sure example has episode feature
    assert eg['episode_feature'], 'Assumes the example has an episode feature.'
    # Path to save figures
    path = pathlib.Path('./')
    # Extract training, validation, and predicted episodes
    ep_t = eg['X_train'][0, 0]
    X_train_ep = eg['X_train'][eg['X_train'][:, 0] == ep_t, 1:]
    ep_v = eg['X_valid'][0, 0]
    X_valid_ep = eg['X_valid'][eg['X_valid'][:, 0] == ep_v, 1:]
    X_pred_ep = X_pred[X_pred[:, 0] == ep_v, 1:]
    Xt_train_ep = kp.lift(X_train_ep, episode_feature=False)
    # Plot states
    for k in range(kp.n_states_in_):
        fig, ax = plt.subplots(constrained_layout=True, figsize=(6.5, 6.5))
        ax.plot(X_train_ep[:, k], lw=6, color=OI['orange'])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(path.joinpath(f'x_{k}.pdf'))
    # Plot lifted states
    for k in range(kp.n_states_out_):
        fig, ax = plt.subplots(constrained_layout=True, figsize=(3, 3))
        ax.plot(Xt_train_ep[:, k], lw=4, color=OI['sky blue'])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(path.joinpath(f'theta_{k}.pdf'))
    # Plot parametric trajectory
    fig, ax = plt.subplots(constrained_layout=True, figsize=(7.8, 7.8))
    ax.plot(
        X_valid_ep[:, 0],
        X_valid_ep[:, 1],
        ls=':',
        lw=6,
        color=OI['grey'],
    )
    ax.plot(
        X_pred_ep[:, 0],
        X_pred_ep[:, 1],
        lw=6,
        color=OI['bluish green'],
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(path.joinpath('pred.pdf'))


if __name__ == '__main__':
    main()
