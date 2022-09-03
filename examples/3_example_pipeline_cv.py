"""Example of cross-validating lifting functions and regressor parameters."""

import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
from matplotlib import pyplot as plt

import pykoop
import pykoop.dynamic_models

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


def example_pipeline_cv() -> None:
    """Cross-validate regressor parameters."""
    # Get example mass-spring-damper data
    eg = pykoop.example_data_msd()

    # Create pipeline. Don't need to set lifting functions since they'll be set
    # during cross-validation.
    kp = pykoop.KoopmanPipeline(regressor=pykoop.Edmd())

    # Split data episode-by-episode
    episode_feature = eg['X_train'][:, 0]
    cv = sklearn.model_selection.GroupShuffleSplit(
        random_state=1234,
        n_splits=3,
    ).split(eg['X_train'], groups=episode_feature)

    # Define search parameters
    params = {
        # Lifting functions to try
        'lifting_functions': [
            [(
                'ss',
                pykoop.SkLearnLiftingFn(
                    sklearn.preprocessing.StandardScaler()),
            )],
            [
                (
                    'ma',
                    pykoop.SkLearnLiftingFn(
                        sklearn.preprocessing.MaxAbsScaler()),
                ),
                (
                    'pl',
                    pykoop.PolynomialLiftingFn(order=2),
                ),
                (
                    'ss',
                    pykoop.SkLearnLiftingFn(
                        sklearn.preprocessing.StandardScaler()),
                ),
            ],
        ],
        # Regressor parameters to try
        'regressor__alpha': [0, 0.1, 1, 10],
    }

    # Set up grid search
    gs = sklearn.model_selection.GridSearchCV(
        kp,
        params,
        cv=cv,
        # Score using short and long prediction time frames
        scoring={
            f'full_episode': pykoop.KoopmanPipeline.make_scorer(),
            f'ten_steps': pykoop.KoopmanPipeline.make_scorer(n_steps=10),
        },
        # Rank according to short time frame
        refit='ten_steps',
    )

    # Fit the pipeline
    gs.fit(
        eg['X_train'],
        n_inputs=eg['n_inputs'],
        episode_feature=eg['episode_feature'],
    )

    # Get results
    cv_results = gs.cv_results_
    best_estimator = gs.best_estimator_

    # Predict using the pipeline
    X_pred = best_estimator.predict_trajectory(eg['x0_valid'], eg['u_valid'])

    # Score using the pipeline
    score = best_estimator.score(eg['X_valid'])

    # Plot results
    fig, ax = plt.subplots(
        best_estimator.n_states_in_ + best_estimator.n_inputs_in_,
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
    example_pipeline_cv()
    plt.show()
