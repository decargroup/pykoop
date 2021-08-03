"""Example of cross-validating lifting functions and regressor parameters."""

import numpy as np
import pykoop
import pykoop.dynamic_models
import sklearn.model_selection
import sklearn.preprocessing


def main() -> None:
    """Cross-validate regressor parameters."""
    # Get sample mass-spring-damper data
    X_msd = pykoop.example_data_msd()

    # Create pipeline
    kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            pykoop.SkLearnLiftingFn(sklearn.preprocessing.MaxAbsScaler()),
            pykoop.PolynomialLiftingFn(order=2),
            pykoop.SkLearnLiftingFn(sklearn.preprocessing.StandardScaler())
        ],
        regressor=pykoop.Edmd(alpha=0.1),
    )

    # Split data episode-by-episode
    episode_feature = X_msd[:, 0]
    cv = sklearn.model_selection.GroupShuffleSplit(
        random_state=1234,
        n_splits=2,
    ).split(X_msd, groups=episode_feature)

    # Define search parameters
    params = {
        # Lifting functions to try
        'lifting_functions': [
            [
                pykoop.SkLearnLiftingFn(sklearn.preprocessing.StandardScaler())
            ],
            [
                pykoop.SkLearnLiftingFn(sklearn.preprocessing.MaxAbsScaler()),
                pykoop.PolynomialLiftingFn(order=2),
                pykoop.SkLearnLiftingFn(sklearn.preprocessing.StandardScaler())
            ],
        ],
        # Regressor parameters to try
        'regressor__alpha': [0.1, 1, 10],
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
    gs.fit(X_msd, n_inputs=1, episode_feature=True)

    # Get results
    cv_results = gs.cv_results_
    best_estimator = gs.best_estimator_

    # Print best estimator
    print(best_estimator)


if __name__ == '__main__':
    main()
