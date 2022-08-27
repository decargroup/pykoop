"""Example of how to use the Koopman pipeline."""

from sklearn.preprocessing import MaxAbsScaler, StandardScaler

import pykoop


def main() -> None:
    """Demonstrate how to use the Koopman pipeline."""
    # Get sample mass-spring-damper data
    X_msd = pykoop.example_data_msd()

    # Create pipeline
    kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            ('ma', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
            ('pl', pykoop.PolynomialLiftingFn(order=2)),
            ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ],
        regressor=pykoop.Edmd(alpha=0.1),
    )

    # Fit the pipeline
    kp.fit(X_msd, n_inputs=1, episode_feature=True)

    # Predict using the pipeline
    X_initial = X_msd[[0], 1:3]
    U = X_msd[:, [3]]
    X_pred = kp.predict_state(X_initial, U, episode_feature=False)

    # Score using the pipeline
    score = kp.score(X_msd)


if __name__ == '__main__':
    main()
