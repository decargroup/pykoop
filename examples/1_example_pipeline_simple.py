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
    kp.plot_predicted_trajectory(eg['X_valid'], plot_input=True)


if __name__ == '__main__':
    example_pipeline_simple()
    plt.show()
