"""Benchmark :func:`pykoop.predict_trajectory()`.

Outputs a ``.prof`` file that can be visualized using ``snakeviz``.
"""

import cProfile

import pykoop


def main():
    """Benchmark :func:`pykoop.predict_trajectory()`."""
    # Get example mass-spring-damper data
    eg = pykoop.example_data_pendulum()
    # Create pipeline
    kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            ('pl', pykoop.PolynomialLiftingFn(order=2)),
            ('dl', pykoop.DelayLiftingFn(n_delays_state=2, n_delays_input=2)),
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
    with cProfile.Profile() as pr:
        X_pred = kp.predict_trajectory(eg['X_train'])
        pr.dump_stats('benchmark_predict_trajectory.prof')


if __name__ == '__main__':
    main()
