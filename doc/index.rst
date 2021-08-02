.. pykoop documentation master file, created by
   sphinx-quickstart on Tue Jun 15 16:46:12 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pykoop Documentation
====================

Koopman operator identification library in Python.

For example, consider Tikhonov-regularized EDMD with polynomial lifting
functions applied to mass-spring-damper data:

.. code-block:: python

    import pykoop
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler

    # Create pipeline
    kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            pykoop.SkLearnLiftingFn(MaxAbsScaler()),
            pykoop.PolynomialLiftingFn(order=2),
            pykoop.SkLearnLiftingFn(StandardScaler())
        ],
        regressor=pykoop.Edmd(alpha=0.1),
    )

    # Fit the pipeline
    kp.fit(X_msd, n_inputs=1, episode_feature=True)

    # Predict using the pipeline
    X_pred = kp.predict_multistep(X_msd)

    # Score using the pipeline
    score = kp.score(X_pred)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pykoop

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
