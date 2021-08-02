Koopman pipeline
----------------

.. automodule:: pykoop.koopman_pipeline
   :members:

Lifting functions
-----------------

.. automodule:: pykoop.lifting_functions
   :members:
   :inherited-members:

Regressors
----------

.. automodule:: pykoop.regressors
   :members:

LMI regressors
--------------

.. automodule:: pykoop.lmi_regressors
   :members:

Utilities
---------

.. automodule:: pykoop.util
   :members:

Dynamic models
--------------

.. automodule:: pykoop.dynamic_models
   :members:

Cross-validation with ``scikit-learn``
--------------------------------------

Regressor parameters can easily be cross-validated using ``scikit-learn``:

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
