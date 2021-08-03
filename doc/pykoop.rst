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

To enable cross-validation, ``pykoop`` strives to be fully-compatible with
``scikit-learn``. All of its regressors and lifting functions pass
``scikit-learn``'s `estimator checks`_, with minor exceptions made when
necessary.

.. _estimator checks: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html

Regressor parameters and lifting functions can easily be cross-validated using
``scikit-learn``:

.. include:: ../examples/example_pipeline_cv.py
   :literal:
