Koopman pipeline
----------------

.. automodule:: pykoop.koopman_pipeline
   :members:
   :inherited-members:
   :show-inheritance:

Lifting functions
-----------------

.. automodule:: pykoop.lifting_functions
   :members:
   :inherited-members:
   :show-inheritance:

Regressors
----------

.. automodule:: pykoop.regressors
   :members:
   :inherited-members:
   :show-inheritance:

Truncated SVD
-------------

.. automodule:: pykoop.tsvd
   :members:
   :inherited-members:
   :show-inheritance:

Utilities
---------

.. automodule:: pykoop.util
   :members:
   :inherited-members:
   :show-inheritance:

LMI regressors
--------------

.. automodule:: pykoop.lmi_regressors
   :members:
   :inherited-members:
   :show-inheritance:

Dynamic models
--------------

.. automodule:: pykoop.dynamic_models
   :members:
   :inherited-members:
   :show-inheritance:

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
