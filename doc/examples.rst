Examples
========

Simple Koopman pipeline
-----------------------

.. plot:: ../examples/1_example_pipeline_simple.py
   :include-source:

Van der Pol Oscillator
-----------------------

.. plot:: ../examples/2_example_pipeline_vdp.py
   :include-source:

Cross-validation with ``scikit-learn``
--------------------------------------

To enable cross-validation, ``pykoop`` strives to be fully-compatible with
``scikit-learn``. All of its regressors and lifting functions pass
``scikit-learn``'s `estimator checks`_, with minor exceptions made when
necessary.

.. _estimator checks: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html

Regressor parameters and lifting functions can easily be cross-validated using
``scikit-learn``:

.. plot:: ../examples/3_example_pipeline_cv.py
   :include-source:

Asymptotic stability constraint
-------------------------------

In this example, three experimental EDMD-based regressors are compared to EDMD.
Specifically, EDMD is compared to the asymptotic stability constraint and the
H-infinity norm regularizer from [DF22]_ and [DF21]_, and the dissipativity
constraint from [HIS19]_.

.. plot:: ../examples/4_example_eigenvalue_comparison.py
   :include-source:

Sparse regression
-----------------

This example shows how to use :class:`pykoop.EdmdMeta` to implement sparse
regression with :class:`sklearn.linear_model.Lasso`. The lasso promotes empty
columns in the Koopman matrix, which means the corresponding lifting functions
can be removed from the model.

.. plot:: ../examples/5_example_sparse_regression.py
   :include-source:

Radial basis functions on a pendulum
------------------------------------

This example shows how thin-plate radial basis functions can be used as lifting
functions to identify pendulum dynamics (where all trajectories have zero
initial velocity). Latin hypercube sampling is used to generate 100 centers.

.. plot:: ../examples/6_example_rbf_pendulum.py
   :include-source:

Random Fourier features on a Duffing oscillator
-----------------------------------------------

This example shows how random Fourier features (and randomly binned features)
can be used as lifting functions to identify Duffing oscillator dynamics.
For more details on how these features are generated, see [RR07]_.

.. plot:: ../examples/7_example_rff_duffing.py
   :include-source:

Control using the Koopman operator
----------------------------------

This example demonstrates Koopman LQR control using a Van der Pol oscillator.

.. plot:: ../examples/8_control_vdp.py
   :include-source:
