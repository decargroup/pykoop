Koopman pipeline
================

Since the Koopman regression problem operates on timeseries data, it has
additional requirements that preclude the use of ``scikit-learn``
:class:`Pipeline` objects:

1. The original state must be kept at the beginning of the lifted state.
2. The input-dependent lifted states must be kept at the end of the lifted
   state.
3. The number of input-independent and input-dependent lifting functions must
   be tracked throughout the pipeline.
4. Samples must not be reordered or subsampled (this would corrupt delay-based
   lifting functions).
5. Concatenated data from different training epidodes must not be mixed (even
   though the states are adjacent in the array, they may not be sequential in
   time).

To meet these requirements, each lifting function, described by the
:class:`KoopmanLiftingFn` interface, supports a feature that indicates which
episode each sample belongs to. Furthermore, each lifting function stores the
number of input-dependent and input-independent features at its input and
output.

The data matrices provided to :func:`fit` (as well as :func:`transform`
and :func:`inverse_transform`) must obey the following format:

1. If ``episode_feature`` is true, the first feature must indicate
   which episode each timestep belongs to.
2. The last ``n_inputs`` features must be exogenous inputs.
3. The remaining features are considered to be states (input-independent).

Consider an example data matrix where the :func:`fit` parameters are
``episode_feature=True`` and ``n_inputs=1``:

======= ======= ======= =======
Episode State 0 State 1 Input 0
======= ======= ======= =======
0.0       0.1    -0.1    0.2
0.0       0.2    -0.2    0.3
0.0       0.3    -0.3    0.4
1.0      -0.1     0.1    0.3
1.0      -0.2     0.2    0.4
1.0      -0.3     0.3    0.5
2.0       0.3    -0.1    0.3
2.0       0.2    -0.2    0.4
======= ======= ======= =======

In the above matrix, there are three distinct episodes with different
numbers of timesteps. The last feature is an input, so the remaining
two features must be states.

If ``n_inputs=0``, the same matrix is interpreted as:

======= ======= ======= =======
Episode State 0 State 1 State 2
======= ======= ======= =======
0.0       0.1    -0.1    0.2
0.0       0.2    -0.2    0.3
0.0       0.3    -0.3    0.4
1.0      -0.1     0.1    0.3
1.0      -0.2     0.2    0.4
1.0      -0.3     0.3    0.5
2.0       0.3    -0.1    0.3
2.0       0.2    -0.2    0.4
======= ======= ======= =======

If ``episode_feature=False`` and the first feature is omitted, the
matrix is interpreted as:

======= ======= =======
State 0 State 1 State 2
======= ======= =======
 0.1    -0.1    0.2
 0.2    -0.2    0.3
 0.3    -0.3    0.4
-0.1     0.1    0.3
-0.2     0.2    0.4
-0.3     0.3    0.5
 0.3    -0.1    0.3
 0.2    -0.2    0.4
======= ======= =======

In the above case, each timestep is assumed to belong to the same
episode.

Koopman regressors, which implement the interface defined in
:class:`KoopmanRegressor` are distinct from ``scikit-learn`` regressors in that
they support the episode feature and state tracking attributes used by the
lifting function objects. Koopman regressors also support being fit with a
single data matrix, which they will split and time-shift according to the
episode feature.

:class:`pykoop.KoopmanPipeline`
-------------------------------
.. autoclass:: pykoop.KoopmanPipeline

:class:`pykoop.SplitPipeline`
-----------------------------
.. autoclass:: pykoop.SplitPipeline

:func:`pykoop.combine_episodes`
-------------------------------
.. autofunction:: pykoop.combine_episodes

:func:`pykoop.extract_initial_conditions`
-----------------------------------------
.. autofunction:: pykoop.extract_initial_conditions

:func:`pykoop.extract_input`
----------------------------
.. autofunction:: pykoop.extract_input

:func:`pykoop.score_state`
--------------------------
.. autofunction:: pykoop.score_state

:func:`pykoop.shift_episodes`
-----------------------------
.. autofunction:: pykoop.shift_episodes

:func:`pykoop.split_episodes`
-----------------------------
.. autofunction:: pykoop.split_episodes

:func:`pykoop.strip_initial_conditions`
---------------------------------------
.. autofunction:: pykoop.strip_initial_conditions


Lifting functions
=================

All of the lifting functions included in this module adhere to the interface
defined in :class:`pykoop.KoopmanLiftingFn`.

:class:`pykoop.BilinearInputLiftingFn`
--------------------------------------
.. autoclass:: pykoop.BilinearInputLiftingFn

:class:`pykoop.DelayLiftingFn`
------------------------------
.. autoclass:: pykoop.DelayLiftingFn

:class:`pykoop.PolynomialLiftingFn`
-----------------------------------
.. autoclass:: pykoop.PolynomialLiftingFn

:class:`pykoop.SkLearnLiftingFn`
--------------------------------
.. autoclass:: pykoop.SkLearnLiftingFn


Regressors
==========

All of the lifting functions included in this module adhere to the interface
defined in :class:`pykoop.KoopmanRegressor`.

:class:`pykoop.Dmd`
-------------------
.. autoclass:: pykoop.Dmd

:class:`pykoop.Dmdc`
--------------------
.. autoclass:: pykoop.Dmdc

:class:`pykoop.Edmd`
--------------------
.. autoclass:: pykoop.Edmd


Truncated SVD
=============

:class:`pykoop.Tsvd`
--------------------
.. autoclass:: pykoop.Tsvd


Utilities
=========

:class:`pykoop.AnglePreprocessor`
---------------------------------
.. autoclass:: pykoop.AnglePreprocessor

:func:`pykoop.example_data_msd`
-------------------------------
.. autofunction:: pykoop.example_data_msd

:func:`pykoop.example_data_vdp`
-------------------------------
.. autofunction:: pykoop.example_data_vdp

:func:`pykoop.random_input`
---------------------------
.. autofunction:: pykoop.random_input

:func:`pykoop.random_state`
---------------------------
.. autofunction:: pykoop.random_state


LMI regressors
==============

Experimental LMI-based Koopman regressors from [lmikoop]_ and [sysnorm]_.

Warning
-------
Importing this module has side effects! When imported, the module creates a
temporary directory with the prefix ``pykoop_``, which is used to memoize long
computations that may be repreated frequently. It also catches ``SIGINT`` so
that long regressions can be stopped politely.

:class:`pykoop.lmi_regressors.LmiEdmd`
--------------------------------------
.. autoclass:: pykoop.lmi_regressors.LmiEdmd

:class:`pykoop.lmi_regressors.LmiEdmdDissipativityConstr`
---------------------------------------------------------
.. autoclass:: pykoop.lmi_regressors.LmiEdmdDissipativityConstr

:class:`pykoop.lmi_regressors.LmiEdmdHinfReg`
---------------------------------------------
.. autoclass:: pykoop.lmi_regressors.LmiEdmdHinfReg

:class:`pykoop.lmi_regressors.LmiEdmdSpectralRadiusConstr`
----------------------------------------------------------
.. autoclass:: pykoop.lmi_regressors.LmiEdmdSpectralRadiusConstr

:class:`pykoop.lmi_regressors.LmiDmdc`
--------------------------------------
.. autoclass:: pykoop.lmi_regressors.LmiDmdc

:class:`pykoop.lmi_regressors.LmiDmdcHinfReg`
---------------------------------------------
.. autoclass:: pykoop.lmi_regressors.LmiDmdcHinfReg

:class:`pykoop.lmi_regressors.LmiDmdcSpectralRadiusConstr`
----------------------------------------------------------
.. autoclass:: pykoop.lmi_regressors.LmiDmdcSpectralRadiusConstr

:class:`pykoop.lmi_regressors.LmiHinfZpkMeta`
---------------------------------------------
.. autoclass:: pykoop.lmi_regressors.LmiHinfZpkMeta


Dynamic models
==============

:class:`pykoop.dynamic_models.DiscreteVanDerPol`
------------------------------------------------
.. autoclass:: pykoop.dynamic_models.DiscreteVanDerPol

:class:`pykoop.dynamic_models.MassSpringDamper`
-----------------------------------------------
.. autoclass:: pykoop.dynamic_models.MassSpringDamper

:class:`pykoop.dynamic_models.Pendulum`
---------------------------------------
.. autoclass:: pykoop.dynamic_models.Pendulum


Extending ``pykoop``
====================

The abstract classes from all of ``pykoop``'s modules have been grouped here.
If you want to write your own lifting functions or regressor, this is the place
to look!

:class:`pykoop.EpisodeDependentLiftingFn`
-----------------------------------------
.. autoclass:: pykoop.EpisodeDependentLiftingFn

:class:`pykoop.EpisodeIndependentLiftingFn`
-------------------------------------------
.. autoclass:: pykoop.EpisodeIndependentLiftingFn

:class:`pykoop.KoopmanLiftingFn`
--------------------------------
.. autoclass:: pykoop.KoopmanLiftingFn

:class:`pykoop.KoopmanRegressor`
--------------------------------
.. autoclass:: pykoop.KoopmanRegressor

:class:`pykoop.dynamic_models.ContinuousDynamicModel`
-----------------------------------------------------
.. autoclass:: pykoop.dynamic_models.ContinuousDynamicModel

:class:`pykoop.dynamic_models.DiscreteDynamicModel`
---------------------------------------------------
.. autoclass:: pykoop.dynamic_models.DiscreteDynamicModel

:class:`pykoop.lmi_regressors.LmiRegressor`
-------------------------------------------
.. autoclass:: pykoop.lmi_regressors.LmiRegressor


Examples
========

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
