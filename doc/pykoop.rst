Koopman pipeline
================

Since the Koopman regression problem operates on timeseries data, it has
additional requirements that preclude the use of ``scikit-learn``
:class:`sklearn.pipeline.Pipeline` objects:

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
:class:`pykoop.KoopmanLiftingFn` interface, supports a feature that indicates
which episode each sample belongs to. Furthermore, each lifting function stores
the number of input-dependent and input-independent features at its input and
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
:class:`pykoop.KoopmanRegressor` are distinct from ``scikit-learn`` regressors
in that they support the episode feature and state tracking attributes used by
the lifting function objects. Koopman regressors also support being fit with a
single data matrix, which they will split and time-shift according to the
episode feature.

If the input is a :class:`pandas.DataFrame`, then ``pykoop`` will store the
column names in ``feature_names_in_`` upon fitting. This applies to both
:class:`pykoop.KoopmanLiftingFn` and :class:`pykoop.KoopmanRegressor`. If these
feature names are specified, calling ``get_feature_names_in()`` will return
them. If they do not exist, this function will return auto-generated ones. For
instances of :class:`pykoop.KoopmanLiftingFn`, calling
``get_feature_names_out()`` will generate the feature names of the lifted
states. Note that :class:`pandas.DataFrame` instances are converted to
:class:`numpy.ndarray` instances as soon as they are processed by ``pykoop``.
You can recreate them using something like ``pandas.DataFrame(X_lifted,
columns=lf.get_feature_names_out())``.

The following class and function implementations are located in
:mod:`pykoop.koopman_pipeline`, but have been imported into the ``pykoop``
namespace for convenience.

.. autosummary::
   :toctree: _autosummary/

   pykoop.KoopmanPipeline
   pykoop.SplitPipeline
   pykoop.combine_episodes
   pykoop.extract_initial_conditions
   pykoop.extract_input
   pykoop.score_trajectory
   pykoop.shift_episodes
   pykoop.split_episodes
   pykoop.strip_initial_conditions


Lifting functions
=================

All of the lifting functions included in this module adhere to the interface
defined in :class:`pykoop.KoopmanLiftingFn`.

The following class and function implementations are located in
``pykoop.lifting_functions``, but have been imported into the ``pykoop``
namespace for convenience.

.. autosummary::
   :toctree: _autosummary/

   pykoop.BilinearInputLiftingFn
   pykoop.ConstantLiftingFn
   pykoop.DelayLiftingFn
   pykoop.PolynomialLiftingFn
   pykoop.SkLearnLiftingFn


Regressors
==========

All of the lifting functions included in this module adhere to the interface
defined in :class:`pykoop.KoopmanRegressor`.

The following class and function implementations are located in
``pykoop.regressors``, but have been imported into the ``pykoop`` namespace for
convenience.

.. autosummary::
   :toctree: _autosummary/

   pykoop.Dmd
   pykoop.Dmdc
   pykoop.Edmd
   pykoop.EdmdMeta


Radial basis function centers
=============================

The following classes are used to generate centers for radial basis function
(RBF) lifting functions.

.. autosummary::
   :toctree: _autosummary/

   pykoop.ClusterCenters
   pykoop.GaussianRandomCenters
   pykoop.GaussianMixtureRandomCenters
   pykoop.GridCenters
   pykoop.QmcCenters
   pykoop.UniformRandomCenters


Truncated SVD
=============

The following class and function implementations are located in
``pykoop.tsvd``, but have been imported into the ``pykoop`` namespace for
convenience.

.. autosummary::
   :toctree: _autosummary/

   pykoop.Tsvd


Utilities
=========

The following class and function implementations are located in
``pykoop.util``, but have been imported into the ``pykoop`` namespace for
convenience.

.. autosummary::
   :toctree: _autosummary/

   pykoop.AnglePreprocessor
   pykoop.example_data_msd
   pykoop.example_data_vdp
   pykoop.random_input
   pykoop.random_state


LMI regressors
==============

Experimental LMI-based Koopman regressors from [DF21]_ and [DF22]_.

.. warning:: 
   Importing this module has side effects! When imported, the module creates a
   temporary directory with the prefix ``pykoop_``, which is used to memoize
   long computations that may be repreated frequently. It also catches
   ``SIGINT`` so that long regressions can be stopped politely.

The following class and function implementations are located in
``pykoop.lmi_regressors``, which must be imported separately.

.. autosummary::
   :toctree: _autosummary/

   pykoop.lmi_regressors.LmiEdmd
   pykoop.lmi_regressors.LmiEdmdDissipativityConstr
   pykoop.lmi_regressors.LmiEdmdHinfReg
   pykoop.lmi_regressors.LmiEdmdSpectralRadiusConstr
   pykoop.lmi_regressors.LmiDmdc
   pykoop.lmi_regressors.LmiDmdcHinfReg
   pykoop.lmi_regressors.LmiDmdcSpectralRadiusConstr
   pykoop.lmi_regressors.LmiHinfZpkMeta


Dynamic models
==============

The following class and function implementations are located in
``pykoop.dynamic_models``, which must be imported separately.

.. autosummary::
   :toctree: _autosummary/

   pykoop.dynamic_models.DiscreteVanDerPol
   pykoop.dynamic_models.MassSpringDamper
   pykoop.dynamic_models.Pendulum


Extending ``pykoop``
====================

The abstract classes from all of ``pykoop``'s modules have been grouped here.
If you want to write your own lifting functions or regressor, this is the place
to look!

The following abstract class implementations are spread across
``pykoop.koopman_pipeline``, ``pykoop.dynamic_models``, and
``pykoop.lmi_regressors``. The most commonly used ones have been imported into
the ``pykoop`` namespace.

.. autosummary::
   :toctree: _autosummary/

   pykoop.EpisodeDependentLiftingFn
   pykoop.EpisodeIndependentLiftingFn
   pykoop.KoopmanLiftingFn
   pykoop.KoopmanRegressor
   pykoop.dynamic_models.ContinuousDynamicModel
   pykoop.dynamic_models.DiscreteDynamicModel
   pykoop.lmi_regressors.LmiRegressor


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


References
==========

.. [GD14] Matan Gavish and David L. Donoho. "The optimal hard threshold for
   singular values is 4/sqrt(3)." IEEE Transactions on Information Theory 60.8
   (2014): 5040-5053. http://arxiv.org/abs/1305.5870
.. [HIS19] Keita Hara, Masaki Inoue, and Noboru Sebe. "Learning Koopman
   operator under dissipativity constraints." arXiv:1911.03884v1 [eess.SY]
   (2019). https://arxiv.org/abs/1911.03884v1
.. [DF21] Steven Dahdah and James Richard Forbes. "Linear matrix inequality
   approaches to Koopman operator approximation." arXiv:2102.03613 [eess.SY]
   (2021). https://arxiv.org/abs/2102.03613
.. [DF22] Steven Dahdah and James Richard Forbes. "System norm regularization
   methods for Koopman operator approximation." arXiv:2110.09658 [eess.SY]
   (2022). https://arxiv.org/abs/2110.09658
.. [BFV20] Daniel Bruder, Xun Fu, and Ram Vasudevan. "Advantages of bilinear
   Koopman realizations for the modeling and control of systems with unknown
   dynamics." arXiv:2010.09961v3 [cs.RO] (2020).
.. [MAM22] Giorgos Mamakoukas, Ian Abraham, and Todd D. Murphey. "Learning
   Stable Models for Prediction and Control." arXiv:2005.04291v2 [cs.RO]
   (2022). https://arxiv.org/abs/2005.04291v2
.. [DTK20] Felix Dietrich, Thomas N. Thiem, and Ioannis G. Kevrekidis. "On the
   Koopman Operator of Algorithms." SIAM Journal on Applied Dynamical Systems,
   vol. 19, no. 2, pp. 860-885, 2020. https://doi.org/10.1137%2F19m1277059
.. [WHDKR16] Matthew O. Williams, Maziar S. Hemati, Scott T. M. Dawson, Ioannis
   G. Kevrekidis, and Clarence W. Rowley. "Extending Data-Driven Koopman
   Analysis to Actuated Systems." IFAC-PapersOnLine, vol. 49, no. 18, pp.
   704-709, 2016. https://doi.org/10.1016%2Fj.ifacol.2016.10.248
.. [CHH19] Vı́t Cibulka, Tomáš Haniš, and Martin Hromčı́k. "Data-driven
   identification of vehicle dynamics using Koopman operator," in 22nd
   International Conference on Process Control, 2019.
   https://doi.org/10.1109%2Fpc.2019.8815104
