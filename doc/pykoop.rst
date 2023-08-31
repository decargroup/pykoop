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
   which episode each timestep belongs to. The episode feature must contain
   positive integers only.
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

.. important::
   The episode feature must contain positive integers only!

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
   pykoop.KernelApproxLiftingFn
   pykoop.PolynomialLiftingFn
   pykoop.RbfLiftingFn
   pykoop.SkLearnLiftingFn


Regressors
==========

All of the lifting functions included in this module adhere to the interface
defined in :class:`pykoop.KoopmanRegressor`.

The following class and function implementations are located in
``pykoop.regressors``, but have been imported into the ``pykoop`` namespace for
convenience.

The :class:`pykoop.DataRegressor` regressor is a dummy regressor if you want to
force the Koopman matrix to take on a specific value (maybe you know what it
should be, or you got it from another library).

.. autosummary::
   :toctree: _autosummary/

   pykoop.Dmd
   pykoop.Dmdc
   pykoop.Edmd
   pykoop.EdmdMeta
   pykoop.DataRegressor


Kernel approximation methods
============================

The following classes are used to generate random feature maps from kernels
for kernel approximation lifting functions (i.e., random Fourier feature
lifting functions).

.. autosummary::
   :toctree: _autosummary/

   pykoop.RandomBinningKernelApprox
   pykoop.RandomFourierKernelApprox

Radial basis function centers
=============================

The following classes are used to generate centers for radial basis function
(RBF) lifting functions.

.. autosummary::
   :toctree: _autosummary/

   pykoop.ClusterCenters
   pykoop.DataCenters
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
   pykoop.example_data_duffing
   pykoop.example_data_msd
   pykoop.example_data_pendulum
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
   pykoop.dynamic_models.DuffingOscillator
   pykoop.dynamic_models.MassSpringDamper
   pykoop.dynamic_models.Pendulum


Configuration
=============

The following functions allow the user to interact with ``pykoop``'s global
configuration.

.. autosummary::
   :toctree: _autosummary/

   pykoop.get_config
   pykoop.set_config
   pykoop.config_context


Extending ``pykoop``
====================

The abstract classes from all of ``pykoop``'s modules have been grouped here.
If you want to write your own lifting functions or regressor, this is the place
to look!

The following abstract class implementations are spread across
:mod:`pykoop.koopman_pipeline`, :mod:`pykoop.dynamic_models`,
:mod:`pykoop.centers`, and :mod:`pykoop.lmi_regressors`. The most commonly used
ones have been imported into the ``pykoop`` namespace.

.. autosummary::
   :toctree: _autosummary/

   pykoop.Centers
   pykoop.EpisodeDependentLiftingFn
   pykoop.EpisodeIndependentLiftingFn
   pykoop.KernelApproximation
   pykoop.KoopmanLiftingFn
   pykoop.KoopmanRegressor
   pykoop.dynamic_models.ContinuousDynamicModel
   pykoop.dynamic_models.DiscreteDynamicModel
   pykoop.lmi_regressors.LmiRegressor
