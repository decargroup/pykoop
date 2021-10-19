pykoop
======

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5576490.svg
    :target: https://doi.org/10.5281/zenodo.5576490
    :alt: DOI
.. image:: https://readthedocs.org/projects/pykoop/badge/?version=stable
    :target: https://pykoop.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation status

``pykoop`` is a Koopman operator identification library written in Python. It
allows the user to specify Koopman lifting functions and regressors in order to
learn a linear model of a given system in the lifted space.

``pykoop`` places heavy emphasis on modular lifting function construction and
``scikit-learn`` compatibility. The library aims to make it easy to
automatically find good lifting functions and regressor hyperparameters by
leveraging ``scikit-learn``'s existing cross-validation infrastructure.
``pykoop`` also gracefully handles control inputs and multi-episode datasets
at every stage of the pipeline.

``pykoop`` also includes several experimental regressors that use linear matrix
inequalities to regularize or constrain the Koopman matrix [lmikoop]_.

Example
=======

Consider Tikhonov-regularized EDMD with polynomial lifting functions applied to
mass-spring-damper data. Using ``pykoop``, this can be implemented as:

.. code-block:: python

    import pykoop
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler

    # Get sample mass-spring-damper data
    X_msd = pykoop.example_data_msd()

    # Create pipeline
    kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            ('ma', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
            ('pl', pykoop.PolynomialLiftingFn(order=2)),
            ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ],
        regressor=pykoop.Edmd(alpha=0.1),
    )

    # Fit the pipeline
    kp.fit(X_msd, n_inputs=1, episode_feature=True)

    # Predict using the pipeline
    X_pred = kp.predict_multistep(X_msd)

    # Score using the pipeline
    score = kp.score(X_msd)

Library layout
==============

Most of the required classes and functions have been imported into the
``pykoop`` namespace. The most important object is the
``KoopmanPipeline``, which requires a list of lifting functions and
a regressor.

Some example lifting functions are

- ``PolynomialLiftingFn``,
- ``DelayLiftingFn``, and
- ``BilinearInputLiftingFn``.

``scikit-learn`` preprocessors can be wrapped into lifting functions using
``SkLearnLiftingFn``. States and inputs can be lifted independently using
``SplitPipeline``. This is useful to avoid lifting inputs.

Some basic regressors included are

- ``Edmd`` (includes Tikhonov regularization),
- ``Dmdc``, and
- ``Dmd``.

More advanced (and experimental) LMI-based regressors are included in the
``pykoop.lmi_regressors`` namespace. They allow for different kinds of
regularization as well as hard constraints on the Koopman operator.

You can roll your own lifting functions and regressors by inheriting from
``KoopmanLiftingFn``, ``EpisodeIndependentLiftingFn``,
``EpisodeDependentLiftingFn``, and ``KoopmanRegressor``.

Some sample dynamic models are also included in the ``pykoop.dynamic_models``
namespace.

Installation and testing
========================

``pykoop`` can be installed from PyPI using

.. code-block:: sh

    $ pip install pykoop

Additional LMI solvers can be installed using

.. code-block:: sh

    $ pip install mosek
    $ pip install smcp

Mosek is recommended, but is nonfree and requires a license.

The library can be tested using

.. code-block:: sh

    $ pip install -r requirements.txt
    $ pytest

Note that ``pytest`` must be run from the repository's root directory.

To skip slow unit tests, including all doctests and examples, run

.. code-block:: sh

    $ pytest ./tests -k-slow

The documentation can be compiled using

.. code-block:: sh

    $ cd doc
    $ make html


Related packages
================

Other excellent Python packages for learning dynamical systems exist,
summarized in the table below:

============ ==================================================================
Library      Unique features
============ ==================================================================
`pykoop`_    - Modular lifting functions
             - Full ``scikit-learn`` compatibility
             - Built-in regularization
             - Multi-episode datasets
`pykoopman`_ - Continuous-time Koopman operator identification
             - Built-in numerical differentiation
             - Detailed DMD outputs
             - DMDc with known control matrix
`PyDMD`_     - Extensive library containing pretty much every variant of DMD
`PySINDy`_   - Python implementation of the famous SINDy method
             - Related to, but not the same as, Koopman operator approximation
============ ==================================================================

.. _pykoop: https://github.com/decarsg/pykoop
.. _pykoopman: https://github.com/dynamicslab/pykoopman
.. _PyDMD: https://github.com/mathLab/PyDMD
.. _PySINDy: https://github.com/dynamicslab/pysindy

References
==========

.. [lmikoop] Steven Dahdah and James Richard Forbes. "Linear matrix inequality
   approaches to Koopman operator approximation." arXiv:2102.03613 [eess.SY]
   (2021). https://arxiv.org/abs/2102.03613

Citation
========

If you use this software in your research, please cite it as below or see
``CITATION.cff``.

.. code-block:: bibtex

    @software{dahdah_pykoop_2021,
        title={{decarsg/pykoop}},
        doi={10.5281/zenodo.5576490},
        url={https://github.com/decarsg/pykoop},
        publisher={Zenodo},
        author={Steven Dahdah and James Richard Forbes},
        year={2021},
    }

License
=======

This project is distributed under the MIT License, except the contents of
``./pykoop/_sklearn_metaestimators/``, which are from the `scikit-learn`_
project, and are distributed under the BSD-3-Clause License.

.. _scikit-learn: https://github.com/scikit-learn/scikit-learn
