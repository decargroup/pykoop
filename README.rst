.. role:: class(code)

pykoop
======

.. image:: https://github.com/decargroup/pykoop/actions/workflows/test-package.yml/badge.svg
    :target: https://github.com/decargroup/pykoop/actions/workflows/test-package.yml
    :alt: Test package
.. image:: https://readthedocs.org/projects/pykoop/badge/?version=stable
    :target: https://pykoop.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation status
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5576490.svg
    :target: https://doi.org/10.5281/zenodo.5576490
    :alt: DOI
.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/decargroup/pykoop/main?labpath=notebooks%2F1_example_pipeline_simple.ipynb
    :alt: Examples on binder

``pykoop`` is a Koopman operator identification library written in Python. It
allows the user to specify Koopman lifting functions and regressors in order to
learn a linear model of a given system in the lifted space.

.. image:: https://raw.githubusercontent.com/decargroup/pykoop/main/doc/_static/pykoop_diagram.png
   :alt: Koopman pipeline diagram

To learn more about Koopman operator theory, check out
`this talk <https://www.youtube.com/watch?v=Lidd_M7gzvA>`_
or
`this review article <https://arxiv.org/abs/2102.12086>`_.


``pykoop`` places heavy emphasis on modular lifting function construction and
``scikit-learn`` compatibility. The library aims to make it easy to
automatically find good lifting functions and regressor hyperparameters by
leveraging ``scikit-learn``'s existing cross-validation infrastructure.
``pykoop`` also gracefully handles control inputs and multi-episode datasets
at every stage of the pipeline.

``pykoop`` also includes several experimental regressors that use linear matrix
inequalities to constraint the asymptotic stability of the Koopman system, or
regularize the regression using its H-infinity norm. Check out
`arXiv:2110.09658 [eess.SY] <https://arxiv.org/abs/2110.09658>`_ and
`arXiv:2102.03613 [eess.SY] <https://arxiv.org/abs/2102.03613>`_ for details.


Example
=======

Consider Tikhonov-regularized EDMD with polynomial lifting functions applied to
mass-spring-damper data. Using ``pykoop``, this can be implemented as:

.. code-block:: python

    import pykoop
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler

    # Get example mass-spring-damper data
    eg = pykoop.example_data_msd()

    # Create pipeline
    kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            ('ma', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
            ('pl', pykoop.PolynomialLiftingFn(order=2)),
            ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
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
    X_pred = kp.predict_trajectory(eg['x0_valid'], eg['u_valid'])

    # Score using the pipeline
    score = kp.score(eg['X_valid'])

    # Plot results
    kp.plot_predicted_trajectory(eg['X_valid'], plot_input=True)

More examples are available in ``examples/``, in ``notebooks/``, or on
`binder <https://mybinder.org/v2/gh/decargroup/pykoop/main?labpath=notebooks%2F1_example_pipeline_simple.ipynb>`_.


Library layout
==============

Most of the required classes and functions have been imported into the
``pykoop`` namespace. The most important object is the
:class:`pykoop.KoopmanPipeline`, which requires a list of lifting functions and
a regressor.

Some example lifting functions are

- :class:`pykoop.PolynomialLiftingFn`,
- :class:`pykoop.RbfLiftingFn`,
- :class:`pykoop.DelayLiftingFn`, and
- :class:`pykoop.BilinearInputLiftingFn`.

``scikit-learn`` preprocessors can be wrapped into lifting functions using
:class:`pykoop.SkLearnLiftingFn`. States and inputs can be lifted independently
using :class:`pykoop.SplitPipeline`. This is useful to avoid lifting inputs.

Some basic regressors included are

- :class:`pykoop.Edmd` (includes Tikhonov regularization),
- :class:`pykoop.Dmdc`, and
- :class:`pykoop.Dmd`.

More advanced (and experimental) LMI-based regressors are included in the
``pykoop.lmi_regressors`` namespace. They allow for different kinds of
regularization as well as hard constraints on the Koopman operator.

You can roll your own lifting functions and regressors by inheriting from
:class:`pykoop.KoopmanLiftingFn`, :class:`pykoop.EpisodeIndependentLiftingFn`,
:class:`pykoop.EpisodeDependentLiftingFn`, and
:class:`pykoop.KoopmanRegressor`.

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
    $ pip install cvxopt
    $ pip install smcp

Mosek is recommended, but is nonfree and requires a license.

The library can be tested using

.. code-block:: sh

    $ pip install -r requirements-dev.txt
    $ pytest

Note that ``pytest`` must be run from the repository's root directory.

To skip unit tests that require a MOSEK license, including all doctests and
examples, run

.. code-block:: sh

    $ pytest ./tests -k "not mosek"

The documentation can be compiled using

.. code-block:: sh

    $ cd doc
    $ make html

If you want a hook to check source code formatting before allowing a commit,
you can use

.. code-block:: sh

   $ cd .git/hooks/
   $ ln -s ../../.githooks/pre-commit .
   $ chmod +x ./pre-commit

You will need ``yapf`` installed for this.


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

.. _pykoop: https://github.com/decargroup/pykoop
.. _pykoopman: https://github.com/dynamicslab/pykoopman
.. _PyDMD: https://github.com/mathLab/PyDMD
.. _PySINDy: https://github.com/dynamicslab/pysindy


Citation
========

If you use this software in your research, please cite it as below or see
``CITATION.cff``.

.. code-block:: bibtex

    @software{dahdah_pykoop_2022,
        title={{decargroup/pykoop}},
        doi={10.5281/zenodo.5576490},
        url={https://github.com/decargroup/pykoop},
        publisher={Zenodo},
        author={Steven Dahdah and James Richard Forbes},
        version = {{v1.2.1}},
        year={2022},
    }


License
=======

This project is distributed under the MIT License, except the contents of
``pykoop/_sklearn_metaestimators/``, which are from the `scikit-learn`_
project, and are distributed under the BSD-3-Clause License.

.. _scikit-learn: https://github.com/scikit-learn/scikit-learn
