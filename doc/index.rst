.. pykoop documentation master file, created by
   sphinx-quickstart on Tue Jun 15 16:46:12 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pykoop Documentation
====================

Koopman operator identification library in Python.

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

.. include:: ../examples/example_pipeline_simple.py
   :literal:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pykoop

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

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
             - DMDc with known control
`PyDMD`_     - Extensive library containing pretty much every variant of DMD
`PySINDy`_   - Python implementation of the famous (SINDy) method
             - Related to, but not the same as, Koopman operator approximation
============ ==================================================================

.. _pykoop: https://github.com/decarsg/pykoop
.. _pykoopman: https://github.com/dynamicslab/pykoopman
.. _PyDMD: https://github.com/mathLab/PyDMD
.. _PySINDy: https://github.com/dynamicslab/pysindy

References
==========

.. [optht] Matan Gavish and David L. Donoho. "The optimal hard threshold for
   singular values is 4/sqrt(3)." IEEE Transactions on Information Theory 60.8
   (2014): 5040-5053. http://arxiv.org/abs/1305.5870
.. [dissip] Keita Hara, Masaki Inoue, and Noboru Sebe. "Learning Koopman
   operator under dissipativity constraints." arXiv:1911.03884v1 [eess.SY]
   (2019). https://arxiv.org/abs/1911.03884v1
.. [lmikoop] Steven Dahdah and James Richard Forbes. "Linear matrix inequality
   approaches to Koopman operator approximation." arXiv:2102.03613 [eess.SY]
   (2021). https://arxiv.org/abs/2102.03613
.. [bilinear] Daniel Bruder, Xun Fu, and Ram Vasudevan. "Advantages of bilinear
   Koopman realizations for the modeling and control of systems with unknown
   dynamics." arXiv:2010.09961v3 [cs.RO] (2020).
   https://arxiv.org/abs/2010.09961v3
