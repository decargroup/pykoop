---
title: 'pykoop: a Python Library for Koopman Operator Approximation'
tags:
    - Python
    - Koopman operator
    - system identification
    - differential equations
    - machine learning
    - dynamical systems
authors:
    - name: Steven Dahdah
      corresponding: true
      orcid: 0000-0003-4930-9634
      affiliation: 1
    - name: James Richard Forbes
      orcid: 0000-0002-1987-9268
      affiliation: 1
affiliations:
    - name: Department of Mechanical Engineering, McGill University, Montreal QC, Canada
      index: 1
date: 26 September 2024
bibliography: paper.bib
---

# Summary

`pykoop` is a Python package for learning differential equations in discretized
form using the Koopman operator. Differential equations are an essential tool
for modelling the physical world. Ordinary differential equations can be used to
describe electric circuits, rigid-body dynamics, or chemical reaction rates,
while the fundamental laws of electromagnetism, fluid dynamics, and heat
transfer can be formulated as partial differential equations. The Koopman
operator allows nonlinear differential equations to be rewritten as
infinite-dimensional linear differential equations by viewing their time
evolution in terms of an infinite number of nonlinear lifting functions. A
finite-dimensional approximation of the Koopman operator can be identified from
data given a user-selected set of lifting functions. Thanks to its linearity,
the approximate Koopman model can be used for analysis, design, and optimal
controller or observer synthesis for a wide range of systems using
well-established linear tools. `pykoop`'s documentation, along with examples in
script and notebook form, can be found at at
https://pykoop.readthedocs.io/en/stable/. Its source code and issue tracker are
available at https://github.com/decargroup/pykoop. Its releases are also
archived on Zenodo [@pykoop].

# Statement of need

Designing Koopman lifting functions for a particular system is largely a
trial-and-error process. As such, `pykoop` is designed to facilitate
experimentation with Koopman lifting functions, with the ultimate goal of
automating the lifting function optimization process.
The `pykoop` library addresses three limitations in current Koopman operator
approximation software packages. First, `pykoop` allows lifting functions to be
constructed in a modular fashion, specifically through the composition of a
series of lifting functions. Second, the library allows regressor
hyperparameters and lifting functions to be selected automatically through full
compatibility with `scikit-learn`'s [@scikit-learn] existing cross-validation
infrastructure. Third, `pykoop` handles datasets with control inputs and
multiple training episodes at all stages of the pipeline.
Furthermore, `pykoop` implements state-of-the-art Koopman operator approximation
methods that enforce stability in the identified system, either through direct
constraints on its eigenvalues or by regularizing the regression problem using its
$\mathcal{H}_\infty$ norm [@dahdah_system_2022].

Open-source Python libraries for Koopman operator approximation include
`PyKoopman` [@pykoopman], `DLKoopman` [@dlkoopman], and `kooplearn` [@kooplearn].
While `PyKoopman` provides a similar selection of lifting functions to `pykoop`,
it does not allow lifting functions to be composed. Furthermore, the library
does not include built-in functionality to handle multiple training episodes.
Unlike `pykoop`, which focuses on discrete-time, `PyKoopman` is able to identify
continuous-time Koopman operators. As such, the library includes several
built-in numerical differentiation methods. Neural network lifting functions are
also supported by `PyKoopman`. The `DLKoopman` library focuses on learning
Koopman models using only neural network lifting functions, which are not
implemented in `pykoop`. The library supports multiple training episodes but
does not support exogenous inputs and does not follow the `scikit-learn`
interface. The `kooplearn` library includes kernel and neural network approaches
to learning Koopman models, but does not include other types of lifting
functions. While multiple training episodes are handled by `kooplearn`,
exogenous inputs are not.

# Scholarly publications using `pykoop`

The Koopman operator regression methods proposed in [@dahdah_system_2022] have
been implemented within `pykoop`, while the methods proposed in
[@dahdah_2024_closed-loop], [@lortie_2024_forward-backward],
[@lortie_2024_asymptotically], and [@dahdah_2024_uncertainty] are all based on
`pykoop`, but are implemented in their own repositories.

# Acknowledgements

This work was supported financially by the NSERC Discovery Grants program, the
FRQNT, IVADO, CIFAR, the CRM, and by Mecademic through the Mitacs Accelerate
program.

# References
