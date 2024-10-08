# Contribution Guidelines

Everyone is welcome to contribute to `pykoop`. To do so, please fork the
repository and create a pull request when you're ready.

Contributed code must

1. be documented using
   [NumPy style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html),
2. be formatted using using
   [YAPF](https://github.com/google/yapf) (see `./.style.yapf`),
3. use [type annotations](https://docs.python.org/3/library/typing.html)
   consistently,
4. include relevant unit tests, and
5. pass existing unit tests and checks.

If you are fixing a bug, please include a set of unit tests in your pull
request that would fail without your changes.

If you notice a problem or would like to suggest an enhancement, please create
an [issue](https://github.com/decargroup/pykoop/issues).
