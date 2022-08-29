"""Test ``koopman_pipeline`` module."""

import sklearn.utils.estimator_checks
from sklearn import preprocessing

import pykoop


class TestSklearn:
    """Test scikit-learn compatibility."""

    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.KoopmanPipeline(regressor=pykoop.Edmd()),
        pykoop.KoopmanPipeline(
            lifting_functions=[
                ('pl', pykoop.PolynomialLiftingFn()),
            ],
            regressor=pykoop.Edmd(),
        ),
        pykoop.SplitPipeline(
            lifting_functions_state=None,
            lifting_functions_input=None,
        ),
        pykoop.SplitPipeline(
            lifting_functions_state=[
                ('pl', pykoop.PolynomialLiftingFn()),
            ],
            lifting_functions_input=None,
        ),
    ])
    def test_compatible_estimator(self, estimator, check):
        """Test scikit-learn compatibility of estimators."""
        check(estimator)


class TestDeepParams:
    """Test deep parameter access from ``_sklearn_metaestimators``."""

    def test_get_set(self):
        """Test get and set with :class:`KoopmanPipeline`."""
        kp = pykoop.KoopmanPipeline(lifting_functions=[
            ('p', pykoop.PolynomialLiftingFn(order=1)),
        ])
        assert kp.get_params()['p__order'] == 1
        kp.set_params(p__order=2)
        assert kp.get_params()['p__order'] == 2

    def test_nested_get_set(self):
        """Test get and set with :class:`SplitPipeline`."""
        kp = pykoop.KoopmanPipeline(lifting_functions=[
            ('s',
             pykoop.SplitPipeline(
                 lifting_functions_state=[
                     ('p_state', pykoop.PolynomialLiftingFn(order=1)),
                 ],
                 lifting_functions_input=[
                     ('p_input', pykoop.PolynomialLiftingFn(order=2)),
                 ],
             )),
        ])
        assert kp.get_params()['s__p_state__order'] == 1
        kp.set_params(s__p_state__order=3)
        assert kp.get_params()['s__p_state__order'] == 3

    def test_invalid_set(self):
        """Test set for invalid parameter type."""
        kp = pykoop.KoopmanPipeline(lifting_functions=[
            ('p', pykoop.PolynomialLiftingFn(order=1)),
        ])
        kp.set_params(p=5)  # Wrong type for ``p``
        assert kp.get_params()['p'] == 5
        kp.set_params(p=pykoop.PolynomialLiftingFn(order=2))
        assert kp.get_params()['p__order'] == 2
