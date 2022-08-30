"""Test :mod:`koopman_pipeline`."""

import numpy as np
import pytest
import sklearn.utils.estimator_checks
from sklearn import preprocessing

import pykoop


@pytest.mark.parametrize(
    'X, episodes, episode_feature',
    [
        # Multiple episodes
        (
            np.array([
                [0, 0, 0, 1, 1, 1],
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
            ]).T,
            [
                (
                    0,
                    np.array([
                        [1, 2, 3],
                        [6, 5, 4],
                    ]).T,
                ),
                (
                    1,
                    np.array([
                        [4, 5, 6],
                        [3, 2, 1],
                    ]).T,
                ),
            ],
            True,
        ),
        # One episode
        (
            np.array([
                [0, 0, 0, 0, 0, 0],
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
            ]).T,
            [
                (
                    0,
                    np.array([
                        [1, 2, 3, 4, 5, 6],
                        [6, 5, 4, 3, 2, 1],
                    ]).T,
                ),
            ],
            True,
        ),
        # No episode feature
        (
            np.array([
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
            ]).T,
            [
                (
                    0,
                    np.array([
                        [1, 2, 3, 4, 5, 6],
                        [6, 5, 4, 3, 2, 1],
                    ]).T,
                ),
            ],
            False,
        ),
        # Out-of-order episodes
        (
            np.array([
                [2, 2, 2, 0, 0, 1],
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
            ]).T,
            [
                (
                    2,
                    np.array([
                        [1, 2, 3],
                        [6, 5, 4],
                    ]).T,
                ),
                (
                    0,
                    np.array([
                        [4, 5],
                        [3, 2],
                    ]).T,
                ),
                (
                    1,
                    np.array([
                        [6],
                        [1],
                    ]).T,
                ),
            ],
            True,
        ),
    ],
)
class TestSplitCombineEpisodes:
    """Test :func:`split_episodes` and :func:`combine_episodes`."""

    def test_split_episodes(self, X, episodes, episode_feature):
        """Test :func:`split_episodes`.

        .. todo::
            Break up multiple asserts.
        """
        # Split episodes
        episodes_actual = pykoop.split_episodes(
            X,
            episode_feature=episode_feature,
        )
        # Compare every episode
        for actual, expected in zip(episodes_actual, episodes):
            i_actual, X_actual = actual
            i_expected, X_expected = expected
            assert i_actual == i_expected
            np.testing.assert_allclose(X_actual, X_expected)

    def test_combine_episodes(self, X, episodes, episode_feature):
        """Test :func:`combine_episodes`."""
        X_actual = pykoop.combine_episodes(
            episodes,
            episode_feature=episode_feature,
        )
        np.testing.assert_allclose(X_actual, X)


@pytest.mark.parametrize(
    'lf, X, Xt_exp, n_inputs, episode_feature, attr_exp',
    [
        # Basic, without episode feature
        (
            pykoop.SplitPipeline(
                lifting_functions_state=None,
                lifting_functions_input=None,
            ),
            np.array([
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array([
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            2,
            False,
            {
                'n_features_in_': 4,
                'n_states_in_': 2,
                'n_inputs_in_': 2,
                'n_features_out_': 4,
                'n_states_out_': 2,
                'n_inputs_out_': 2,
                'min_samples_': 1,
            },
        ),
        # Basic, with episode feature
        (
            pykoop.SplitPipeline(
                lifting_functions_state=None,
                lifting_functions_input=None,
            ),
            np.array([
                [0, 0, 0, 0, 1, 1],
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array([
                [0, 0, 0, 0, 1, 1],
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            2,
            True,
            {
                'n_features_in_': 5,
                'n_states_in_': 2,
                'n_inputs_in_': 2,
                'n_features_out_': 5,
                'n_states_out_': 2,
                'n_inputs_out_': 2,
                'min_samples_': 1,
            },
        ),
        # Lifting only state
        (
            pykoop.SplitPipeline(
                lifting_functions_state=[(
                    'pl',
                    pykoop.PolynomialLiftingFn(order=2),
                )],
                lifting_functions_input=None,
            ),
            np.array([
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array([
                # State
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [0, 1, 4, 9, 16, 25],
                [0, 4, 6, 6, 4, 0],
                [25, 16, 9, 4, 1, 0],
                # Input
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            2,
            False,
            {
                'n_features_in_': 4,
                'n_states_in_': 2,
                'n_inputs_in_': 2,
                'n_features_out_': 7,
                'n_states_out_': 5,
                'n_inputs_out_': 2,
                'min_samples_': 1,
            },
        ),
        # Lifting only input
        (
            pykoop.SplitPipeline(
                lifting_functions_state=None,
                lifting_functions_input=[(
                    'pl',
                    pykoop.PolynomialLiftingFn(order=2),
                )],
            ),
            np.array([
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array([
                # State
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                # Input
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
                [16, 25, 36, 49, 64, 81],
                [0, 40, 42, 42, 40, 36],
                [0, 64, 49, 36, 25, 16],
            ]).T,
            2,
            False,
            {
                'n_features_in_': 4,
                'n_states_in_': 2,
                'n_inputs_in_': 2,
                'n_features_out_': 7,
                'n_states_out_': 2,
                'n_inputs_out_': 5,
                'min_samples_': 1,
            },
        ),
        # Lifting both
        (
            pykoop.SplitPipeline(
                lifting_functions_state=[(
                    'pla',
                    pykoop.PolynomialLiftingFn(order=2, interaction_only=True),
                )],
                lifting_functions_input=[(
                    'plb',
                    pykoop.PolynomialLiftingFn(order=2),
                )],
            ),
            np.array([
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array([
                # State
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [0, 4, 6, 6, 4, 0],
                # Input
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
                [16, 25, 36, 49, 64, 81],
                [0, 40, 42, 42, 40, 36],
                [0, 64, 49, 36, 25, 16],
            ]).T,
            2,
            False,
            {
                'n_features_in_': 4,
                'n_states_in_': 2,
                'n_inputs_in_': 2,
                'n_features_out_': 8,
                'n_states_out_': 3,
                'n_inputs_out_': 5,
                'min_samples_': 1,
            },
        ),
    ])
class TestSplitPipeline:
    """Test :class:`SplitPipeline`."""

    def test_split_lifting_fn_attrs(self, lf, X, Xt_exp, n_inputs,
                                    episode_feature, attr_exp):
        """Test expected :class:`SplitPipeline` object attributes."""
        # Fit estimator
        lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        # Check attributes
        attr = {key: getattr(lf, key) for key in attr_exp.keys()}
        assert attr == attr_exp

    def test_split_lifting_fn_transform(self, lf, X, Xt_exp, n_inputs,
                                        episode_feature, attr_exp):
        """Test :class:`SplitPipeline` transform."""
        # Fit estimator
        lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        Xt = lf.transform(X)
        np.testing.assert_allclose(Xt, Xt_exp)

    def test_split_lifting_fn_inverse_transform(self, lf, X, Xt_exp, n_inputs,
                                                episode_feature, attr_exp):
        """Test :class:`SplitPipeline` inverse transform."""
        # Fit estimator
        lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        Xt = lf.transform(X)
        Xi = lf.inverse_transform(Xt)
        np.testing.assert_allclose(Xi, X)


@pytest.mark.parametrize('lf', [
    pykoop.PolynomialLiftingFn(order=2),
    pykoop.KoopmanPipeline(
        lifting_functions=[
            ('p', pykoop.PolynomialLiftingFn(order=2)),
        ],
        regressor=pykoop.Edmd(),
    )
])
class TestLiftRetract:
    """Test estimators with :class:`_LiftRetractMixin`.

    Attributes
    ----------
    X : np.ndarray
        Data matrix.
    Xt : np.ndarray
        Transformed data matrix.

    .. todo::
        Break up multiple asserts.
    """

    X = np.array([
        [0, 1, 2, 3, 4, 5],
        [0, 2, 4, 5, 6, 10],
    ]).T

    Xt = np.array([
        [0, 1, 2, 3, 4, 5],
        [0, 1, 4, 9, 16, 25],
        [0, 2, 4, 5, 6, 10],
        [0, 2, 8, 15, 24, 50],
        [0, 4, 16, 25, 36, 100],
    ]).T

    def test_lift_retract_ff(self, lf):
        """Test with no episode feature during fit or lift/retract."""
        lf.fit(self.X, n_inputs=1, episode_feature=False)
        # Test lift
        Xt_l = lf.lift(self.X, episode_feature=False)
        np.testing.assert_allclose(Xt_l, self.Xt)
        # Test retract
        X_r = lf.retract(Xt_l, episode_feature=False)
        np.testing.assert_allclose(X_r, self.X)
        # Test lift state
        Xt_l = lf.lift_state(
            self.X[:, :lf.n_states_in_],
            episode_feature=False,
        )
        np.testing.assert_allclose(Xt_l, self.Xt[:, :lf.n_states_out_])
        # # Test retract state
        X_r = lf.retract_state(Xt_l, episode_feature=False)
        np.testing.assert_allclose(X_r, self.X[:, :lf.n_states_in_])
        # Test lift input
        Xt_l = lf.lift_input(self.X, episode_feature=False)
        np.testing.assert_allclose(Xt_l, self.Xt[:, lf.n_states_out_:])
        # Test retract input
        X_r = lf.retract_input(Xt_l, episode_feature=False)
        np.testing.assert_allclose(X_r, self.X[:, lf.n_states_in_:])

    def test_lift_retract_tt(self, lf):
        """Test with episode feature during fit and lift/retract."""
        # Add episode features
        X = np.hstack((
            np.array([0, 0, 0, 1, 1, 1]).reshape((-1, 1)),
            self.X,
        ))
        Xt = np.hstack((
            np.array([0, 0, 0, 1, 1, 1]).reshape((-1, 1)),
            self.Xt,
        ))
        lf.fit(X, n_inputs=1, episode_feature=True)
        # Test lift
        Xt_l = lf.lift(X, episode_feature=True)
        np.testing.assert_allclose(Xt_l, Xt)
        # Test retract
        X_r = lf.retract(Xt_l, episode_feature=True)
        np.testing.assert_allclose(X_r, X)
        # Test lift state
        Xt_l = lf.lift_state(X[:, :lf.n_states_in_ + 1], episode_feature=True)
        np.testing.assert_allclose(Xt_l, Xt[:, :lf.n_states_out_ + 1])
        # # Test retract state
        X_r = lf.retract_state(Xt_l, episode_feature=True)
        np.testing.assert_allclose(X_r, X[:, :lf.n_states_in_ + 1])
        # Test lift input
        Xt_l = lf.lift_input(X, episode_feature=True)
        np.testing.assert_allclose(Xt_l[:, 0], Xt[:, 0])
        np.testing.assert_allclose(
            Xt_l[:, 1:],
            Xt[:, lf.n_states_out_ + 1:],
        )
        # Test retract input
        X_r = lf.retract_input(Xt_l, episode_feature=True)
        np.testing.assert_allclose(X_r[:, 0], X[:, 0])
        np.testing.assert_allclose(X_r[:, 1:], X[:, lf.n_states_in_ + 1:])

    def test_lift_retract_ft(self, lf):
        """Test with episode feature during fit but not lift/retract."""
        # Add episode features
        X = np.hstack((
            np.array([0, 0, 0, 1, 1, 1]).reshape((-1, 1)),
            self.X,
        ))
        Xt = np.hstack((
            np.array([0, 0, 0, 1, 1, 1]).reshape((-1, 1)),
            self.Xt,
        ))
        lf.fit(X, n_inputs=1, episode_feature=True)
        # Test lift
        Xt_l = lf.lift(self.X, episode_feature=False)
        np.testing.assert_allclose(Xt_l, self.Xt)
        # Test retract
        X_r = lf.retract(Xt_l, episode_feature=False)
        np.testing.assert_allclose(X_r, self.X)
        # Test lift state
        Xt_l = lf.lift_state(
            self.X[:, :lf.n_states_in_],
            episode_feature=False,
        )
        np.testing.assert_allclose(Xt_l, self.Xt[:, :lf.n_states_out_])
        # # Test retract state
        X_r = lf.retract_state(Xt_l, episode_feature=False)
        np.testing.assert_allclose(X_r, self.X[:, :lf.n_states_in_])
        # Test lift input
        Xt_l = lf.lift_input(self.X, episode_feature=False)
        np.testing.assert_allclose(Xt_l, self.Xt[:, lf.n_states_out_:])
        # Test retract input
        X_r = lf.retract_input(Xt_l, episode_feature=False)
        np.testing.assert_allclose(X_r, self.X[:, lf.n_states_in_:])

    def test_lift_retract_tf(self, lf):
        """Test with no episode feature during fit but with in lift/retract."""
        # Add episode features
        X = np.hstack((
            np.array([0, 0, 0, 1, 1, 1]).reshape((-1, 1)),
            self.X,
        ))
        Xt = np.hstack((
            np.array([0, 0, 0, 1, 1, 1]).reshape((-1, 1)),
            self.Xt,
        ))
        lf.fit(self.X, n_inputs=1, episode_feature=False)
        # Test lift
        Xt_l = lf.lift(X, episode_feature=True)
        np.testing.assert_allclose(Xt_l, Xt)
        # Test retract
        X_r = lf.retract(Xt_l, episode_feature=True)
        np.testing.assert_allclose(X_r, X)
        # Test lift state
        Xt_l = lf.lift_state(X[:, :lf.n_states_in_ + 1], episode_feature=True)
        np.testing.assert_allclose(Xt_l, Xt[:, :lf.n_states_out_ + 1])
        # # Test retract state
        X_r = lf.retract_state(Xt_l, episode_feature=True)
        np.testing.assert_allclose(X_r, X[:, :lf.n_states_in_ + 1])
        # Test lift input
        Xt_l = lf.lift_input(X, episode_feature=True)
        np.testing.assert_allclose(Xt_l[:, 0], Xt[:, 0])
        np.testing.assert_allclose(
            Xt_l[:, 1:],
            Xt[:, lf.n_states_out_ + 1:],
        )
        # Test retract input
        X_r = lf.retract_input(Xt_l, episode_feature=True)
        np.testing.assert_allclose(X_r[:, 0], X[:, 0])
        np.testing.assert_allclose(X_r[:, 1:], X[:, lf.n_states_in_ + 1:])


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
