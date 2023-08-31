"""Test :mod:`koopman_pipeline`."""

import numpy as np
import pandas
import pytest
import sklearn.metrics
import sklearn.preprocessing
import sklearn.utils.estimator_checks

import pykoop


@pytest.mark.parametrize(
    'lf, names_in, X, names_out, Xt_exp, n_inputs, episode_feature, attr_exp',
    [
        (
            pykoop.KoopmanPipeline(
                lifting_functions=None,
                regressor=None,
            ),
            np.array(['x0', 'x1', 'u0']),
            np.array([
                [1, 2, 3, 4, 5, 6],
                [-1, -2, -3, -4, -5, -6],
                [2, 4, 6, 8, 10, 12],
            ]).T,
            np.array(['x0', 'x1', 'u0']),
            np.array([
                [1, 2, 3, 4, 5, 6],
                [-1, -2, -3, -4, -5, -6],
                [2, 4, 6, 8, 10, 12],
            ]).T,
            1,
            False,
            {
                'n_features_in_': 3,
                'n_states_in_': 2,
                'n_inputs_in_': 1,
                'n_features_out_': 3,
                'n_states_out_': 2,
                'n_inputs_out_': 1,
                'min_samples_': 1,
            },
        ),
        (
            pykoop.KoopmanPipeline(
                lifting_functions=[(
                    'dl',
                    pykoop.DelayLiftingFn(
                        n_delays_state=1,
                        n_delays_input=1,
                    ),
                )],
                regressor=None,
            ),
            np.array(['x0', 'x1', 'u0']),
            np.array([
                [1, 2, 3, 4, 5, 6],
                [-1, -2, -3, -4, -5, -6],
                [2, 4, 6, 8, 10, 12],
            ]).T,
            np.array([
                'x0',
                'x1',
                'D1(x0)',
                'D1(x1)',
                'u0',
                'D1(u0)',
            ]),
            np.array([
                # State
                [2, 3, 4, 5, 6],
                [-2, -3, -4, -5, -6],
                [1, 2, 3, 4, 5],
                [-1, -2, -3, -4, -5],
                # Input
                [4, 6, 8, 10, 12],
                [2, 4, 6, 8, 10],
            ]).T,
            1,
            False,
            {
                'n_features_in_': 3,
                'n_states_in_': 2,
                'n_inputs_in_': 1,
                'n_features_out_': 6,
                'n_states_out_': 4,
                'n_inputs_out_': 2,
                'min_samples_': 2,
            },
        ),
        (
            pykoop.KoopmanPipeline(
                lifting_functions=[
                    ('pl', pykoop.PolynomialLiftingFn(order=2)),
                    ('sc',
                     pykoop.SkLearnLiftingFn(
                         sklearn.preprocessing.MaxAbsScaler())),
                ],
                regressor=None,
            ),
            np.array(['x0', 'u0']),
            np.array([
                [1, 2, 3, 4, 5, 10],
                [-1, -2, -3, -4, -5, -10],
            ]).T,
            np.array([
                'MaxAbsScaler(x0)',
                'MaxAbsScaler(x0^2)',
                'MaxAbsScaler(u0)',
                'MaxAbsScaler(x0*u0)',
                'MaxAbsScaler(u0^2)',
            ]),
            np.array([
                [1 / 10, 2 / 10, 3 / 10, 4 / 10, 5 / 10, 1],
                [1 / 100, 4 / 100, 9 / 100, 16 / 100, 25 / 100, 1],
                # Inputs
                [-1 / 10, -2 / 10, -3 / 10, -4 / 10, -5 / 10, -1],
                [-1 / 100, -4 / 100, -9 / 100, -16 / 100, -25 / 100, -1],
                [1 / 100, 4 / 100, 9 / 100, 16 / 100, 25 / 100, 1],
            ]).T,
            1,
            False,
            {
                'n_features_in_': 2,
                'n_states_in_': 1,
                'n_inputs_in_': 1,
                'n_features_out_': 5,
                'n_states_out_': 2,
                'n_inputs_out_': 3,
                'min_samples_': 1,
            },
        ),
        (
            pykoop.KoopmanPipeline(
                lifting_functions=[
                    ('sc',
                     pykoop.SkLearnLiftingFn(
                         sklearn.preprocessing.MaxAbsScaler())),
                    ('pl', pykoop.PolynomialLiftingFn(order=2)),
                ],
                regressor=None,
            ),
            np.array(['x0', 'u0']),
            np.array([
                [1, 2, 3, 4, 5, 10],
                [-1, -2, -3, -4, -5, -10],
            ]).T,
            np.array([
                'MaxAbsScaler(x0)',
                'MaxAbsScaler(x0)^2',
                'MaxAbsScaler(u0)',
                'MaxAbsScaler(x0)*MaxAbsScaler(u0)',
                'MaxAbsScaler(u0)^2',
            ]),
            np.array([
                [1 / 10, 2 / 10, 3 / 10, 4 / 10, 5 / 10, 1],
                [1 / 100, 4 / 100, 9 / 100, 16 / 100, 25 / 100, 1],
                # Inputs
                [-1 / 10, -2 / 10, -3 / 10, -4 / 10, -5 / 10, -1],
                [-1 / 100, -4 / 100, -9 / 100, -16 / 100, -25 / 100, -1],
                [1 / 100, 4 / 100, 9 / 100, 16 / 100, 25 / 100, 1],
            ]).T,
            1,
            False,
            {
                'n_features_in_': 2,
                'n_states_in_': 1,
                'n_inputs_in_': 1,
                'n_features_out_': 5,
                'n_states_out_': 2,
                'n_inputs_out_': 3,
                'min_samples_': 1,
            },
        ),
    ],
)
class TestKoopmanPipelineTransform:
    """Test :class:`KoopmanPipeline` transform and inverse transform."""

    def test_koopman_pipeline_attrs(self, lf, names_in, X, names_out, Xt_exp,
                                    n_inputs, episode_feature, attr_exp):
        """Test expected :class:`KoopmanPipeline` object attributes."""
        # Fit estimator
        lf.fit_transformers(
            X,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        # Check attributes
        attr = {key: getattr(lf, key) for key in attr_exp.keys()}
        assert attr == attr_exp

    def test_transform(self, lf, names_in, X, names_out, Xt_exp, n_inputs,
                       episode_feature, attr_exp):
        """Test :class:`KoopmanPipeline` transform."""
        # Fit estimator
        lf.fit_transformers(
            X,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        Xt = lf.transform(X)
        np.testing.assert_allclose(Xt, Xt_exp)

    def test_inverse_transform(self, lf, names_in, X, names_out, Xt_exp,
                               n_inputs, episode_feature, attr_exp):
        """Test :class:`KoopmanPipeline` inverse transform."""
        # Fit estimator
        lf.fit_transformers(
            X,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        Xt = lf.transform(X)
        Xi = lf.inverse_transform(Xt)
        np.testing.assert_allclose(Xi, X)

    def test_feature_names_in(self, lf, names_in, X, names_out, Xt_exp,
                              n_inputs, episode_feature, attr_exp):
        """Test input feature names."""
        lf.fit_transformers(
            X,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        names_in_actual = lf.get_feature_names_in()
        assert np.all(names_in == names_in_actual)
        assert names_in_actual.dtype == object

    def test_feature_names_out(self, lf, names_in, X, names_out, Xt_exp,
                               n_inputs, episode_feature, attr_exp):
        """Test input feature names."""
        lf.fit_transformers(
            X,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        names_out_actual = lf.get_feature_names_out()
        assert np.all(names_out == names_out_actual)
        assert names_out_actual.dtype == object


class TestKoopmanPipelineFit:
    """Test Koopman pipeline fit."""

    def test_fit(self, mass_spring_damper_sine_input):
        """Test Koopman pipeline fit on a mass-spring-damper."""
        kp = pykoop.KoopmanPipeline(
            lifting_functions=[(
                'dl',
                pykoop.DelayLiftingFn(
                    n_delays_state=0,
                    n_delays_input=0,
                ),
            )],
            regressor=pykoop.Edmd(),
        )
        kp.fit(
            mass_spring_damper_sine_input['X_train'],
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=mass_spring_damper_sine_input['episode_feature'],
        )
        np.testing.assert_allclose(
            kp.regressor_.coef_.T,
            mass_spring_damper_sine_input['U_valid'],
            atol=0.1,
        )

    def test_fit_feature_names(self, mass_spring_damper_sine_input):
        """Test Koopman pipeline fit feature names."""
        kp = pykoop.KoopmanPipeline(
            lifting_functions=[(
                'dl',
                pykoop.DelayLiftingFn(
                    n_delays_state=0,
                    n_delays_input=0,
                ),
            )],
            regressor=pykoop.Edmd(),
        )
        X = mass_spring_damper_sine_input['X_train']
        names = [f'x{k}' for k in range(X.shape[1])]
        X_names = pandas.DataFrame(X, columns=names)
        kp.fit(
            X_names,
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=mass_spring_damper_sine_input['episode_feature'],
        )
        # Test fails if ``lf.transform()`` raises an exception because the
        # feature names do not match.
        kp.transform(X_names)
        assert all(kp.feature_names_in_ == names)

    def test_frequency_response(
        self,
        ndarrays_regression,
        mass_spring_damper_sine_input,
    ):
        """Test Koopman pipeline frequency response."""
        kp = pykoop.KoopmanPipeline(regressor=pykoop.Edmd())
        kp.fit(
            mass_spring_damper_sine_input['X_train'],
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=mass_spring_damper_sine_input['episode_feature'],
        )
        freq, mag = kp.frequency_response(
            t_step=mass_spring_damper_sine_input['t_step'], )
        ndarrays_regression.check(
            {
                'freq': freq,
                'mag': mag,
            },
            default_tolerance=dict(atol=1e-6, rtol=0),
        )


@pytest.mark.filterwarnings('ignore:Score `score=')
@pytest.mark.filterwarnings('ignore:Prediction diverged or error occured')
@pytest.mark.filterwarnings('ignore:overflow encountered in square')
class TestKoopmanPipelineScore:
    """Test Koopman pipeline scoring."""

    @pytest.mark.parametrize(
        'X_predicted, X_expected, n_steps, discount_factor, '
        'regression_metric, error_score, min_samples, episode_feature, '
        'score_exp',
        [
            (
                np.array([
                    [1, 2, 3, 4],
                    [2, 3, 3, 2],
                ]).T,
                np.array([
                    [1, 2, 3, 4],
                    [2, 3, 3, 2],
                ]).T,
                None,
                1,
                'neg_mean_squared_error',
                np.nan,
                1,
                False,
                0,
            ),
            (
                np.array([
                    [1, 2],
                    [2, 3],
                ]).T,
                np.array([
                    [1, 4],
                    [2, 2],
                ]).T,
                None,
                1,
                'neg_mean_squared_error',
                np.nan,
                1,
                False,
                -np.mean([2**2, 1]),
            ),
            (
                np.array([
                    [1, 2, 3, 5],
                ]).T,
                np.array([
                    [1, 2, 3, 3],
                ]).T,
                None,
                1,
                'neg_mean_squared_error',
                np.nan,
                1,
                False,
                -np.mean([0, 0, 2**2]),
            ),
            (
                np.array([
                    [1, 2, 3, 5],
                ]).T,
                np.array([
                    [1, 2, 3, 3],
                ]).T,
                None,
                1,
                'neg_mean_absolute_error',
                np.nan,
                1,
                False,
                -np.mean([0, 0, 2]),
            ),
            (
                np.array([
                    [1, 2, 3, 5],
                ]).T,
                np.array([
                    [1, 2, 3, 4],
                ]).T,
                None,
                1,
                'neg_mean_squared_error',
                np.nan,
                2,
                False,
                -np.mean([0, 1]),
            ),
            (
                np.array([
                    [1, 2, 3, 5],
                ]).T,
                np.array([
                    [1, 2, 3, 4],
                ]).T,
                2,
                1,
                'neg_mean_squared_error',
                np.nan,
                1,
                False,
                0,
            ),
            (
                np.array([
                    [1, 2, 3, 5],
                ]).T,
                np.array([
                    [1, 2, 3, 4],
                ]).T,
                None,
                0.5,
                'neg_mean_squared_error',
                np.nan,
                1,
                False,
                # (0 * 1 + 0 * 0.5 + 1 * 0.25) / (1 + 0.5 + 0.25)
                -0.25 / 1.75,
            ),
            (
                np.array([
                    [0, 0, 0, 1, 1, 1],
                    [1, 2, 3, 4, 5, 6],
                ]).T,
                np.array([
                    [0, 0, 0, 1, 1, 1],
                    [1, 2, 4, 4, 6, 6],
                ]).T,
                None,
                1,
                'neg_mean_squared_error',
                np.nan,
                1,
                True,
                -np.mean([0, 1, 1, 0]),
            ),
            (
                np.array([
                    [0, 0, 0, 1, 1, 1],
                    [1, 2, 3, 4, 5, 6],
                ]).T,
                np.array([
                    [0, 0, 0, 1, 1, 1],
                    [1, 2, 4, 4, 6, 6],
                ]).T,
                1,
                1,
                'neg_mean_squared_error',
                np.nan,
                1,
                True,
                -np.mean([0, 1]),
            ),
            (
                np.array([
                    [0, 0, 0, 1, 1, 1],
                    [1, 2, 3, 4, 5, 6],
                ]).T,
                np.array([
                    [0, 0, 0, 1, 1, 1],
                    [1, 2, 4, 4, 6, 6],
                ]).T,
                None,
                0.5,
                'neg_mean_squared_error',
                np.nan,
                1,
                True,
                -(0.5 + 1) / (1 + 0.5 + 1 + 0.5),
            ),
            (
                np.array([
                    [0, 0, 0, 1, 1, 1],
                    [1, 2, 3, 4, 5, 6],
                ]).T,
                np.array([
                    [0, 0, 0, 1, 1, 1],
                    [1, 2, 4, 4, 6, 6],
                ]).T,
                1,
                0.5,
                'neg_mean_squared_error',
                np.nan,
                1,
                True,
                -(0 + 1) / (1 + 0 + 1 + 0),
            ),
            (
                np.array([
                    [1, np.nan, 3, 4],
                    [2, 3, 3, 2],
                ]).T,
                np.array([
                    [1, 2, 3, 4],
                    [2, 3, 3, 2],
                ]).T,
                None,
                1,
                'neg_mean_squared_error',
                np.nan,
                1,
                False,
                np.nan,
            ),
            (
                np.array([
                    [1, np.nan, 3, 4],
                    [2, 3, 3, 2],
                ]).T,
                np.array([
                    [1, 2, 3, 4],
                    [2, 3, 3, 2],
                ]).T,
                None,
                1,
                'neg_mean_squared_error',
                -100,
                1,
                False,
                -100,
            ),
            (
                np.array([
                    [1, np.nan, 3, 4],
                    [2, 3, 3, 2],
                ]).T,
                np.array([
                    [1, 2, 3, 4],
                    [2, 3, 3, 2],
                ]).T,
                None,
                1,
                'neg_mean_squared_error',
                'raise',
                1,
                False,
                None,
            ),
            (
                np.array([
                    [1e-2, 1e-3],
                ]).T,
                np.array([
                    [1e150, 1e250],
                ]).T,
                None,
                1,
                'neg_mean_squared_error',
                -100,
                1,
                False,
                -100,
            ),
            (
                np.array([
                    [1e-2, 1e-3],
                ]).T,
                np.array([
                    [1e150, 1e250],
                ]).T,
                None,
                1,
                'neg_mean_squared_error',
                'raise',
                1,
                False,
                None,
            ),
            # Finite score worse than error score should return error score.
            (
                np.array([
                    [1e-2, 1e-3],
                ]).T,
                np.array([
                    [1e5, 1e6],
                ]).T,
                None,
                1,
                'neg_mean_squared_error',
                -100,
                1,
                False,
                -100,
            ),
        ],
    )
    def test_score_trajectory(
        self,
        X_predicted,
        X_expected,
        n_steps,
        discount_factor,
        regression_metric,
        error_score,
        min_samples,
        episode_feature,
        score_exp,
    ):
        if (error_score == 'raise') and (score_exp is None):
            with pytest.raises(ValueError):
                pykoop.score_trajectory(
                    X_predicted=X_predicted,
                    X_expected=X_expected,
                    n_steps=n_steps,
                    discount_factor=discount_factor,
                    regression_metric=regression_metric,
                    error_score=error_score,
                    min_samples=min_samples,
                    episode_feature=episode_feature,
                )
        else:
            score = pykoop.score_trajectory(
                X_predicted=X_predicted,
                X_expected=X_expected,
                n_steps=n_steps,
                discount_factor=discount_factor,
                regression_metric=regression_metric,
                error_score=error_score,
                min_samples=min_samples,
                episode_feature=episode_feature,
            )
            np.testing.assert_allclose(score, score_exp)

    @pytest.mark.filterwarnings('ignore:invalid value encountered in matmul')
    @pytest.mark.filterwarnings('ignore:overflow encountered in multiply')
    @pytest.mark.parametrize(
        'kp, scorer, X_training, X_validation, episode_feature, n_inputs', [
            (
                pykoop.KoopmanPipeline(regressor=pykoop.Edmd()),
                pykoop.KoopmanPipeline.make_scorer(),
                np.array([
                    [1, 2, 3, 4],
                    [2, 3, 3, 2],
                ]).T,
                np.array([
                    [1, 2, 1, 4],
                    [1, 3, 3, 2],
                ]).T,
                False,
                0,
            ),
            (
                pykoop.KoopmanPipeline(regressor=pykoop.Edmd()),
                pykoop.KoopmanPipeline.make_scorer(
                    n_steps=2,
                    discount_factor=0.9,
                    regression_metric='r2',
                ),
                np.array([
                    [1, 2, 3, 4],
                    [2, 3, 3, 2],
                ]).T,
                np.array([
                    [1, 2, 1, 4],
                    [1, 3, 3, 2],
                ]).T,
                False,
                0,
            ),
            (
                pykoop.KoopmanPipeline(
                    lifting_functions=[
                        ('poly', pykoop.PolynomialLiftingFn(order=10))
                    ],
                    regressor=pykoop.Edmd(),
                ),
                pykoop.KoopmanPipeline.make_scorer(error_score=-100),
                pykoop.example_data_msd()['X_train'],
                pykoop.example_data_msd()['X_valid'],
                True,
                1,
            ),
            (
                pykoop.KoopmanPipeline(regressor=pykoop.Edmd()),
                pykoop.KoopmanPipeline.make_scorer(
                    n_steps=2,
                    discount_factor=0.9,
                    regression_metric='explained_variance',
                    regression_metric_kw={'multioutput': 'variance_weighted'},
                ),
                np.array([
                    [1, 2, 3, 4],
                    [2, 3, 3, 2],
                ]).T,
                np.array([
                    [1, 2, 1, 4],
                    [1, 3, 3, 2],
                ]).T,
                False,
                0,
            ),
            (
                pykoop.KoopmanPipeline(regressor=pykoop.Edmd()),
                pykoop.KoopmanPipeline.make_scorer(
                    n_steps=2,
                    discount_factor=0.9,
                    regression_metric=sklearn.metrics.r2_score,
                ),
                np.array([
                    [1, 2, 3, 4],
                    [2, 3, 3, 2],
                ]).T,
                np.array([
                    [1, 2, 1, 4],
                    [1, 3, 3, 2],
                ]).T,
                False,
                0,
            ),
        ])
    def test_make_scorer_regression(
        self,
        ndarrays_regression,
        kp,
        scorer,
        X_training,
        X_validation,
        episode_feature,
        n_inputs,
    ):
        kp.fit(X_training, episode_feature=episode_feature, n_inputs=0)
        score_default = kp.score(X_validation)
        score_scorer = scorer(kp, X_validation, None)
        ndarrays_regression.check(
            {
                'score_default': score_default,
                'score_scorer': score_scorer,
            },
            default_tolerance=dict(atol=1e-6, rtol=0),
        )

    @pytest.mark.parametrize(
        'X, w_exp, n_steps, discount_factor, episode_feature',
        [
            (
                np.array([
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ]).T,
                np.array([1, 1, 0, 0]),
                2,
                1,
                False,
            ),
            (
                np.array([
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ]).T,
                np.array([1, 0.5, 0.25, 0]),
                3,
                0.5,
                False,
            ),
            (
                np.array([
                    [0, 0, 0, 1, 1, 1, 1],
                    [1, 2, 3, 4, 5, 6, 7],
                    [5, 6, 7, 8, 9, 10, 11],
                ]).T,
                np.array([1, 0.5, 0, 1, 0.5, 0, 0]),
                2,
                0.5,
                True,
            ),
            (
                np.array([
                    [0, 0, 0, 1, 1, 1, 1],
                    [1, 2, 3, 4, 5, 6, 7],
                    [5, 6, 7, 8, 9, 10, 11],
                ]).T,
                np.array([1, 1, 1, 1, 1, 1, 1]),
                10,
                1,
                True,
            ),
            (
                np.array([
                    [0, 0, 0, 1, 1, 1, 1],
                    [1, 2, 3, 4, 5, 6, 7],
                    [5, 6, 7, 8, 9, 10, 11],
                ]).T,
                np.array([1, 0.1, 0.01, 1, 0.1, 0.01, 0.001]),
                10,
                0.1,
                True,
            ),
        ],
    )
    def test_weights_from_data_matrix(
        self,
        X,
        w_exp,
        n_steps,
        discount_factor,
        episode_feature,
    ):
        """Test weight generation for scoring."""
        w = pykoop.koopman_pipeline._weights_from_data_matrix(
            X,
            n_steps,
            discount_factor,
            episode_feature,
        )
        np.testing.assert_allclose(w, w_exp)


class TestEpisodeManipulation:
    """Test episode manipulation, including shifting and splitting."""

    @pytest.mark.parametrize(
        'X, X_unsh_exp, X_sh_exp, n_inputs, episode_feature',
        [
            # Test without input or episode feature
            (
                np.array([
                    [1, 2, 3, 4, 5],
                    [-1, -2, -3, -4, -5],
                ]).T,
                np.array([
                    [1, 2, 3, 4],
                    [-1, -2, -3, -4],
                ]).T,
                np.array([
                    [2, 3, 4, 5],
                    [-2, -3, -4, -5],
                ]).T,
                0,
                False,
            ),
            # Test with input, without episode feature
            (
                np.array([
                    [1, 2, 3, 4, 5],
                    [-1, -2, -3, -4, -5],
                ]).T,
                np.array([
                    [1, 2, 3, 4],
                    [-1, -2, -3, -4],
                ]).T,
                np.array([
                    [2, 3, 4, 5],
                ]).T,
                1,
                False,
            ),
            # Test without input, with episode feature
            (
                np.array([
                    [0, 0, 0, 1, 1],
                    [1, 2, 3, 4, 5],
                    [-1, -2, -3, -4, -5],
                ]).T,
                np.array([
                    [0, 0, 1],
                    [1, 2, 4],
                    [-1, -2, -4],
                ]).T,
                np.array([
                    [0, 0, 1],
                    [2, 3, 5],
                    [-2, -3, -5],
                ]).T,
                0,
                True,
            ),
            # Test with input and episode feature
            (
                np.array([
                    [0, 0, 1, 1, 1],
                    [1, 2, 3, 4, 5],
                    [-1, -2, -3, -4, -5],
                ]).T,
                np.array([
                    [0, 1, 1],
                    [1, 3, 4],
                    [-1, -3, -4],
                ]).T,
                np.array([
                    [0, 1, 1],
                    [2, 4, 5],
                ]).T,
                1,
                True,
            ),
        ],
    )
    def test_shift_episodes(
        self,
        X,
        X_unsh_exp,
        X_sh_exp,
        n_inputs,
        episode_feature,
    ):
        """Test :func:`shift_episodes`.

        .. todo:: Break up multiple asserts.
        """
        X_unsh, X_sh = pykoop.shift_episodes(X, n_inputs, episode_feature)
        np.testing.assert_allclose(X_unsh, X_unsh_exp)
        np.testing.assert_allclose(X_sh, X_sh_exp)

    @pytest.mark.parametrize(
        'X, ic_exp, min_samples, n_inputs, episode_feature',
        [
            (
                np.array([
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                ]).T,
                np.array([
                    [1],
                    [4],
                ]).T,
                1,
                0,
                False,
            ),
            (
                np.array([
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                ]).T,
                np.array([
                    [1, 2],
                    [4, 5],
                ]).T,
                2,
                0,
                False,
            ),
            (
                np.array([
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                    [5, 5, 5, 5],
                ]).T,
                np.array([
                    [1],
                    [4],
                ]).T,
                1,
                1,
                False,
            ),
            (
                np.array([
                    [0, 0, 1, 1],
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                ]).T,
                np.array([
                    [0, 1],
                    [1, 3],
                    [4, 6],
                ]).T,
                1,
                0,
                True,
            ),
            (
                np.array([
                    [0, 0, 1, 1],
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                    [9, 9, 9, 9],
                ]).T,
                np.array([
                    [0, 1],
                    [1, 3],
                    [4, 6],
                ]).T,
                1,
                1,
                True,
            ),
            (
                np.array([
                    [0, 0, 0, 1, 1, 1],
                    [1, 2, 2, 3, 4, 5],
                    [4, 5, 5, 6, 7, 6],
                    [9, 9, 9, 9, 9, 6],
                ]).T,
                np.array([
                    [0, 0, 1, 1],
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                ]).T,
                2,
                1,
                True,
            ),
        ],
    )
    def test_extract_initial_conditions(
        self,
        X,
        ic_exp,
        min_samples,
        n_inputs,
        episode_feature,
    ):
        """Test :func:`extract_initial_conditions`."""
        ic = pykoop.extract_initial_conditions(
            X,
            min_samples,
            n_inputs,
            episode_feature,
        )
        np.testing.assert_allclose(ic, ic_exp)

    @pytest.mark.parametrize(
        'X, u_exp, n_inputs, episode_feature',
        [
            (
                np.array([
                    [1, 2, 3, 4],
                    [6, 7, 8, 9],
                ]).T,
                np.array([
                    [6, 7, 8, 9],
                ]).T,
                1,
                False,
            ),
            (
                np.array([
                    [1, 2, 3, 4],
                    [6, 7, 8, 9],
                ]).T,
                np.array([]).reshape((0, 4)).T,
                0,
                False,
            ),
            (
                np.array([
                    [0, 0, 1, 1],
                    [1, 2, 3, 4],
                    [6, 7, 8, 9],
                ]).T,
                np.array([
                    [0, 0, 1, 1],
                    [6, 7, 8, 9],
                ]).T,
                1,
                True,
            ),
        ],
    )
    def test_extract_input(self, X, u_exp, n_inputs, episode_feature):
        """Test :func:`extract_input`."""
        u = pykoop.extract_input(X, n_inputs, episode_feature)
        np.testing.assert_allclose(u, u_exp)

    def test_strip_initial_conditons(self):
        """Test :func:`strip_initial_conditions`."""
        X1 = np.array([
            [0, 0, 1, 1, 1, 2, 2, 2],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 3, 4, 5, 6, 7, 8, 9],
        ]).T
        X2 = np.array([
            [0, 0, 1, 1, 1, 2, 2, 2],
            [-1, 2, -1, 4, 5, -1, 7, 8],
            [-1, 3, -1, 5, 6, -1, 8, 9],
        ]).T
        X1s = pykoop.strip_initial_conditions(
            X1,
            min_samples=1,
            episode_feature=True,
        )
        X2s = pykoop.strip_initial_conditions(
            X2,
            min_samples=1,
            episode_feature=True,
        )
        np.testing.assert_allclose(X1s, X2s)


@pytest.mark.filterwarnings(
    'ignore:Call to deprecated method predict_multistep')
@pytest.mark.parametrize(
    'kp',
    [
        pykoop.KoopmanPipeline(
            lifting_functions=[
                ('dl', pykoop.DelayLiftingFn(n_delays_state=1,
                                             n_delays_input=1))
            ],
            regressor=pykoop.Edmd(),
        ),
        pykoop.KoopmanPipeline(
            lifting_functions=[
                ('dla',
                 pykoop.DelayLiftingFn(n_delays_state=2, n_delays_input=2)),
                ('dlb',
                 pykoop.DelayLiftingFn(n_delays_state=2, n_delays_input=2)),
            ],
            regressor=pykoop.Edmd(),
        ),
        pykoop.KoopmanPipeline(
            lifting_functions=[
                ('dla',
                 pykoop.DelayLiftingFn(n_delays_state=2, n_delays_input=1)),
                ('ply', pykoop.PolynomialLiftingFn(order=2)),
                ('dlb',
                 pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=2)),
            ],
            regressor=pykoop.Edmd(),
        ),
        pykoop.KoopmanPipeline(
            lifting_functions=[
                (
                    'sp',
                    pykoop.SplitPipeline(
                        lifting_functions_state=[
                            ('pl', pykoop.PolynomialLiftingFn(order=2)),
                        ],
                        lifting_functions_input=None,
                    ),
                ),
                ('dl', pykoop.DelayLiftingFn(n_delays_state=2,
                                             n_delays_input=2)),
            ],
            regressor=pykoop.Edmd(),
        ),
        pykoop.KoopmanPipeline(
            lifting_functions=[
                ('dla',
                 pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=1)),
                (
                    'sp',
                    pykoop.SplitPipeline(
                        lifting_functions_state=[
                            ('pla', pykoop.PolynomialLiftingFn(order=2)),
                            ('plb', pykoop.PolynomialLiftingFn(order=2)),
                        ],
                        lifting_functions_input=[
                            ('plc', pykoop.PolynomialLiftingFn(order=2)),
                        ],
                    ),
                ),
                ('dlb',
                 pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=1)),
            ],
            regressor=pykoop.Edmd(),
        ),
    ],
)
class TestPrediction:
    """Test fit Koopman pipeline prediction."""

    def test_predict_trajectory(self, kp, mass_spring_damper_sine_input):
        """Test :func:`predict_trajectory`."""
        msg = 'Test only works when there is no episode feature.'
        assert (not mass_spring_damper_sine_input['episode_feature']), msg
        # Fit estimator
        kp.fit(
            mass_spring_damper_sine_input['X_train'],
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=False,
        )
        # Extract initial conditions
        x0 = pykoop.extract_initial_conditions(
            mass_spring_damper_sine_input['X_train'],
            kp.min_samples_,
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=False,
        )
        # Extract input
        u = pykoop.extract_input(
            mass_spring_damper_sine_input['X_train'],
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=False,
        )
        # Predict new states
        X_sim = kp.predict_trajectory(
            x0,
            u,
            episode_feature=False,
            relift_state=True,
        )
        # Predict manually
        X_sim_exp = np.zeros(X_sim.shape)
        X_sim_exp[:kp.min_samples_, :] = x0
        for k in range(kp.min_samples_, u.shape[0]):
            X = np.hstack((
                X_sim_exp[(k - kp.min_samples_):k, :],
                u[(k - kp.min_samples_):k, :],
            ))
            Xp = kp.predict(X)
            X_sim_exp[[k], :] = Xp[[-1], :]
        np.testing.assert_allclose(X_sim, X_sim_exp)

    def test_predict_trajectory_no_U(self, kp, mass_spring_damper_sine_input):
        """Test :func:`predict_trajectory`."""
        msg = 'Test only works when there is no episode feature.'
        assert (not mass_spring_damper_sine_input['episode_feature']), msg
        # Fit estimator
        kp.fit(
            mass_spring_damper_sine_input['X_train'],
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=False,
        )
        # Extract initial conditions
        x0 = pykoop.extract_initial_conditions(
            mass_spring_damper_sine_input['X_train'],
            kp.min_samples_,
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=False,
        )
        # Extract input
        u = pykoop.extract_input(
            mass_spring_damper_sine_input['X_train'],
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=False,
        )
        # Predict new states
        X_sim = kp.predict_trajectory(
            mass_spring_damper_sine_input['X_train'],
            U=None,
            episode_feature=False,
            relift_state=True,
        )
        # Predict manually
        X_sim_exp = np.zeros(X_sim.shape)
        X_sim_exp[:kp.min_samples_, :] = x0
        for k in range(kp.min_samples_, u.shape[0]):
            X = np.hstack((
                X_sim_exp[(k - kp.min_samples_):k, :],
                u[(k - kp.min_samples_):k, :],
            ))
            Xp = kp.predict(X)
            X_sim_exp[[k], :] = Xp[[-1], :]
        np.testing.assert_allclose(X_sim, X_sim_exp)

    def test_predict_trajectory_no_relift_state(self, ndarrays_regression, kp,
                                                mass_spring_damper_sine_input):
        """Test :func:`predict_trajectory` without relifting state."""
        msg = 'Test only works when there is no episode feature.'
        assert (not mass_spring_damper_sine_input['episode_feature']), msg
        # Fit estimator
        kp.fit(
            mass_spring_damper_sine_input['X_train'],
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=False,
        )
        # Extract initial conditions
        x0 = pykoop.extract_initial_conditions(
            mass_spring_damper_sine_input['X_train'],
            kp.min_samples_,
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=False,
        )
        # Extract input
        u = pykoop.extract_input(
            mass_spring_damper_sine_input['X_train'],
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=False,
        )
        # Predict new states
        X_sim = kp.predict_trajectory(
            x0,
            u,
            episode_feature=False,
            relift_state=False,
        )
        ndarrays_regression.check(
            {
                'X_sim': X_sim,
            },
            default_tolerance=dict(atol=1e-6, rtol=0),
        )

    def test_predict_multistep(self, kp, mass_spring_damper_sine_input):
        """Test :func:`predict_multistep` (deprecated)."""
        msg = 'Test only works when there is no episode feature.'
        assert (not mass_spring_damper_sine_input['episode_feature']), msg
        # Extract initial conditions
        x0 = pykoop.extract_initial_conditions(
            mass_spring_damper_sine_input['X_train'],
            kp.min_samples_,
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=mass_spring_damper_sine_input['episode_feature'],
        )
        # Extract input
        u = pykoop.extract_input(
            mass_spring_damper_sine_input['X_train'],
            n_inputs=mass_spring_damper_sine_input['n_inputs'],
            episode_feature=mass_spring_damper_sine_input['episode_feature'],
        )
        # Set up initial conditions and input
        X_ic = np.zeros(mass_spring_damper_sine_input['X_train'].shape)
        X_ic[:kp.min_samples_, :x0.shape[1]] = x0
        X_ic[:, x0.shape[1]:] = u
        # Predict using ``predict_multistep``
        X_sim = kp.predict_multistep(X_ic)
        # Predict manually
        X_sim_exp = np.zeros(mass_spring_damper_sine_input['Xp_train'].shape)
        X_sim_exp[:kp.min_samples_, :] = x0
        for k in range(kp.min_samples_, u.shape[0]):
            X = np.hstack((
                X_sim_exp[(k - kp.min_samples_):k, :],
                u[(k - kp.min_samples_):k, :],
            ))
            Xp = kp.predict(X)
            X_sim_exp[[k], :] = Xp[[-1], :]
        np.testing.assert_allclose(X_sim, X_sim_exp)


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
        # Negative episode
        (
            np.array([
                [0, 0, 0, -1, -1, -1],
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
            ]).T,
            'raise',
            True,
        ),
        # Fractional episode
        (
            np.array([
                [0, 0, 0, 1.1, 1.1, 1.1],
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
            ]).T,
            'raise',
            True,
        ),
    ],
)
class TestSplitCombineEpisodes:
    """Test :func:`split_episodes` and :func:`combine_episodes`."""

    def test_split_episodes(self, X, episodes, episode_feature):
        """Test :func:`split_episodes`."""
        if episodes == 'raise':
            with pytest.raises(ValueError):
                pykoop.split_episodes(X, episode_feature=episode_feature)
        else:
            episodes_actual = pykoop.split_episodes(
                X, episode_feature=episode_feature)
            episodes_expected = list(sorted(episodes))
            zipped_episodes = zip(episodes_actual, episodes_expected)
            if episode_feature:
                X_ep = X[:, 0]
                X = X[:, 1:]
                for (i_act, X_i_act), (i_exp, X_i_exp) in zipped_episodes:
                    assert i_act == i_exp
                    np.testing.assert_allclose(X_i_act, X_i_exp)
            else:
                assert len(episodes) == 1
                X_exp = episodes[0][1]
                np.testing.assert_allclose(X, X_exp)

    def test_combine_episodes(self, X, episodes, episode_feature):
        """Test :func:`combine_episodes`."""
        if episodes == 'raise':
            with pytest.raises(ValueError):
                pykoop.combine_episodes(
                    episodes,
                    episode_feature=episode_feature,
                )
        else:
            X_actual = pykoop.combine_episodes(
                episodes,
                episode_feature=episode_feature,
            )
            np.testing.assert_allclose(X_actual, X)


@pytest.mark.parametrize(
    'lf, names_in, X, names_out, Xt_exp, n_inputs, episode_feature, attr_exp',
    [
        # Basic, without episode feature
        (
            pykoop.SplitPipeline(
                lifting_functions_state=None,
                lifting_functions_input=None,
            ),
            np.array(['x0', 'x1', 'u0', 'u1']),
            np.array([
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array(['x0', 'x1', 'u0', 'u1']),
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
            np.array(['ep', 'x0', 'x1', 'u0', 'u1']),
            np.array([
                [0, 0, 0, 0, 1, 1],
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array(['ep', 'x0', 'x1', 'u0', 'u1']),
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
            np.array(['x0', 'x1', 'u0', 'u1']),
            np.array([
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array([
                'x0',
                'x1',
                'x0^2',
                'x0*x1',
                'x1^2',
                'u0',
                'u1',
            ]),
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
            np.array(['x0', 'x1', 'u0', 'u1']),
            np.array([
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array([
                'x0',
                'x1',
                'u0',
                'u1',
                'u0^2',
                'u0*u1',
                'u1^2',
            ]),
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
            np.array(['x0', 'x1', 'u0', 'u1']),
            np.array([
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array([
                'x0',
                'x1',
                'x0*x1',
                'u0',
                'u1',
                'u0^2',
                'u0*u1',
                'u1^2',
            ]),
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
        # State delay
        (
            pykoop.SplitPipeline(
                lifting_functions_state=[('dl', pykoop.DelayLiftingFn(1, 0))],
                lifting_functions_input=None,
            ),
            np.array(['ep', 'x0', 'x1', 'u0', 'u1']),
            np.array([
                # ep
                [0, 0, 0, 0, 1, 1],
                # x
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                # u
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array(['ep', 'x0', 'x1', 'D1(x0)', 'D1(x1)', 'u0', 'u1']),
            np.array([
                # ep
                [0, 0, 0, 1],
                # x
                [1, 2, 3, 5],
                [4, 3, 2, 0],
                [0, 1, 2, 4],
                [5, 4, 3, 1],
                # u
                [5, 6, 7, 9],
                [8, 7, 6, 4],
            ]).T,
            2,
            True,
            {
                'n_features_in_': 5,
                'n_states_in_': 2,
                'n_inputs_in_': 2,
                'n_features_out_': 7,
                'n_states_out_': 4,
                'n_inputs_out_': 2,
                'min_samples_': 2,
            },
        ),
        # Input delay
        (
            pykoop.SplitPipeline(
                lifting_functions_state=None,
                lifting_functions_input=[('dl', pykoop.DelayLiftingFn(0, 1))],
            ),
            np.array(['ep', 'x0', 'x1', 'u0', 'u1']),
            np.array([
                # ep
                [0, 0, 0, 0, 1, 1],
                # x
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                # u
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array(['ep', 'x0', 'x1', 'u0', 'u1', 'D1(u0)', 'D1(u1)']),
            np.array([
                # ep
                [0, 0, 0, 1],
                # x
                [1, 2, 3, 5],
                [4, 3, 2, 0],
                # u
                [5, 6, 7, 9],
                [8, 7, 6, 4],
                [4, 5, 6, 8],
                [0, 8, 7, 5],
            ]).T,
            2,
            True,
            {
                'n_features_in_': 5,
                'n_states_in_': 2,
                'n_inputs_in_': 2,
                'n_features_out_': 7,
                'n_states_out_': 2,
                'n_inputs_out_': 4,
                'min_samples_': 2,
            },
        ),
        # State delay with nonzero ``n_delays_input``
        (
            pykoop.SplitPipeline(
                lifting_functions_state=[('dl', pykoop.DelayLiftingFn(1, 1))],
                lifting_functions_input=None,
            ),
            np.array(['ep', 'x0', 'x1', 'u0', 'u1']),
            np.array([
                # ep
                [0, 0, 0, 0, 1, 1],
                # x
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                # u
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array(['ep', 'x0', 'x1', 'D1(x0)', 'D1(x1)', 'u0', 'u1']),
            np.array([
                # ep
                [0, 0, 0, 1],
                # x
                [1, 2, 3, 5],
                [4, 3, 2, 0],
                [0, 1, 2, 4],
                [5, 4, 3, 1],
                # u
                [5, 6, 7, 9],
                [8, 7, 6, 4],
            ]).T,
            2,
            True,
            {
                'n_features_in_': 5,
                'n_states_in_': 2,
                'n_inputs_in_': 2,
                'n_features_out_': 7,
                'n_states_out_': 4,
                'n_inputs_out_': 2,
                'min_samples_': 2,
            },
        ),
        # Input delay with nonzero ``n_delays_state``
        (
            pykoop.SplitPipeline(
                lifting_functions_state=None,
                lifting_functions_input=[('dl', pykoop.DelayLiftingFn(1, 1))],
            ),
            np.array(['ep', 'x0', 'x1', 'u0', 'u1']),
            np.array([
                # ep
                [0, 0, 0, 0, 1, 1],
                # x
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                # u
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array(['ep', 'x0', 'x1', 'u0', 'u1', 'D1(u0)', 'D1(u1)']),
            np.array([
                # ep
                [0, 0, 0, 1],
                # x
                [1, 2, 3, 5],
                [4, 3, 2, 0],
                # u
                [5, 6, 7, 9],
                [8, 7, 6, 4],
                [4, 5, 6, 8],
                [0, 8, 7, 5],
            ]).T,
            2,
            True,
            {
                'n_features_in_': 5,
                'n_states_in_': 2,
                'n_inputs_in_': 2,
                'n_features_out_': 7,
                'n_states_out_': 2,
                'n_inputs_out_': 4,
                'min_samples_': 2,
            },
        ),
        # State and input delay
        (
            pykoop.SplitPipeline(
                lifting_functions_state=[('dls', pykoop.DelayLiftingFn(1, 0))],
                lifting_functions_input=[('dli', pykoop.DelayLiftingFn(0, 1))],
            ),
            np.array(['ep', 'x0', 'x1', 'u0', 'u1']),
            np.array([
                # ep
                [0, 0, 0, 0, 1, 1],
                # x
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                # u
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array([
                'ep', 'x0', 'x1', 'D1(x0)', 'D1(x1)', 'u0', 'u1', 'D1(u0)',
                'D1(u1)'
            ]),
            np.array([
                # ep
                [0, 0, 0, 1],
                # x
                [1, 2, 3, 5],
                [4, 3, 2, 0],
                [0, 1, 2, 4],
                [5, 4, 3, 1],
                # u
                [5, 6, 7, 9],
                [8, 7, 6, 4],
                [4, 5, 6, 8],
                [0, 8, 7, 5],
            ]).T,
            2,
            True,
            {
                'n_features_in_': 5,
                'n_states_in_': 2,
                'n_inputs_in_': 2,
                'n_features_out_': 9,
                'n_states_out_': 4,
                'n_inputs_out_': 4,
                'min_samples_': 2,
            },
        ),
        # State and input delay with nonzero input/state delays
        (
            pykoop.SplitPipeline(
                lifting_functions_state=[('dls', pykoop.DelayLiftingFn(1, 1))],
                lifting_functions_input=[('dli', pykoop.DelayLiftingFn(1, 1))],
            ),
            np.array(['ep', 'x0', 'x1', 'u0', 'u1']),
            np.array([
                # ep
                [0, 0, 0, 0, 1, 1],
                # x
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                # u
                [4, 5, 6, 7, 8, 9],
                [0, 8, 7, 6, 5, 4],
            ]).T,
            np.array([
                'ep', 'x0', 'x1', 'D1(x0)', 'D1(x1)', 'u0', 'u1', 'D1(u0)',
                'D1(u1)'
            ]),
            np.array([
                # ep
                [0, 0, 0, 1],
                # x
                [1, 2, 3, 5],
                [4, 3, 2, 0],
                [0, 1, 2, 4],
                [5, 4, 3, 1],
                # u
                [5, 6, 7, 9],
                [8, 7, 6, 4],
                [4, 5, 6, 8],
                [0, 8, 7, 5],
            ]).T,
            2,
            True,
            {
                'n_features_in_': 5,
                'n_states_in_': 2,
                'n_inputs_in_': 2,
                'n_features_out_': 9,
                'n_states_out_': 4,
                'n_inputs_out_': 4,
                'min_samples_': 2,
            },
        ),
    ],
)
class TestSplitPipeline:
    """Test :class:`SplitPipeline`."""

    def test_split_lifting_fn_attrs(self, lf, names_in, X, names_out, Xt_exp,
                                    n_inputs, episode_feature, attr_exp):
        """Test expected :class:`SplitPipeline` object attributes."""
        # Fit estimator
        lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        # Check attributes
        attr = {key: getattr(lf, key) for key in attr_exp.keys()}
        assert attr == attr_exp

    def test_split_lifting_fn_transform(self, lf, names_in, X, names_out,
                                        Xt_exp, n_inputs, episode_feature,
                                        attr_exp):
        """Test :class:`SplitPipeline` transform."""
        # Fit estimator
        lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        Xt = lf.transform(X)
        np.testing.assert_allclose(Xt, Xt_exp)

    def test_split_lifting_fn_inverse_transform(self, lf, names_in, X,
                                                names_out, Xt_exp, n_inputs,
                                                episode_feature, attr_exp):
        """Test :class:`SplitPipeline` inverse transform."""
        # Fit estimator
        lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        Xt = lf.transform(X)
        Xt_inv = lf.inverse_transform(Xt)
        # If the number of delays for ``x`` and ``u`` are different, only the
        # last samples will be the same in each episode. Must compare the last
        # samples of each episode to ensure correctness.
        if episode_feature:
            episodes = []
            for i in pandas.unique(X[:, 0]):
                # Select episode and inverse
                X_i = X[X[:, 0] == i, :]
                Xt_inv_i = Xt_inv[Xt_inv[:, 0] == i, :]
                episodes.append(X_i[-Xt_inv_i.shape[0]:, :])
            Xt_inv_trimmed = np.vstack(episodes)
        else:
            Xt_inv_trimmed = X[-Xt_inv.shape[0]:, :]
        np.testing.assert_allclose(Xt_inv_trimmed, Xt_inv)

    def test_feature_names_in(self, lf, names_in, X, names_out, Xt_exp,
                              n_inputs, episode_feature, attr_exp):
        """Test input feature names."""
        lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        names_in_actual = lf.get_feature_names_in()
        assert np.all(names_in == names_in_actual)
        assert names_in_actual.dtype == object

    def test_feature_names_out(self, lf, names_in, X, names_out, Xt_exp,
                               n_inputs, episode_feature, attr_exp):
        """Test input feature names."""
        lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        names_out_actual = lf.get_feature_names_out()
        assert np.all(names_out == names_out_actual)
        assert names_out_actual.dtype == object


@pytest.mark.parametrize(
    'lf',
    [
        pykoop.PolynomialLiftingFn(order=2),
        pykoop.KoopmanPipeline(
            lifting_functions=[
                ('p', pykoop.PolynomialLiftingFn(order=2)),
            ],
            regressor=pykoop.Edmd(),
        )
    ],
)
class TestLiftRetract:
    """Test estimators with :class:`_LiftRetractMixin`.

    Attributes
    ----------
    X : np.ndarray
        Data matrix.
    Xt : np.ndarray
        Transformed data matrix.

    .. todo:: Break up multiple asserts.
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


class TestFeatureNames:
    """Test feature naming."""

    string_names = np.array(['a', 'b', 'c'], dtype=object)

    different_string_names = np.array(['c', 'b', 'c'], dtype=object)

    numerical_names = np.array([1, 2, 3], dtype=object)

    mixed_names = np.array(['a', 2, 'c'], dtype=object)

    numerical_data = np.array([
        [1, 2, 3, 4, 5, 6],
        [-1, -2, -3, -4, -5, -6],
        [2, 4, 6, 8, 10, 12],
    ]).T

    def test_valid_names(self):
        """Test invalid feature names."""
        X = pandas.DataFrame(self.numerical_data, columns=self.string_names)
        kp = pykoop.KoopmanPipeline(
            lifting_functions=None,
            regressor=pykoop.Edmd(),
        )
        kp.fit(X)
        kp.transform(X)

    def test_invalid_names(self):
        """Test invalid feature names."""
        X_invalid = pandas.DataFrame(
            self.numerical_data,
            columns=self.mixed_names,
        )
        kp = pykoop.KoopmanPipeline(
            lifting_functions=None,
            regressor=pykoop.Edmd(),
        )
        kp.fit(X_invalid)
        assert kp.feature_names_in_ is None

    def test_numerical_names(self):
        """Test numerical feature names."""
        X_invalid = pandas.DataFrame(
            self.numerical_data,
            columns=self.numerical_names,
        )
        kp = pykoop.KoopmanPipeline(
            lifting_functions=None,
            regressor=pykoop.Edmd(),
        )
        kp.fit(X_invalid)
        assert kp.feature_names_in_ is None

    @pytest.mark.parametrize('X_fit, X_transform', [
        (
            pandas.DataFrame(
                numerical_data,
                columns=string_names,
            ),
            pandas.DataFrame(
                numerical_data,
                columns=mixed_names,
            ),
        ),
        (
            pandas.DataFrame(
                numerical_data,
                columns=mixed_names,
            ),
            pandas.DataFrame(
                numerical_data,
                columns=string_names,
            ),
        ),
        (
            pandas.DataFrame(
                numerical_data,
                columns=string_names,
            ),
            pandas.DataFrame(
                numerical_data,
                columns=different_string_names,
            ),
        ),
    ])
    def test_different_fit_transform(self, X_fit, X_transform):
        """Test numerical feature names."""
        kp = pykoop.KoopmanPipeline(
            lifting_functions=None,
            regressor=pykoop.Edmd(),
        )
        kp.fit(X_fit)
        with pytest.raises(ValueError):
            kp.transform(X_transform)


@pytest.mark.parametrize(
    'kp, X, n_inputs, episode_feature',
    [
        (
            pykoop.KoopmanPipeline(regressor=pykoop.Edmd()),
            np.array([
                [0, 0, 0, 1, 1, 1],
                [1, 2, 3, 4, 5, 7],
                [6, 5, 4, 3, 2, 1],
            ]).T,
            1,
            True,
        ),
    ],
)
class TestSplitStateInputEpisodes:
    """Test :func:`pykoop.KoopmanPipeline._split_state_input_episodes`."""

    def test_X_initial_onearg(self, kp, X, n_inputs, episode_feature):
        """Test state with one argument (input is ``None``)."""
        kp.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        episodes = kp._split_state_input_episodes(X, None, episode_feature)
        # Extract initial conditions
        x0 = pykoop.extract_initial_conditions(
            X,
            kp.min_samples_,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        X_actual = pykoop.combine_episodes(
            [(ep[0], ep[1]) for ep in episodes],
            episode_feature=episode_feature,
        )
        np.testing.assert_allclose(X_actual, x0)

    def test_U_onearg(self, kp, X, n_inputs, episode_feature):
        """Test input with one argument (input is ``None``)."""
        kp.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        episodes = kp._split_state_input_episodes(X, None, episode_feature)
        # Extract input
        u = pykoop.extract_input(
            X,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        u_actual = pykoop.combine_episodes(
            [(ep[0], ep[2]) for ep in episodes],
            episode_feature=episode_feature,
        )
        np.testing.assert_allclose(u_actual, u)

    def test_X_initial_twoarg(self, kp, X, n_inputs, episode_feature):
        """Test state with two arguments (input is not ``None``)."""
        # Extract initial conditions
        x0 = pykoop.extract_initial_conditions(
            X,
            kp.min_samples_,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        # Extract input
        u = pykoop.extract_input(
            X,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        kp.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        episodes = kp._split_state_input_episodes(x0, u, episode_feature)
        ep_X0 = pykoop.split_episodes(
            x0,
            episode_feature=episode_feature,
        )
        # ep_U = split_episodes(U, episode_feature=episode_feature)
        for e in range(len(episodes)):
            np.testing.assert_allclose(episodes[e][1], ep_X0[e][1])

    def test_U_twoarg(self, kp, X, n_inputs, episode_feature):
        """Test input with two arguments (input is not ``None``)."""
        # Extract initial conditions
        x0 = pykoop.extract_initial_conditions(
            X,
            kp.min_samples_,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        # Extract input
        u = pykoop.extract_input(
            X,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        kp.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
        episodes = kp._split_state_input_episodes(x0, u, episode_feature)
        ep_U = pykoop.split_episodes(u, episode_feature=episode_feature)
        for e in range(len(episodes)):
            np.testing.assert_allclose(episodes[e][2], ep_U[e][1])


class TestSkLearn:
    """Test ``scikit-learn`` compatibility."""

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
        """Test ``scikit-learn`` compatibility of estimators."""
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
        assert kp.get_params()['p__order'] == 2
