"""Test :mod:`regressors`."""

import numpy as np
import pytest
import sklearn.linear_model
import sklearn.utils.estimator_checks

import pykoop


@pytest.mark.parametrize(
    'regressor, mass_spring_damper, fit_tol, predict_tol',
    [
        (
            pykoop.Edmd(),
            'mass_spring_damper_no_input',
            1e-5,
            1e-5,
        ),
        (
            pykoop.EdmdMeta(),
            'mass_spring_damper_no_input',
            1e-5,
            1e-5,
        ),
        (
            pykoop.EdmdMeta(regressor=sklearn.linear_model.Ridge(alpha=0)),
            'mass_spring_damper_no_input',
            1e-5,
            1e-5,
        ),
        (
            pykoop.Dmdc(),
            'mass_spring_damper_no_input',
            1e-5,
            1e-5,
        ),
        (
            pykoop.Dmd(),
            'mass_spring_damper_no_input',
            1e-5,
            1e-5,
        ),
        (
            pykoop.Edmd(),
            'mass_spring_damper_sine_input',
            1e-1,
            1e-3,
        ),
        (
            pykoop.EdmdMeta(),
            'mass_spring_damper_sine_input',
            1e-1,
            1e-3,
        ),
        (
            pykoop.EdmdMeta(regressor=sklearn.linear_model.Ridge(alpha=0)),
            'mass_spring_damper_sine_input',
            1e-1,
            1e-3,
        ),
        (
            pykoop.Dmdc(),
            'mass_spring_damper_sine_input',
            1e-1,
            1e-3,
        ),
    ],
)
class TestRegressorsExact:
    """Test regressors with exact solutions."""

    def test_fit(
        self,
        request,
        regressor,
        mass_spring_damper,
        fit_tol,
        predict_tol,
    ):
        """Test fit accuracy by comparing Koopman matrix."""
        # Get fixture from name
        msd = request.getfixturevalue(mass_spring_damper)
        # Fit regressor
        regressor.fit(
            msd['X_train'],
            msd['Xp_train'],
            n_inputs=msd['n_inputs'],
            episode_feature=msd['episode_feature'],
        )
        # Test value of Koopman operator
        np.testing.assert_allclose(
            regressor.coef_.T,
            msd['U_valid'],
            atol=fit_tol,
            rtol=0,
        )

    def test_predict(
        self,
        request,
        regressor,
        mass_spring_damper,
        fit_tol,
        predict_tol,
    ):
        """Test fit accuracy by comparing prediction."""
        # Get fixture from name
        msd = request.getfixturevalue(mass_spring_damper)
        # Fit regressor
        regressor.fit(
            msd['X_train'],
            msd['Xp_train'],
            n_inputs=msd['n_inputs'],
            episode_feature=msd['episode_feature'],
        )
        # Test prediction
        prediction = regressor.predict(msd['X_valid'])
        np.testing.assert_allclose(
            prediction,
            msd['Xp_valid'],
            atol=predict_tol,
            rtol=0,
        )


class TestDataRegressorExact:
    """Test :class:`DataRegressor` with exact solutions."""

    def test_fit(self, request):
        """Test fit accuracy by comparing Koopman matrix."""
        # Get fixture from name
        msd = request.getfixturevalue('mass_spring_damper_sine_input')
        regressor = pykoop.DataRegressor(coef=msd['U_valid'].T)
        # Fit regressor
        regressor.fit(
            msd['X_train'],
            msd['Xp_train'],
            n_inputs=msd['n_inputs'],
            episode_feature=msd['episode_feature'],
        )
        # Test value of Koopman operator
        np.testing.assert_allclose(
            regressor.coef_.T,
            msd['U_valid'],
            atol=1e-5,
            rtol=0,
        )

    def test_predict(self, request):
        """Test fit accuracy by comparing prediction."""
        # Get fixture from name
        msd = request.getfixturevalue('mass_spring_damper_sine_input')
        regressor = pykoop.DataRegressor(coef=msd['U_valid'].T)
        # Fit regressor
        regressor.fit(
            msd['X_train'],
            msd['Xp_train'],
            n_inputs=msd['n_inputs'],
            episode_feature=msd['episode_feature'],
        )
        # Test prediction
        prediction = regressor.predict(msd['X_valid'])
        np.testing.assert_allclose(
            prediction,
            msd['Xp_valid'],
            atol=1e-3,
            rtol=0,
        )


@pytest.mark.parametrize(
    'regressor, mass_spring_damper',
    [
        (
            pykoop.Dmdc(
                tsvd_unshifted=pykoop.Tsvd('known_noise', 1),
                tsvd_shifted=pykoop.Tsvd('known_noise', 1),
            ),
            'mass_spring_damper_no_input',
        ),
        (
            pykoop.Dmd(tsvd=pykoop.Tsvd('known_noise', 1)),
            'mass_spring_damper_no_input',
        ),
        (
            pykoop.EdmdMeta(regressor=sklearn.linear_model.Ridge(
                alpha=1, random_state=1234)),
            'mass_spring_damper_sine_input',
        ),
        (
            pykoop.EdmdMeta(regressor=sklearn.linear_model.Lasso(
                alpha=1, random_state=1234)),
            'mass_spring_damper_sine_input',
        ),
        (
            pykoop.EdmdMeta(regressor=sklearn.linear_model.ElasticNet(
                alpha=1, random_state=1234)),
            'mass_spring_damper_sine_input',
        ),
        (
            pykoop.EdmdMeta(
                regressor=sklearn.linear_model.OrthogonalMatchingPursuit(
                    n_nonzero_coefs=2)),
            'mass_spring_damper_sine_input',
        ),
    ],
)
class TestRegressorsRegression:
    """Run regression tests for regressors without easy-to-check solutions.

    Attributes
    ----------
    tol : float
        Tolerance for regression test.
    """

    tol = 1e-12

    def test_fit_predict(
        self,
        request,
        ndarrays_regression,
        regressor,
        mass_spring_damper,
    ):
        """Test fit accuracy by comparing Koopman matrix and prediction."""
        # Get fixture from name
        msd = request.getfixturevalue(mass_spring_damper)
        # Fit regressor
        regressor.fit(
            msd['X_train'],
            msd['Xp_train'],
            n_inputs=msd['n_inputs'],
            episode_feature=msd['episode_feature'],
        )
        # Compute prediction
        prediction = regressor.predict(msd['X_valid'])
        # Compare to prior results
        ndarrays_regression.check(
            {
                'regressor.coef_': regressor.coef_,
                'prediction': prediction,
            },
            default_tolerance=dict(atol=self.tol, rtol=0),
        )


class TestSkLearn:
    """Test ``scikit-learn`` compatibility."""

    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.Edmd(),
        pykoop.EdmdMeta(),
        pykoop.EdmdMeta(regressor=sklearn.linear_model.Ridge(alpha=1)),
        pykoop.Dmdc(),
        pykoop.Dmd(),
        pykoop.DataRegressor(),
    ])
    def test_compatible_estimator(self, estimator, check):
        """Test ``scikit-learn`` compatibility of estimators."""
        check(estimator)


class TestExceptions:
    """Test a selection of invalid estimator parameter."""

    X = np.array([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
    ])

    @pytest.mark.parametrize(
        'estimator',
        [
            pykoop.Edmd(alpha=-1),
            pykoop.Dmdc(mode_type='blah'),
            pykoop.Dmd(mode_type='blah'),
            pykoop.DataRegressor(coef=np.eye(2)),  # Wrong dimensions for data
        ])
    def test_invalid_params(self, estimator):
        """Test a selection of invalid estimator parameter."""
        with pytest.raises(ValueError):
            estimator.fit(self.X)
