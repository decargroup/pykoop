"""Test :mod:`regressors`."""

import numpy as np
import pytest
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


class TestSklearn:
    """Test scikit-learn compatibility."""

    @sklearn.utils.estimator_checks.parametrize_with_checks([
        pykoop.Edmd(),
        pykoop.Dmdc(),
        pykoop.Dmd(),
    ])
    def test_compatible_estimator(self, estimator, check):
        """Test scikit-learn compatibility of estimators."""
        check(estimator)


class TestExceptions:
    """Test a selection of invalid estimator parameter."""

    X = np.array([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
    ])

    @pytest.mark.parametrize('estimator', [
        pykoop.Edmd(alpha=-1),
        pykoop.Dmdc(mode_type='blah'),
        pykoop.Dmd(mode_type='blah'),
    ])
    def test_invalid_params(self, estimator):
        """Test a selection of invalid estimator parameter."""
        with pytest.raises(ValueError):
            estimator.fit(self.X)
