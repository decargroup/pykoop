"""Collection of regressors for use with or without a Koopman Pipeline.

All of the lifting functions included in this module adhere to the interface
defined in :class:`KoopmanRegressor`.
"""

import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
import sklearn.base
import sklearn.linear_model
import sklearn.utils.validation
from scipy import linalg

from . import koopman_pipeline, tsvd

# Create logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Edmd(koopman_pipeline.KoopmanRegressor):
    """Extended Dynamic Mode Decomposition with Tikhonov regularization.

    Attributes
    ----------
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    coef_ : np.ndarray
        Fit coefficient matrix.

    Examples
    --------
    EDMD without regularization on mass-spring-damper data

    >>> kp = pykoop.KoopmanPipeline(regressor=pykoop.Edmd())
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)
    KoopmanPipeline(regressor=Edmd())

    EDMD with Tikhonov regularization on mass-spring-damper data

    >>> kp = pykoop.KoopmanPipeline(regressor=pykoop.Edmd(alpha=1))
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)
    KoopmanPipeline(regressor=Edmd(alpha=1))
    """

    def __init__(self, alpha: float = 0) -> None:
        """Instantiate :class:`Edmd`.

        Parameters
        ----------
        alpha : float
            Tikhonov regularization coefficient. Can be zero without
            introducing numerical problems.
        """
        self.alpha = alpha

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        Psi = X_unshifted.T
        Theta_p = X_shifted.T
        p, q = Psi.shape
        G = (Theta_p @ Psi.T) / q
        H_unreg = (Psi @ Psi.T) / q
        H_reg = H_unreg + (self.alpha * np.eye(p)) / q
        coef = linalg.lstsq(H_reg.T, G.T)[0]
        return coef

    def _validate_parameters(self) -> None:
        if self.alpha < 0:
            raise ValueError('Parameter `alpha` must be positive or zero.')


class EdmdMeta(koopman_pipeline.KoopmanRegressor):
    """Extended Dynamic Mode Decomposition with ``scikit-learn`` regressor.

    This meta-estimator lets you use any ``scikit-learn`` linear regressor to
    approximate the Koopman matrix. For example, you can use

    - :class:`sklearn.linear_model.LinearRegression`,
    - :class:`sklearn.linear_model.Ridge`,
    - :class:`sklearn.linear_model.ElasticNet`,
    - :class:`sklearn.linear_model.Lasso`, or
    - :class:`sklearn.linear_model.OrthogonalMatchingPursuit`.

    Attributes
    ----------
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    regressor_ : sklearn.base.BaseEstimator
        Fit ``scikit-learn`` regressor.
    coef_ : np.ndarray
        Fit coefficient matrix.

    Examples
    --------
    EDMD with the lasso regularizer on mass-spring-damper data

    >>> kp = pykoop.KoopmanPipeline(regressor=pykoop.EdmdMeta(
    ...     regressor=sklearn.linear_model.Lasso(alpha=1)))
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)
    KoopmanPipeline(regressor=EdmdMeta(regressor=Lasso(alpha=1)))
    """

    def __init__(self, regressor: sklearn.base.BaseEstimator = None) -> None:
        """Instantiate :class:`EdmdMeta`.

        Parameters
        ----------
        regressor : sklearn.base.BaseEstimator
            Wrapped ``scikit-learn`` regressor. Defaults to
            :class:`sklearn.linear_model.LinearRegression`.
        """
        self.regressor = regressor

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        self.regressor_ = (sklearn.base.clone(self.regressor)
                           if self.regressor is not None else
                           sklearn.linear_model.LinearRegression())
        if hasattr(self.regressor_, 'fit_intercept'):
            if self.regressor_.fit_intercept:
                log.warning('Regressor parameter `fit_intercept`  must be set '
                            'to `False` if present. Setting it `False` now.')
                self.regressor_.fit_intercept = False
        Psi = X_unshifted.T
        Theta_p = X_shifted.T
        p, q = Psi.shape
        G = (Theta_p @ Psi.T) / q
        H = (Psi @ Psi.T) / q
        self.regressor_.fit(H.T, G.T)
        coef = self.regressor_.coef_.T
        return coef

    def _validate_parameters(self) -> None:
        # No parameters need validation.
        pass


class Dmdc(koopman_pipeline.KoopmanRegressor):
    """Dynamic Mode Decomposition with control.

    Attributes
    ----------
    eigenvalues_ : np.ndarray
        DMD eigenvalues.
    modes_ : np.ndarray
        DMD modes (exact or projected depending on constructor params).
    B_tilde_ : np.ndarray
        ``B`` matrix in transformed basis.
    tsvd_unshifted_ : pykoop.Tsvd
        Fit truncated SVD object for unshifted data matrix.
    tsvd_shifted_ : pykoop.Tsvd
        Fit truncated SVD object for shifted data matrix.
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    coef_ : np.ndarray
        Fit coefficient matrix.

    Examples
    --------
    DMDc without singular value truncation on mass-spring-damper data

    >>> kp = pykoop.KoopmanPipeline(regressor=pykoop.Dmdc())
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)
    KoopmanPipeline(regressor=Dmdc())

    DMDc with singular value truncation on mass-spring-damper data

    >>> kp = pykoop.KoopmanPipeline(regressor=pykoop.Dmdc(
    ...     tsvd_unshifted=pykoop.Tsvd('rank', 1),
    ...     tsvd_shifted=pykoop.Tsvd('rank', 2)))
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)
    KoopmanPipeline(regressor=Dmdc(tsvd_shifted=Tsvd(truncation='rank',
    truncation_param=2), tsvd_unshifted=Tsvd(truncation='rank',
    truncation_param=1)))
    """

    def __init__(
        self,
        mode_type: str = 'projected',
        tsvd_unshifted: tsvd.Tsvd = None,
        tsvd_shifted: tsvd.Tsvd = None,
    ) -> None:
        """Instantiate :class:`Dmdc`.

        Parameters
        ----------
        mode_type : str
            DMD mode type, either ``'exact'`` or ``'projected'``.
        tsvd_unshifted : pykoop.Tsvd
            Singular value truncation method used to change basis of unshifted
            data matrix. If ``None``, economy SVD is used.
        tsvd_shifted : pykoop.Tsvd
            Singular value truncation method used to change basis of shifted
            data matrix. If ``None``, economy SVD is used.
        """
        self.mode_type = mode_type
        self.tsvd_unshifted = tsvd_unshifted
        self.tsvd_shifted = tsvd_shifted

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        Psi = X_unshifted.T
        Theta_p = X_shifted.T
        # Clone TSVDs
        self.tsvd_unshifted_ = (sklearn.base.clone(self.tsvd_unshifted)
                                if self.tsvd_unshifted is not None else
                                tsvd.Tsvd())
        self.tsvd_shifted_ = (sklearn.base.clone(self.tsvd_shifted) if
                              self.tsvd_shifted is not None else tsvd.Tsvd())
        # Compute truncated SVDs
        self.tsvd_unshifted_.fit(Psi)
        Q_tld = self.tsvd_unshifted_.left_singular_vectors_
        sig_tld = self.tsvd_unshifted_.singular_values_
        Z_tld = self.tsvd_unshifted_.right_singular_vectors_
        self.tsvd_shifted_.fit(Theta_p)
        Q_hat = self.tsvd_shifted_.left_singular_vectors_
        sig_hat = self.tsvd_shifted_.singular_values_
        Z_hat = self.tsvd_shifted_.right_singular_vectors_
        Sig_tld_inv = np.diag(1 / sig_tld)
        # Compute ``A`` and ``B``
        Q_tld_1 = Q_tld[:Theta_p.shape[0], :]
        Q_tld_2 = Q_tld[Theta_p.shape[0]:, :]
        A = Theta_p @ Z_tld @ Sig_tld_inv @ Q_tld_1.T
        B = Theta_p @ Z_tld @ Sig_tld_inv @ Q_tld_2.T
        # Compute ``A_tilde`` and ``B_tilde``
        A_tld = Q_hat.T @ Theta_p @ Z_tld @ Sig_tld_inv @ Q_tld_1.T @ Q_hat
        B_tld = Q_hat.T @ Theta_p @ Z_tld @ Sig_tld_inv @ Q_tld_2.T
        self.B_tilde_ = B_tld
        # Eigendecompose ``A``
        lmb, V_tld = linalg.eig(A_tld)
        self.eigenvalues_ = lmb
        # Compute DMD modes
        if self.mode_type == 'exact':
            V_exact = Theta_p @ Z_tld @ Sig_tld_inv @ Q_tld_1.T @ Q_hat @ V_tld
            self.modes_ = V_exact
        elif self.mode_type == 'projected':
            V_proj = Q_hat @ V_tld
            self.modes_ = V_proj
        else:
            # Already checked
            assert False
        # Reconstruct ``A`` and form Koopman matrix ``U``.
        Sigma = np.diag(self.eigenvalues_)
        A_r = np.real(
            linalg.lstsq(self.modes_.T, (self.modes_ @ Sigma).T)[0].T)
        coef = np.hstack((A_r, B)).T
        return coef

    def _validate_parameters(self) -> None:
        valid_mode_types = ['exact', 'projected']
        if self.mode_type not in valid_mode_types:
            raise ValueError(f'`mode_type` must be one of {valid_mode_types}')


class Dmd(koopman_pipeline.KoopmanRegressor):
    """Dynamic Mode Decomposition.

    Attributes
    ----------
    eigenvalues_ : np.ndarray
        DMD eigenvalues.
    modes_ : np.ndarray
        DMD modes (exact or projected depending on constructor params).
    tsvd_ : pykoop.Tsvd
        Fit truncated SVD object.
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    coef_ : np.ndarray
        Fit coefficient matrix.

    Examples
    --------
    DMD without singular value truncation on inputless mass-spring-damper data

    >>> kp = pykoop.KoopmanPipeline(regressor=pykoop.Dmd())
    >>> kp.fit(X_msd_no_input, n_inputs=0, episode_feature=True)
    KoopmanPipeline(regressor=Dmd())

    DMD with singular value truncation on inputless mass-spring-damper data

    >>> kp = pykoop.KoopmanPipeline(regressor=pykoop.Dmd(
    ...     tsvd=pykoop.Tsvd('known_noise', 1)))
    >>> kp.fit(X_msd_no_input, n_inputs=0, episode_feature=True)
    KoopmanPipeline(regressor=Dmd(tsvd=Tsvd(truncation='known_noise',
    truncation_param=1)))
    """

    # Override check parameters to skip ``check_fit2d_1sample`` sklearn test.
    _check_X_y_params: Dict[str, Any] = {
        'multi_output': True,
        'y_numeric': True,
        'ensure_min_samples': 2,
    }

    def __init__(self,
                 mode_type: str = 'projected',
                 tsvd: tsvd.Tsvd = None) -> None:
        """Instantiate :class:`Dmd`.

        Parameters
        ----------
        mode_type : str
            DMD mode type, either ``'exact'`` or ``'projected'``.
        tsvd : pykoop.Tsvd
            Truncated singular value object used to change bases. If ``None``,
            economy SVD is used.

        Warning
        -------
        :class:`Dmd` has some compatibility issues with ``scikit-learn``
        because the class has relatively strict input requirements.
        Specifically, both inputs must have the same number of features.
        Any ``scikit-learn`` unit tests that use different numbers of input
        and output features are therefore disabled.
        """
        self.mode_type = mode_type
        self.tsvd = tsvd

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        Psi = X_unshifted.T
        Psi_p = X_shifted.T
        # Clone TSVD
        self.tsvd_ = (sklearn.base.clone(self.tsvd)
                      if self.tsvd is not None else tsvd.Tsvd())
        # Compute truncated SVD
        self.tsvd_.fit(Psi)
        Q = self.tsvd_.left_singular_vectors_
        sigma = self.tsvd_.singular_values_
        Z = self.tsvd_.right_singular_vectors_
        # Compute ``U_tilde``
        Sigma_inv = np.diag(1 / sigma)
        U_tilde = Q.T @ Psi_p @ Z @ Sigma_inv
        # Eigendecompose ``U_tilde``
        lmb, V_tilde = linalg.eig(U_tilde)
        self.eigenvalues_ = lmb
        # Compute DMD modes
        if self.mode_type == 'exact':
            V_exact = Psi_p @ Z @ Sigma_inv @ V_tilde
            self.modes_ = V_exact
        elif self.mode_type == 'projected':
            V_proj = Q @ V_tilde
            self.modes_ = V_proj
        else:
            # Already checked
            assert False
        # Compute Koopman matrix
        Sigma = np.diag(self.eigenvalues_)
        U = linalg.lstsq(self.modes_.T, (self.modes_ @ Sigma).T)[0].T
        coef = np.real(U.T)
        return coef

    def _validate_parameters(self) -> None:
        valid_mode_types = ['exact', 'projected']
        if self.mode_type not in valid_mode_types:
            raise ValueError(f'`mode_type` must be one of {valid_mode_types}')

    def _more_tags(self):
        reason = ('The `dmd.Dmd` class requires X and y to have the same '
                  'number of features. This test does not meet that '
                  'requirement and must be skipped for now.')
        return {
            'multioutput': True,
            'multioutput_only': True,
            '_xfail_checks': {
                'check_estimators_dtypes': reason,
                'check_fit_score_takes_y': reason,
                'check_estimators_fit_returns_self': reason,
                'check_estimators_fit_returns_self(readonly_memmap=True)':
                reason,
                'check_dtype_object': reason,
                'check_pipeline_consistency': reason,
                'check_estimators_nan_inf': reason,
                'check_estimators_overwrite_params': reason,
                'check_estimators_pickle': reason,
                'check_regressors_train': reason,
                'check_regressors_train(readonly_memmap=True)': reason,
                'check_regressors_train(readonly_memmap=True,X_dtype=float32)':
                reason,
                'check_regressor_data_not_an_array': reason,
                'check_regressor_multioutput': reason,
                'check_regressors_no_decision_function': reason,
                'check_regressors_int': reason,
                'check_methods_sample_order_invariance': reason,
                'check_methods_subset_invariance': reason,
                'check_dict_unchanged': reason,
                'check_dont_overwrite_parameters': reason,
                'check_fit_idempotent': reason,
                'check_n_features_in': reason,
                'check_fit2d_predict1d': reason,
                'check_fit_check_is_fitted': reason,
            }
        }
