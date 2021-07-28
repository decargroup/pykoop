"""Collection of regressors for use with or without a Koopman Pipeline.

All of the lifting functions included in this module adhere to the interface
defined in :class:`KoopmanRegressor`.
"""

from typing import Any, Union

import numpy as np
import sklearn.base
import sklearn.utils.validation
from scipy import linalg

from . import _tsvd, koopman_pipeline


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
    coef_ : np.ndarray
        Fit coefficient matrix.
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
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    coef_ : np.ndarray
        Fit coefficient matrix.
    """

    def __init__(self,
                 mode_type: str = 'projected',
                 tsvd_method: Union[str, tuple] = 'economy') -> None:
        """Instantiate :class:`Dmdc`.

        Parameters
        ----------
        mode_type : str
            DMD mode type, either ``'exact'`` or ``'projected'``.

        tsvd_method : Union[str, tuple]
            Singular value truncation method used to change basis. Parameters
            for ``X_unshifted`` and ``X_shifted`` are specified independently.
            Possible values are

            - ``'economy'`` or ``('economy', )`` -- do not truncate (use
              economy SVD),
            - ``'unknown_noise'`` or ``('unknown_noise', )`` -- truncate using
              optimal hard truncation [optht]_ with unknown noise,
            - ``('known_noise', sigma_unshifted, sigma_shifted)`` -- truncate
              using optimal hard truncation [optht]_ with known noise,
            - ``('cutoff', cutoff_unshifted, cutoff_shifted)`` -- truncate
              singular values smaller than a cutoff, or
            - ``('rank', rank_unshifted, rank_shifted)`` -- truncate singular
              values to a fixed rank.
        """
        self.mode_type = mode_type
        self.tsvd_method = tsvd_method

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        Psi = X_unshifted.T
        Theta_p = X_shifted.T
        # Compute truncated SVDs
        tsvd_method_tld, tsvd_method_hat = Dmdc._get_tsvd_methods(
            self.tsvd_method)
        Q_tld, sig_tld, Z_tld = _tsvd._tsvd(Psi, *tsvd_method_tld)
        Q_hat, sig_hat, Z_hat = _tsvd._tsvd(Theta_p, *tsvd_method_hat)
        Sig_tld_inv = np.diag(1 / sig_tld)
        # Compute ``A`` and ``B``
        Q_tld_1 = Q_tld[:Theta_p.shape[0], :]
        Q_tld_2 = Q_tld[Theta_p.shape[0]:, :]
        A = Theta_p @ Z_tld @ Sig_tld_inv @ Q_tld_1.T
        B = Theta_p @ Z_tld @ Sig_tld_inv @ Q_tld_2.T
        # Compute ``A_tilde`` and ``B_tilde``
        A_tld = Q_hat.T @ Theta_p @ Z_tld @ Sig_tld_inv @ Q_tld_1.T @ Q_hat
        B_tld = Q_hat.T @ Theta_p @ Z_tld @ Sig_tld_inv @ Q_tld_2.T
        self.B_tld_ = B_tld
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

    @staticmethod
    def _get_tsvd_methods(
            tsvd_method: Union[str, tuple]) -> tuple[tuple, tuple]:
        """Format truncated SVD methods for ``_tsvd``."""
        # Convert string if needed
        if type(tsvd_method) is not tuple:
            tsvd_method = (tsvd_method, )
        # Form tuples
        if len(tsvd_method) == 1:
            tms = (tsvd_method, tsvd_method)
        else:
            try:
                tm_tld = (tsvd_method[0], tsvd_method[1])
                tm_hat = (tsvd_method[0], tsvd_method[2])
            except IndexError:
                raise ValueError('Incorrect number of tuple items in '
                                 '`tsvd_method`.')
            tms = (tm_tld, tm_hat)
        return tms


class Dmd(koopman_pipeline.KoopmanRegressor):
    """Dynamic Mode Decomposition.

    Attributes
    ----------
    eigenvalues_ : np.ndarray
        DMD eigenvalues.
    modes_ : np.ndarray
        DMD modes (exact or projected depending on constructor params).
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    coef_ : np.ndarray
        Fit coefficient matrix.
    """

    # Override check parameters to skip ``check_fit2d_1sample`` sklearn test.
    _check_X_y_params: dict[str, Any] = {
        'multi_output': True,
        'y_numeric': True,
        'ensure_min_samples': 2,
    }

    def __init__(self,
                 mode_type: str = 'projected',
                 tsvd_method: Union[str, tuple] = 'economy') -> None:
        """Instantiate :class:`Dmd`.

        Parameters
        ----------
        mode_type : str
            DMD mode type, either ``'exact'`` or ``'projected'``.

        tsvd_method : Union[str, tuple]
            Singular value truncation method used to change basis. Possible
            values are

            - ``'economy'`` or ``('economy', )`` -- do not truncate (use
              economy SVD),
            - ``'unknown_noise'`` or ``('unknown_noise', )`` -- truncate using
              optimal hard truncation [optht]_ with unknown noise,
            - ``('known_noise', sigma)`` -- truncate using optimal hard
              truncation [optht]_ with known noise,
            - ``('cutoff', cutoff)`` -- truncate singular values smaller than a
              cutoff, or
            - ``('rank', rank)`` -- truncate singular values to a fixed rank.

        Warning
        -------
        :class:`Dmd` has some compatibility issues with ``scikit-learn``
        because of its relatively strict input requirements. Since both inputs
        must have the same number of features, many incompatible
        ``scikit-learn`` unit tests are disabled.
        """
        self.mode_type = mode_type
        self.tsvd_method = tsvd_method

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        Psi = X_unshifted.T
        Psi_p = X_shifted.T
        # Compute truncated SVD
        if type(self.tsvd_method) is tuple:
            tsvd_method = self.tsvd_method
        else:
            tsvd_method = (self.tsvd_method, )
        Q, sigma, Z = _tsvd._tsvd(Psi, *tsvd_method)
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
            }
        }
