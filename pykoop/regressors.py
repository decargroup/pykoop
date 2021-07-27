"""Collection of regressors for use with or without a Koopman Pipeline.

All of the lifting functions included in this module adhere to the interface
defined in :class:`KoopmanRegressor`.
"""

import numpy as np
import sklearn.base
import sklearn.utils.validation
from scipy import linalg

from . import koopman_pipeline


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

    pass


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
