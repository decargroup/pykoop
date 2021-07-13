import numpy as np
import sklearn.base
import sklearn.utils.validation
from scipy import linalg

from . import koopman_pipeline


class Edmd(koopman_pipeline.KoopmanRegressor):

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        Psi = X_unshifted.T
        Theta_p = X_shifted.T
        q = Psi.shape[1]
        G = (Theta_p @ Psi.T) / q
        H = (Psi @ Psi.T) / q
        coef = linalg.lstsq(H.T, G.T)[0]
        return coef

    def _validate_parameters(self) -> None:
        pass
