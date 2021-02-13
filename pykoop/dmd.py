import sklearn.base
import sklearn.utils.validation
from scipy import linalg


class Edmd(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):

    def fit(self, X, y):
        X, y = sklearn.utils.validation.check_X_y(X, y,
                                                  multi_output=True,
                                                  y_numeric=True)
        Psi = X.T
        Theta_p = y.T
        q = Psi.shape[1]
        G = (Theta_p @ Psi.T) / q
        H = (Psi @ Psi.T) / q
        self.U_ = linalg.lstsq(H.T, G.T)[0]
        self.n_features_in_ = Psi.shape[0]
        return self

    def predict(self, X):
        X = sklearn.utils.validation.check_array(X)
        sklearn.utils.validation.check_is_fitted(self)
        Psi = X.T
        Theta_p = self.U_.T @ Psi
        return Theta_p.T

    def _more_tags(self):
        return {
            'multioutput': True,
            'multioutput_only': True,
        }
