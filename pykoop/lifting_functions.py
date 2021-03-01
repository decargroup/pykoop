import sklearn.base
import numpy as np


class Delay(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Sadly, transform() and inverse_transform() are not exact inverses unless
    n_delay_x and n_delay_u are the same. Only the last samples will be the
    same, since some will need to be dropped.
    """

    def __init__(self, n_delay_x=0, n_delay_u=0):
        self.n_delay_x = n_delay_x
        self.n_delay_u = n_delay_u

    def fit(self, X, y=None, n_u=0):
        self._validate_parameters()
        X = self._validate_data(X, reset=True)
        self.n_x_ = X.shape[1] - n_u
        self.n_u_ = n_u
        self.n_xd_ = self.n_x_ * (self.n_delay_x + 1)
        self.n_ud_ = self.n_u_ * (self.n_delay_u + 1)
        self.largest_delay_ = max(self.n_delay_x, self.n_delay_u)
        self.n_samples_needed_ = max(self.n_delay_x, self.n_delay_u) + 1
        return self

    def transform(self, X):
        X = self._validate_data(X, reset=False)
        sklearn.utils.validation.check_is_fitted(self)
        X_x = X[:, :self.n_x_]
        X_u = X[:, self.n_x_:]
        Xd_x = self._delay(X_x, self.n_delay_x)
        Xd_u = self._delay(X_u, self.n_delay_u)
        n_samples = min(Xd_x.shape[0], Xd_u.shape[0])
        Xd = np.hstack((Xd_x[-n_samples:, :], Xd_u[-n_samples:, :]))
        return Xd

    def inverse_transform(self, X):
        X = self._validate_data(X, reset=False)
        sklearn.utils.validation.check_is_fitted(self)
        X_x = X[:, :self.n_xd_]
        X_u = X[:, self.n_xd_:]
        Xu_x = self._undelay(X_x, self.n_delay_x, self.n_x_)
        Xu_u = self._undelay(X_u, self.n_delay_u, self.n_u_)
        n_samples = min(Xu_x.shape[0], Xu_u.shape[0])
        Xu = np.hstack((Xu_x[-n_samples:, :], Xu_u[-n_samples:, :]))
        return Xu

    def _validate_parameters(self):
        if self.n_delay_x < 0:
            raise ValueError('`n_delay_x` must be greater than or equal to '
                             'zero.')
        if self.n_delay_u < 0:
            raise ValueError('`n_delay_u` must be greater than or equal to '
                             'zero.')

    def _validate_data(self, X, y=None, reset=True, **check_array_params):
        X = sklearn.utils.validation.check_array(X, **check_array_params)
        if reset:
            self.n_features_in_ = X.shape[1]
        return X

    def _delay(self, X, n_delay):
        n_samples_out = X.shape[0] - n_delay
        delays = []
        for i in range(n_delay, -1, -1):
            delays.append(X[i:(n_samples_out + i), :])
        Xd = np.concatenate(delays, axis=1)
        return Xd

    def _undelay(self, X, n_delay, n_features):
        if X.shape[1] == 0:
            # TODO This seems like a hack. Maybe revisit.
            n_samples = X.shape[0] + self.largest_delay_
            Xu = np.array([[]] * n_samples)
        else:
            Xu_1 = [X[:-1, -n_features:]]
            Xu_2 = np.split(X[[-1], :], n_delay + 1, axis=1)[::-1]
            Xu = np.vstack(Xu_1 + Xu_2)
        return Xu
