import sklearn.base
import sklearn.preprocessing
import numpy as np


class AnglePreprocessor(sklearn.base.BaseEstimator,
                        sklearn.base.TransformerMixin):
    """
    Warning, inverse is not true inverse unless data is inside [-pi, pi]
    """

    def __init__(self, unwrap=False):
        self.unwrap = unwrap

    def fit(self, X, y=None, angles=None):
        X = self._validate_data(X, reset=True)
        if angles is None:
            self.angles_ = np.zeros((self.n_features_in_,), dtype=bool)
        else:
            self.angles_ = angles
        # Figure out what the new angle features will be
        self.n_features_out_ = np.sum(~self.angles_) + 2 * np.sum(self.angles_)
        self.lin_ = np.zeros((self.n_features_out_,), dtype=bool)
        self.cos_ = np.zeros((self.n_features_out_,), dtype=bool)
        self.sin_ = np.zeros((self.n_features_out_,), dtype=bool)
        i = 0
        for j in range(self.n_features_in_):
            if self.angles_[j]:
                self.cos_[i] = True
                self.sin_[i + 1] = True
                i += 2
            else:
                self.lin_[i] = True
                i += 1

    def transform(self, X):
        X = self._validate_data(X, reset=False)
        sklearn.utils.validation.check_is_fitted(self)
        Xt = np.zeros((X.shape[0], self.n_features_out_))
        Xt[:, self.lin_] = X[:, ~self.angles_]
        Xt[:, self.cos_] = np.cos(X[:, self.angles_])
        Xt[:, self.sin_] = np.sin(X[:, self.angles_])
        return Xt

    def inverse_transform(self, X):
        X = self._validate_data(X, reset=False)
        sklearn.utils.validation.check_is_fitted(self)
        Xt = np.zeros((X.shape[0], self.n_features_in_))
        Xt[:, ~self.angles_] = X[:, self.lin_]
        angles = np.arctan2(X[:, self.sin_], X[:, self.cos_])
        if self.unwrap:
            Xt[:, self.angles_] = np.unwrap(angles, axis=0)
        else:
            Xt[:, self.angles_] = angles
        return Xt

    def _validate_data(self, X, y=None, reset=True, **check_array_params):
        X = sklearn.utils.validation.check_array(X, **check_array_params)
        if reset:
            self.n_features_in_ = X.shape[1]
        return X


class PolynomialLiftingFn(sklearn.base.BaseEstimator,
                          sklearn.base.TransformerMixin):

    def __init__(self, order=1, interaction_only=False):
        self.order = order
        self.interaction_only = interaction_only

    def fit(self, X, y=None, inputs=None):
        self._validate_parameters()
        X = self._validate_data(X, reset=True)
        if inputs is None:
            inputs = np.zeros((self.n_features_in_,), dtype=bool)
        self.transformer_ = sklearn.preprocessing.PolynomialFeatures(
            degree=self.order,
            interaction_only=self.interaction_only,
            include_bias=False
        )
        self.transformer_.fit(X)
        self.n_features_out_ = self.transformer_.powers_.shape[0]
        # Figure out which lifted states correspond to the original states and
        # original inputs
        orig_states = []
        orig_inputs = []
        eye = np.eye(self.n_features_in_)
        for i in range(self.n_features_in_):
            index = np.nonzero(np.all(self.transformer_.powers_ == eye[i, :],
                                      axis=1))
            if not inputs[i]:
                orig_states.append(index)
            else:
                orig_inputs.append(index)
        self.original_state_features_ = np.ravel(orig_states).astype(int)
        self.original_input_features_ = np.ravel(orig_inputs).astype(int)
        # Figure out which other lifted states contain inputs
        other_inputs = np.array([
            inputs[i] if i not in self.original_input_features_ else False
            for i in range(inputs.shape[0])
        ])
        self.other_input_features_ = np.nonzero(np.any(
            (self.transformer_.powers_ != 0)[:, other_inputs],
            axis=1
        ))[0].astype(int)
        # Figure out which other lifted states contain states (but are not the
        # original states themselves
        self.other_state_features_ = np.array([
            i for i in range(self.n_features_out_)
            if ((i not in self.original_state_features_)
                and (i not in self.original_input_features_)
                and (i not in self.other_input_features_))
        ]).astype(int)
        return self

    def transform(self, X):
        X = self._validate_data(X, reset=False)
        sklearn.utils.validation.check_is_fitted(self)
        Xt = self.transformer_.transform(X)
        Xt_reordered = [Xt[:, self.original_state_features_]]
        if self.other_state_features_.shape[0] != 0:
            Xt_reordered.append(Xt[:, self.other_state_features_])
        if self.original_input_features_.shape[0] != 0:
            Xt_reordered.append(Xt[:, self.original_input_features_])
        if self.other_input_features_.shape[0] != 0:
            Xt_reordered.append(Xt[:, self.other_input_features_])
        return np.hstack(Xt_reordered)

    def inverse_transform(self, X):
        X = self._validate_data(X, reset=False)
        sklearn.utils.validation.check_is_fitted(self)
        original_features = np.concatenate((self.original_state_features_,
                                            self.original_input_features_))
        breakpoint()
        return X[:, original_features]

    def _validate_data(self, X, y=None, reset=True, **check_array_params):
        X = sklearn.utils.validation.check_array(X, **check_array_params)
        if reset:
            self.n_features_in_ = X.shape[1]
        return X

    def _validate_parameters(self):
        if self.order <= 0:
            raise ValueError('`order` must be greater than or equal to 1.')


class Delay(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Sadly, transform() and inverse_transform() are not exact inverses unless
    n_delay_x and n_delay_u are the same. Only the last samples will be the
    same, since some will need to be dropped.
    """

    def __init__(self, n_delay_x=0, n_delay_u=0):
        self.n_delay_x = n_delay_x
        self.n_delay_u = n_delay_u

    def fit(self, X, y=None, n_u=-1):  # TODO THIS IS WRONG, RIGHT???
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
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Delay `fit()` called wth {self.n_features_in_} '
                             'features, but `transform() called with '
                             f'{X.shape[1]}' 'features.')
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
