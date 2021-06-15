"""Lifting functions and preprocessors compatible with the ``KoopmanPipeline``
object.
"""

import sklearn.base
import sklearn.preprocessing
import abc
import numpy as np


class LiftingFunction(sklearn.base.BaseEstimator,
                      sklearn.base.TransformerMixin,
                      metaclass=abc.ABCMeta):
    """Koopman lifting function."""

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = True) -> 'LiftingFunction':
        """Fit lifting function.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.
        n_inputs : int
            Number of input features at the end of ``X``.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.

        Returns
        -------
        LiftingFunction :
            Instance of itself.
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray :
            Transformed data matrix.
        """
        pass

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert transformed data.

        Parameters
        ----------
        X : np.ndarray
            Transformed data matrix.

        Returns
        -------
        np.ndarray :
            Inverted transformed data matrix.
        """
        pass

    @abc.abstractmethod
    def _fit(self,
             X: np.ndarray,
             y: np.ndarray = None,
             n_inputs: int = 0) -> 'LiftingFunction':
        """Internal fit method that operates on a single episode."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Internal transform method that operates on a single episode."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Internal inverse transform method that operates on a single
        episode."""
        raise NotImplementedError()


class AnglePreprocessor(sklearn.base.BaseEstimator,
                        sklearn.base.TransformerMixin):
    """
    Warning, inverse is not true inverse unless data is inside [-pi, pi]
    """

    def __init__(self, unwrap=False):
        self.unwrap = unwrap

    def fit(self, X, y=None, angles=None):  # TODO Make angles list of ints?
        X = self._validate_data(X, reset=True)
        if angles is None:
            self.angles_ = np.zeros((self.n_features_in_, ), dtype=bool)
        else:
            self.angles_ = angles
        # Figure out what the new angle features will be
        self.n_features_out_ = np.sum(~self.angles_) + 2 * np.sum(self.angles_)
        self.lin_ = np.zeros((self.n_features_out_, ), dtype=bool)
        self.cos_ = np.zeros((self.n_features_out_, ), dtype=bool)
        self.sin_ = np.zeros((self.n_features_out_, ), dtype=bool)
        i = 0
        for j in range(self.n_features_in_):
            if self.angles_[j]:
                self.cos_[i] = True
                self.sin_[i + 1] = True
                i += 2
            else:
                self.lin_[i] = True
                i += 1
        return self

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

    def fit(self, X, y=None, n_u=0):
        self._validate_parameters()
        X = self._validate_data(X, reset=True)
        self.transformer_ = sklearn.preprocessing.PolynomialFeatures(
            degree=self.order,
            interaction_only=self.interaction_only,
            include_bias=False)
        self.transformer_.fit(X)
        self.n_features_out_ = self.transformer_.powers_.shape[0]
        self.n_x_ = X.shape[1] - n_u
        self.n_u_ = n_u
        # Figure out which lifted states correspond to the original states and
        # original inputs
        orig_states = []
        orig_inputs = []
        eye = np.eye(self.n_features_in_)
        for i in range(self.n_features_in_):
            index = np.nonzero(
                np.all(self.transformer_.powers_ == eye[i, :], axis=1))
            if i < self.n_x_:
                orig_states.append(index)
            else:
                orig_inputs.append(index)
        original_state_features = np.ravel(orig_states).astype(int)
        original_input_features = np.ravel(orig_inputs).astype(int)
        # Figure out which other lifted states contain inputs
        all_input_features = np.nonzero(
            np.any((self.transformer_.powers_ != 0)[:, self.n_x_:],
                   axis=1))[0].astype(int)
        other_input_features = np.setdiff1d(all_input_features,
                                            original_input_features)
        # Figure out which other lifted states contain states (but are not the
        # original states themselves
        other_state_features = np.setdiff1d(
            np.arange(self.n_features_out_),
            np.union1d(
                np.union1d(original_state_features, original_input_features),
                other_input_features)).astype(int)
        # Form new order
        self.transform_order_ = np.concatenate(
            (original_state_features, other_state_features,
             original_input_features, other_input_features))
        # Figure out original order of features
        self.inverse_transform_order_ = np.concatenate(
            (np.arange(original_state_features.shape[0]),
             np.arange((original_state_features.shape[0] +
                        other_state_features.shape[0]),
                       (original_state_features.shape[0] +
                        other_state_features.shape[0] +
                        original_input_features.shape[0]))))
        # Compute how many input-independent lifted states and input-dependent
        # lifted states there are
        self.n_xl_ = (original_state_features.shape[0] +
                      other_state_features.shape[0])
        self.n_ul_ = (original_input_features.shape[0] +
                      other_state_features.shape[0])
        return self

    def transform(self, X):
        X = self._validate_data(X, reset=False)
        sklearn.utils.validation.check_is_fitted(self)
        Xt = self.transformer_.transform(X)
        return Xt[:, self.transform_order_]

    def inverse_transform(self, X):
        X = self._validate_data(X, reset=False)
        sklearn.utils.validation.check_is_fitted(self)
        # Extract the original features from the lifted features
        return X[:, self.inverse_transform_order_]

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
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Delay `fit()` called wth {self.n_features_in_} '
                             'features, but `transform() called with '
                             f'{X.shape[1]}'
                             'features.')
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
