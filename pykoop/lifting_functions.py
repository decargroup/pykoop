"""Lifting functions and preprocessors compatible with ``KoopmanPipeline``."""

import abc
import typing

import numpy as np
import sklearn.base
import sklearn.preprocessing
import sklearn.utils.validation


class LiftingFn(sklearn.base.BaseEstimator,
                sklearn.base.TransformerMixin,
                metaclass=abc.ABCMeta):
    """Base class for Koopman lifting functions.

    The format of the data matrices expected by this class must obey the
    parameters set in :func:`fit`.

    Attributes
    ----------
    n_features_in_ : int
        Number of features before transformation, including episode feature.
    n_states_in_ : int
        Number of states before transformation.
    n_inputs_in_ : int
        Number of inputs before transformation.
    n_states_out_ : int
        Number of states after transformation.
    n_inputs_out_ : int
        Number of inputs after transformation.
    """

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = True) -> 'LiftingFn':
        """Fit the lifting function.

        The data matrices provided to :func:`fit` (as well as :func:`transform`
        and :func:`inverse_transform`) must obey the following format:

        1. If ``episode_feature`` is true, the first feature must indicate
           which episode each timestep belongs to.
        2. The last ``n_inputs`` features must be exogenous inputs.
        3. The remaining features are considered to be states.

        Input example where ``episode_feature=True`` and ``n_inputs=1``:

        ======= ======= ======= =======
        Episode State 0 State 1 Input 0
        ======= ======= ======= =======
        0.0       0.1    -0.1    0.2
        0.0       0.2    -0.2    0.3
        0.0       0.3    -0.3    0.4
        1.0      -0.1     0.1    0.3
        1.0      -0.2     0.2    0.4
        1.0      -0.3     0.3    0.5
        2.0       0.3    -0.1    0.3
        2.0       0.2    -0.2    0.4
        ======= ======= ======= =======

        In the above example, there are three distinct episodes with different
        numbers of timesteps. The last feature is an input, so the remaining
        two features must be states.

        If ``n_inputs=0``, the same matrix is interpreted as:

        ======= ======= ======= =======
        Episode State 0 State 1 State 2
        ======= ======= ======= =======
        0.0       0.1    -0.1    0.2
        0.0       0.2    -0.2    0.3
        0.0       0.3    -0.3    0.4
        1.0      -0.1     0.1    0.3
        1.0      -0.2     0.2    0.4
        1.0      -0.3     0.3    0.5
        2.0       0.3    -0.1    0.3
        2.0       0.2    -0.2    0.4
        ======= ======= ======= =======

        If ``episode_feature=False`` and the first feature is omitted, the
        matrix is interpreted as:

        ======= ======= =======
        State 0 State 1 State 2
        ======= ======= =======
         0.1    -0.1    0.2
         0.2    -0.2    0.3
         0.3    -0.3    0.4
        -0.1     0.1    0.3
        -0.2     0.2    0.4
        -0.3     0.3    0.5
         0.3    -0.1    0.3
         0.2    -0.2    0.4
        ======= ======= =======

        In the above case, each timestep is assumed to belong to the same
        episode.

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
        LiftingFn
            Instance of itself.

        Raises
        -----
        ValueError
            If constructor or fit parameters are incorrect.
        """
        raise NotImplementedError()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Transformed data matrix.
        """
        raise NotImplementedError()

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert transformed data.

        Parameters
        ----------
        X : np.ndarray
            Transformed data matrix.

        Returns
        -------
        np.ndarray
            Inverted transformed data matrix.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _fit(self,
             X: np.ndarray,
             y: np.ndarray = None,
             n_inputs: int = 0) -> 'LiftingFn':
        """Fit lifting function using a single episode.

        Expects and returns data without an episode header. Data is assumed to
        belong to a single episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.
        n_inputs : int
            Number of input features at the end of ``X``.

        Returns
        -------
        LiftingFn
            Instance of itself.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using a single episode.

        Expects and returns data without an episode header. Data is assumed to
        belong to a single episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Transformed data matrix.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert transformed data using a single episode.

        Expects and returns data without an episode header. Data is assumed to
        belong to a single episode.

        Parameters
        ----------
        X : np.ndarray
            Transformed data matrix.

        Returns
        -------
        np.ndarray
            Inverted transformed data matrix.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _validate_parameters(self) -> None:
        """Validate parameters passed in constructor.

        Raises
        ------
        ValueError
            If constructor parameters are incorrect.
        """
        raise NotImplementedError()


class EpisodeIndependentLiftingFn(LiftingFn):
    """Base class for Koopman lifting functions that are episode-independent.

    Episode-independent lifting functions can be applied to a complete data
    matrix while ignoring the episode feature.

    For example, when rescaling a data matrix, it does not matter which episode
    a sample comes from.

    The format of the data matrices expected by this class must obey the
    parameters set in :func:`fit`
    """

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = True) -> 'EpisodeIndependentLiftingFn':
        # Validate constructor parameters
        self._validate_parameters()
        # Validate fit parameters
        if n_inputs < 0:
            raise ValueError('`n_inputs` must be a natural number.')
        # Validate data
        X = self._validate_data(X, reset=True)
        # Extract episode feature
        self.episode_feature_ = episode_feature
        if episode_feature:
            X = X[:, 1:]
        # Set states and inputs in
        self.n_inputs_in_ = n_inputs
        self.n_states_in_ = X.shape[1] - n_inputs
        return self._fit(X, n_inputs=n_inputs)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._apply_transform_or_inverse(X, self._transform)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self._apply_transform_or_inverse(X,
                                                self._inverse_transform,
                                                ensure_2d=False)

    def _apply_transform_or_inverse(self, X: np.ndarray,
                                    transform: typing.Callable,
                                    **check_params) -> np.ndarray:
        """Strip episode feature, apply transform or inverse transform, then
        put it back.

        Attributes
        ----------
        X : np.ndarray
            Data matrix.
        transform : typing.Callable
            Transform or inverse transform.
        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array` or
            :func:`sklearn.utils.check_X_y`. In this case, it is used to stop
            :func:`_validate_data` from raising an exception when doing an
            inverse transform.

        Returns
        -------
        np.ndarray
            Transformed or inverse transformed data matrix.
        """
        # Validate data
        X = self._validate_data(X, reset=False, **check_params)
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        # Extract episode feature
        if self.episode_feature_:
            X_ep = X[:, [0]]
            X = X[:, 1:]
        # Transform data
        Xt = transform(X)
        # Put feature back if needed
        if self.episode_feature_:
            Xt = np.hstack((X_ep, Xt))
        return Xt


class EpisodeDependentLiftingFn(LiftingFn):
    """Base class for Koopman lifting functions that are episode-dependent.

    Episode-dependent lifting functions cannot be applied to a complete data
    matrix. The data matrix must be split into episodes, and the lifting
    function must be applied to each one. The resulting lifted episodes are
    then concatenated.

    For example, when applying delay coordinates to a data matrix, samples
    from different episodes must not be intermingled. While a sample from
    episode 0 and a sample from episode 1 are adjacent in the data matrix, they
    did not take place one timestep apart!

    The format of the data matrices expected by this class must obey the
    parameters set in :func:`fit`.
    """

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = True) -> 'EpisodeDependentLiftingFn':
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        pass


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


class PolynomialLiftingFn(EpisodeIndependentLiftingFn):

    def __init__(self, order=1, interaction_only=False):
        self.order = order
        self.interaction_only = interaction_only

    def _fit(self, X, y=None, n_inputs=0):
        self.transformer_ = sklearn.preprocessing.PolynomialFeatures(
            degree=self.order,
            interaction_only=self.interaction_only,
            include_bias=False)
        self.transformer_.fit(X)
        n_features_out_ = self.transformer_.powers_.shape[0]
        # Figure out which lifted states correspond to the original states and
        # original inputs
        orig_states = []
        orig_inputs = []
        eye = np.eye(self.n_states_in_ + self.n_inputs_in_)
        for i in range(self.n_states_in_ + self.n_inputs_in_):
            index = np.nonzero(
                np.all(self.transformer_.powers_ == eye[i, :], axis=1))
            if i < self.n_states_in_:
                orig_states.append(index)
            else:
                orig_inputs.append(index)
        original_state_features = np.ravel(orig_states).astype(int)
        original_input_features = np.ravel(orig_inputs).astype(int)
        # Figure out which other lifted states contain inputs
        all_input_features = np.nonzero(
            np.any((self.transformer_.powers_ != 0)[:, self.n_states_in_:],
                   axis=1))[0].astype(int)
        other_input_features = np.setdiff1d(all_input_features,
                                            original_input_features)
        # Figure out which other lifted states contain states (but are not the
        # original states themselves
        other_state_features = np.setdiff1d(
            np.arange(n_features_out_),
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
        self.n_states_out_ = (original_state_features.shape[0] +
                              other_state_features.shape[0])
        self.n_inputs_out_ = (original_input_features.shape[0] +
                              other_state_features.shape[0])
        return self

    def _transform(self, X):
        Xt = self.transformer_.transform(X)
        return Xt[:, self.transform_order_]

    def _inverse_transform(self, X):
        # Extract the original features from the lifted features
        return X[:, self.inverse_transform_order_]

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
