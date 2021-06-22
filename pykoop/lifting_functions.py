"""Lifting functions and preprocessors compatible with ``KoopmanPipeline``."""

import abc

import numpy as np
import sklearn.base
import sklearn.preprocessing
import sklearn.utils.validation


class LiftingFn(sklearn.base.BaseEstimator,
                sklearn.base.TransformerMixin,
                metaclass=abc.ABCMeta):
    """Base class for Koopman lifting functions.

    Attributes
    ----------
    n_features_in_ : int
        Number of features before transformation, including episode feature if
        present.
    n_states_in_ : int
        Number of states before transformation.
    n_inputs_in_ : int
        Number of inputs before transformation.
    n_features_out_ : int
        Number of features after transformation, including episode feature if
        present.
    n_states_out_ : int
        Number of states after transformation.
    n_inputs_out_ : int
        Number of inputs after transformation.
    min_samples_ : int
        Minimum number of samples needed to use the transformer.
    """

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = False) -> 'LiftingFn':
        """Fit the lifting function.

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
        # Validate constructor parameters
        self._validate_parameters()
        # Validate fit parameters
        if n_inputs < 0:
            raise ValueError('`n_inputs` must be a natural number.')
        # Set up array checks
        # If you have an episode feature, you need at least one other!
        self._check_params = {
            'ensure_min_features': 2 if episode_feature else 1,
        }
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_params)
        # Set numbre of input features (including episode feature)
        self.n_features_in_ = X.shape[1]
        # Extract episode feature
        self.episode_feature_ = episode_feature
        if episode_feature:
            X = X[:, 1:]
        # Set states and inputs in
        self.n_inputs_in_ = n_inputs
        self.n_states_in_ = X.shape[1] - n_inputs
        # Set default value for minimum samples needed. For an
        # episode-independent transformer, the minimum number of samples is
        # always 1.
        self.min_samples_ = 1
        return self._fit(X)

    @abc.abstractmethod
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

    @abc.abstractmethod
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
    def _fit(self, X: np.ndarray, y: np.ndarray = None) -> 'LiftingFn':
        """Fit lifting function using a single episode.

        Expects and returns data without an episode header. Data is assumed to
        belong to a single episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.

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

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_params)
        # Check input shape
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'{self.__class__.__name__} `fit()` called '
                             f'with {self.n_features_in_} features, but '
                             f'`transform()` called with {X.shape[1]} '
                             'features.')
        return self._apply_transform_or_inverse(X, 'transform')

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_params)
        # Check input shape
        if X.shape[1] != self.n_features_out_:
            raise ValueError(f'{self.__class__.__name__} `fit()` output '
                             f'{self.n_features_out_} features, but '
                             '`inverse_transform()` called with '
                             f'{X.shape[1]} features.')
        return self._apply_transform_or_inverse(X, 'inverse_transform')

    def _apply_transform_or_inverse(self, X: np.ndarray,
                                    transform: str) -> np.ndarray:
        """Strip episode feature, apply transform or inverse, then put it back.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        transform : str
            ``transform`` to apply transform or ``inverse_transform`` to apply
            inverse transform.

        Returns
        -------
        np.ndarray
            Transformed or inverse transformed data matrix.
        """
        # Extract episode feature
        if self.episode_feature_:
            X_ep = X[:, [0]]
            X = X[:, 1:]
        # Transform or inverse transform data
        if transform == 'transform':
            Xt = self._transform(X)
        elif transform == 'inverse_transform':
            Xt = self._inverse_transform(X)
        else:
            raise ValueError("Parameter `transform` must be one of "
                             "['transform', 'inverse_transform']")
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

    For example, when applying delay coordinates to a data matrix, samples from
    different episodes must not be intermingled. While a sample from episode 0
    and a sample from episode 1 are adjacent in the data matrix, they did not
    take place one timestep apart!

    However, :func:`fit` must not depend on the episode feature. Only
    :func:`transform` and :func:`inverse_transform` really need to work on an
    episode-by-episode basis.

    The format of the data matrices expected by this class must obey the
    parameters set in :func:`fit`.
    """

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_params)
        # Check input shape
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'{self.__class__.__name__} `fit()` called '
                             f'with {self.n_features_in_} features, but '
                             f'`transform()` called with {X.shape[1]} '
                             'features.')
        return self._apply_transform_or_inverse(X, 'transform')

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_params)
        # Check input shape
        if X.shape[1] != self.n_features_out_:
            raise ValueError(f'{self.__class__.__name__} `fit()` output '
                             f'{self.n_features_out_} features, but '
                             '`inverse_transform()` called with '
                             f'{X.shape[1]} features.')
        return self._apply_transform_or_inverse(X, 'inverse_transform')

    def _apply_transform_or_inverse(self, X: np.ndarray,
                                    transform: str) -> np.ndarray:
        """Split up episodes, apply transform or inverse transform on each
        episode individually, then combine them again.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        transform : str
            ``transform`` to apply transform or ``inverse_transform`` to apply
            inverse transform.

        Returns
        -------
        np.ndarray
            Transformed or inverse transformed data matrix.
        """
        # Extract episode feature
        if self.episode_feature_:
            X_ep = X[:, 0]
            X = X[:, 1:]
        else:
            X_ep = np.zeros((X.shape[0], ))
        # Split X into list of episodes. Each episode is a tuple containing
        # its index and its associated data matrix.
        episodes = []
        for i in np.unique(X_ep):
            episodes.append((i, X[X_ep == i, :]))
        # Transform episodes one-by-one
        transformed_episodes = []
        for (i, X_i) in episodes:
            # Apply transform or inverse to individual episodes
            if transform == 'transform':
                transformed_episode = self._transform(X_i)
            elif transform == 'inverse_transform':
                transformed_episode = self._inverse_transform(X_i)
            else:
                raise ValueError("Parameter `transform` must be one of "
                                 "['transform', 'inverse_transform']")
            # Add new episode feature back if needed. This is necessary because
            # some transformations may modify the episode length.
            if self.episode_feature_:
                transformed_episodes.append(
                    np.hstack((
                        i * np.ones((transformed_episode.shape[0], 1)),
                        transformed_episode,
                    )))
            else:
                transformed_episodes.append(transformed_episode)
        # Concatenate the transformed episodes
        Xt = np.vstack(transformed_episodes)
        return Xt


class AnglePreprocessor(EpisodeIndependentLiftingFn):
    """Intended for preprocessing, not as a lifting function.
    Warning, inverse is not true inverse unless data is inside [-pi, pi]
    """

    def __init__(self, angles=None, unwrap_inverse=False):
        self.angles = angles
        self.unwrap_inverse = unwrap_inverse

    def _fit(self, X, y=None):
        n_states_inputs_in = self.n_states_in_ + self.n_inputs_in_
        if self.angles is None:
            self.angles_ = np.zeros((n_states_inputs_in, ), dtype=bool)
        elif self.angles.dtype == 'bool':
            self.angles_ = self.angles
        else:
            # Create an array with all ```False``
            angles_bool = np.zeros((n_states_inputs_in, ), dtype=bool)
            # Set indicated entries to ``True``
            angles_bool[self.angles] = True
            self.angles_ = angles_bool
        # Figure out what the new angle features will be
        n_lin_states = np.sum(~self.angles_[:self.n_states_in_])
        n_lin_inputs = np.sum(~self.angles_[self.n_states_in_:])
        n_ang_states = 2 * np.sum(self.angles_[:self.n_states_in_])
        n_ang_inputs = 2 * np.sum(self.angles_[self.n_states_in_:])
        self.n_states_out_ = n_lin_states + n_ang_states
        self.n_inputs_out_ = n_lin_inputs + n_ang_inputs
        n_states_inputs_out = self.n_states_out_ + self.n_inputs_out_
        self.n_features_out_ = (n_states_inputs_out +
                                (1 if self.episode_feature_ else 0))
        self.lin_ = np.zeros((n_states_inputs_out, ), dtype=bool)
        self.cos_ = np.zeros((n_states_inputs_out, ), dtype=bool)
        self.sin_ = np.zeros((n_states_inputs_out, ), dtype=bool)
        i = 0
        for j in range(n_states_inputs_in):
            if self.angles_[j]:
                self.cos_[i] = True
                self.sin_[i + 1] = True
                i += 2
            else:
                self.lin_[i] = True
                i += 1
        return self

    def _transform(self, X):
        n_states_inputs_out = self.n_states_out_ + self.n_inputs_out_
        Xt = np.zeros((X.shape[0], n_states_inputs_out))
        Xt[:, self.lin_] = X[:, ~self.angles_]
        Xt[:, self.cos_] = np.cos(X[:, self.angles_])
        Xt[:, self.sin_] = np.sin(X[:, self.angles_])
        return Xt

    def _inverse_transform(self, X):
        n_states_inputs_in = self.n_states_in_ + self.n_inputs_in_
        Xt = np.zeros((X.shape[0], n_states_inputs_in))
        Xt[:, ~self.angles_] = X[:, self.lin_]
        angles = np.arctan2(X[:, self.sin_], X[:, self.cos_])
        if self.unwrap_inverse:
            Xt[:, self.angles_] = np.unwrap(angles, axis=0)
        else:
            Xt[:, self.angles_] = angles
        return Xt

    def _validate_parameters(self):
        pass


class PolynomialLiftingFn(EpisodeIndependentLiftingFn):

    def __init__(self, order=1, interaction_only=False):
        self.order = order
        self.interaction_only = interaction_only

    def _fit(self, X, y=None):
        self.transformer_ = sklearn.preprocessing.PolynomialFeatures(
            degree=self.order,
            interaction_only=self.interaction_only,
            include_bias=False)
        self.transformer_.fit(X)
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
            np.arange(self.transformer_.powers_.shape[0]),
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
             np.arange((original_state_features.shape[0]
                        + other_state_features.shape[0]),
                       (original_state_features.shape[0]
                        + other_state_features.shape[0]
                        + original_input_features.shape[0]))))
        # Compute how many input-independent lifted states and input-dependent
        # lifted states there are
        self.n_states_out_ = (original_state_features.shape[0]
                              + other_state_features.shape[0])
        self.n_inputs_out_ = (original_input_features.shape[0]
                              + other_input_features.shape[0])
        self.n_features_out_ = (self.n_states_out_ + self.n_inputs_out_ +
                                (1 if self.episode_feature_ else 0))
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


class DelayLiftingFn(EpisodeDependentLiftingFn):
    """
    Sadly, transform() and inverse_transform() are not exact inverses unless
    n_delays_x and n_delays_u are the same. Only the last samples will be the
    same, since some will need to be dropped.
    """

    def __init__(self, n_delays_x=0, n_delays_u=0):
        self.n_delays_x = n_delays_x
        self.n_delays_u = n_delays_u

    def _fit(self, X, y=None):
        self.n_states_out_ = self.n_states_in_ * (self.n_delays_x + 1)
        self.n_inputs_out_ = self.n_inputs_in_ * (self.n_delays_u + 1)
        self.n_features_out_ = (self.n_states_out_ + self.n_inputs_out_ +
                                (1 if self.episode_feature_ else 0))
        self.n_samples_needed_ = max(self.n_delays_x, self.n_delays_u) + 1
        return self

    def _transform(self, X):
        X_x = X[:, :self.n_states_in_]
        X_u = X[:, self.n_states_in_:]
        Xd_x = self._delay(X_x, self.n_delays_x)
        Xd_u = self._delay(X_u, self.n_delays_u)
        n_samples = min(Xd_x.shape[0], Xd_u.shape[0])
        Xd = np.hstack((Xd_x[-n_samples:, :], Xd_u[-n_samples:, :]))
        return Xd

    def _inverse_transform(self, X):
        X_x = X[:, :self.n_states_out_]
        X_u = X[:, self.n_states_out_:]
        Xu_x = self._undelay(X_x, self.n_delays_x, self.n_states_in_)
        Xu_u = self._undelay(X_u, self.n_delays_u, self.n_inputs_in_)
        n_samples = min(Xu_x.shape[0], Xu_u.shape[0])
        Xu = np.hstack((Xu_x[-n_samples:, :], Xu_u[-n_samples:, :]))
        return Xu

    def _validate_parameters(self):
        if self.n_delays_x < 0:
            raise ValueError('`n_delays_x` must be greater than or equal to '
                             'zero.')
        if self.n_delays_u < 0:
            raise ValueError('`n_delays_u` must be greater than or equal to '
                             'zero.')

    def _delay(self, X, n_delay):
        n_samples_out = X.shape[0] - n_delay
        delays = []
        for i in range(n_delay, -1, -1):
            delays.append(X[i:(n_samples_out + i), :])
        Xd = np.concatenate(delays, axis=1)
        return Xd

    def _undelay(self, X, n_delay, n_features):
        Xu_1 = [X[:-1, -n_features:]]
        Xu_2 = np.split(X[[-1], :], n_delay + 1, axis=1)[::-1]
        Xu = np.vstack(Xu_1 + Xu_2)
        return Xu
