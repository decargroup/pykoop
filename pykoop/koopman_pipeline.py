"""Koopman pipeline meta-estimators and related interfaces."""

import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas
import sklearn.base
import sklearn.metrics
from deprecated import deprecated

from ._sklearn_metaestimators import metaestimators

# Create logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class _LiftRetractMixin(metaclass=abc.ABCMeta):
    """Mixin providing more convenient lift/retract functions.

    Assumes child class implements :func:`transform` and
    :func:`inverse_transform`. See :class:`KoopmanLiftingFn` and
    :class:`KoopmanPipeline`` for details concerning the class methods and
    attributes.

    All attributes with a trailing underscore must be set in the subclass'
    :func:`fit`.

    Attributes
    ----------
    n_states_in_ : int
        Number of states before transformation.
    n_inputs_in_ : int
        Number of inputs before transformation.
    n_states_out_ : int
        Number of states after transformation.
    n_inputs_out_ : int
        Number of inputs after transformation.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    """

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

    def lift(self, X: np.ndarray, episode_feature: bool = None) -> np.ndarray:
        """Lift state and input.

        Potentially more convenient alternative to calling :func:`transform`.

        Parameters
        ----------
        X : np.ndarray
            State and input.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            Lifted state and input.
        """
        if ((episode_feature == self.episode_feature_)
                or (episode_feature is None)):
            # Can use ``transform`` without modification because
            # ``episode_feature`` is either ``None``, or it's the same as
            # ``self.episode_feature_``.
            Xt = self.transform(X)
        else:
            # ``episode_feature`` and ``self.episode_feature_`` differ, so the
            # input and output need to be padded and/or stripped.
            if self.episode_feature_:
                # Estimator was fit with an episode feature, but the input does
                # not have one. Need to add a fake one and them remove it after
                # transforming.
                X_ep = np.hstack((np.zeros((X.shape[0], 1)), X))
                Xt_ep = self.transform(X_ep)
                Xt = Xt_ep[:, 1:]
            else:
                # Estimator was fit without an episode feature, but the input
                # has one. Need to break up and recombine episodes.
                eps = split_episodes(X, episode_feature=episode_feature)
                eps_t = []
                for (i, X_i) in eps:
                    Xt_i = self.transform(X_i)
                    eps_t.append((i, Xt_i))
                Xt = combine_episodes(eps_t, episode_feature=episode_feature)
        return Xt

    def retract(
        self,
        X: np.ndarray,
        episode_feature: bool = None,
    ) -> np.ndarray:
        """Retract lifted state and input.

        Potentially more convenient alternative to calling
        :func:`inverse_transform`.

        Parameters
        ----------
        X : np.ndarray
            Lifted state and input.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            State and input.
        """
        if ((episode_feature == self.episode_feature_)
                or (episode_feature is None)):
            # Can use ``inverse_transform`` without modification because
            # ``episode_feature`` is either ``None``, or it's the same as
            # ``self.episode_feature_``.
            Xt = self.inverse_transform(X)
        else:
            # ``episode_feature`` and ``self.episode_feature_`` differ, so the
            # input and output need to be padded and/or stripped.
            if self.episode_feature_:
                # Estimator was fit with an episode feature, but the input does
                # not have one. Need to add a fake one and them remove it after
                # inverse-transforming.
                X_ep = np.hstack((np.zeros((X.shape[0], 1)), X))
                Xt_ep = self.inverse_transform(X_ep)
                Xt = Xt_ep[:, 1:]
            else:
                # Estimator was fit without an episode feature, but the input
                # has one. Need to break up and recombine episodes.
                eps = split_episodes(X, episode_feature=episode_feature)
                eps_t = []
                for (i, X_i) in eps:
                    Xt_i = self.inverse_transform(X_i)
                    eps_t.append((i, Xt_i))
                Xt = combine_episodes(eps_t, episode_feature=episode_feature)
        return Xt

    def lift_state(
        self,
        X: np.ndarray,
        episode_feature: bool = None,
    ) -> np.ndarray:
        """Lift state only.

        More convenient alternative to padding the state with dummy inputs,
        calling :func:`transform`, then stripping the unwanted lifted inputs.

        Parameters
        ----------
        X : np.ndarray
            State.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            Lifted state.
        """
        # Pad fake inputs
        X_pad = np.hstack((X, np.zeros((X.shape[0], self.n_inputs_in_))))
        # Lift states with fake inputs
        Xt_pad = self.lift(X_pad, episode_feature=episode_feature)
        # Strip fake lifted inputs
        Xt = Xt_pad[:, :self.n_states_out_ + (1 if episode_feature else 0)]
        return Xt

    def retract_state(
        self,
        X: np.ndarray,
        episode_feature: bool = None,
    ) -> np.ndarray:
        """Retract lifted state only.

        More convenient alternative to padding the lifted state with dummy
        lifted inputs, calling :func:`inverse_transform`.

        Parameters
        ----------
        X : np.ndarray
            Lifted state.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            State.
        """
        # Pad fake lifted inputs
        X_pad = np.hstack((X, np.zeros((X.shape[0], self.n_inputs_out_))))
        # Retract states with fake inputs
        Xt_pad = self.retract(X_pad, episode_feature=episode_feature)
        # Strip fake inputs
        Xt = Xt_pad[:, :self.n_states_in_ + (1 if episode_feature else 0)]
        return Xt

    def lift_input(
        self,
        X: np.ndarray,
        episode_feature: bool = None,
    ) -> np.ndarray:
        """Lift input only.

        More convenient alternative to calling :func:`transform`, then
        stripping the unwanted lifted states.

        Parameters
        ----------
        X : np.ndarray
            State and input.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            Lifted input.
        """
        # Lift states and inputs
        Xt_pad = self.lift(X, episode_feature=episode_feature)
        # Strip lifted states while retaining episode feature as needed
        if episode_feature:
            Xt = np.hstack((
                Xt_pad[:, [0]],
                Xt_pad[:, self.n_states_out_ + 1:],
            ))
        else:
            Xt = Xt_pad[:, self.n_states_out_:]
        return Xt

    def retract_input(
        self,
        X: np.ndarray,
        episode_feature: bool = None,
    ) -> np.ndarray:
        """Retract lifted input only.

        More convenient alternative to padding the lifted state with dummy
        lifted states, calling :func:`inverse_transform`, then stripping the
        unwanted states.

        Parameters
        ----------
        X : np.ndarray
            Lifted input.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            Input.
        """
        # Pad fake lifted states
        if episode_feature:
            X_pad = np.hstack((
                X[:, [0]],
                np.zeros((X.shape[0], self.n_states_out_)),
                X[:, 1:],
            ))
        else:
            X_pad = np.hstack((np.zeros((X.shape[0], self.n_states_out_)), X))
        # Retract inputs with fake states
        Xt_pad = self.retract(X_pad, episode_feature=episode_feature)
        # Strip fake states
        if episode_feature:
            Xt = np.hstack((Xt_pad[:, [0]], Xt_pad[:, self.n_states_in_ + 1:]))
        else:
            Xt = Xt_pad[:, self.n_states_in_:]
        return Xt


class KoopmanLiftingFn(sklearn.base.BaseEstimator,
                       sklearn.base.TransformerMixin,
                       _LiftRetractMixin,
                       metaclass=abc.ABCMeta):
    """Base class for Koopman lifting functions.

    All attributes with a trailing underscore must be set in the subclass'
    :func:`fit`.

    Attributes
    ----------
    n_features_in_ : int
        Number of features before transformation, including episode feature.
    n_states_in_ : int
        Number of states before transformation.
    n_inputs_in_ : int
        Number of inputs before transformation.
    n_features_out_ : int
        Number of features after transformation, including episode feature.
    n_states_out_ : int
        Number of states after transformation.
    n_inputs_out_ : int
        Number of inputs after transformation.
    min_samples_ : int
        Minimum number of samples needed to use the transformer.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    """

    @abc.abstractmethod
    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = False) -> 'KoopmanLiftingFn':
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
        KoopmanLiftingFn
            Instance of itself.

        Raises
        -----
        ValueError
            If constructor or fit parameters are incorrect.
        """
        raise NotImplementedError()

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
    def n_samples_in(self, n_samples_out: int = 1) -> int:
        """Calculate number of input samples required for given output length.

        Parameters
        ----------
        n_samples_out : int
            Number of samples needed at the output.

        Returns
        -------
        int
            Number of samples needed at the input.
        """
        raise NotImplementedError()


class EpisodeIndependentLiftingFn(KoopmanLiftingFn):
    """Base class for Koopman lifting functions that are episode-independent.

    Episode-independent lifting functions can be applied to a complete data
    matrix while ignoring the episode feature.

    For example, when rescaling a data matrix, it does not matter which episode
    a sample comes from.

    Attributes
    ----------
    n_features_in_ : int
        Number of features before transformation, including episode feature.
        Set by :func:`fit`.
    n_states_in_ : int
        Number of states before transformation.
        Set by :func:`fit`.
    n_inputs_in_ : int
        Number of inputs before transformation.
        Set by :func:`fit`.
    n_features_out_ : int
        Number of features after transformation, including episode feature.
        Set by :func:`fit`.
    n_states_out_ : int
        Number of states after transformation.
        Set by :func:`fit`.
    n_inputs_out_ : int
        Number of inputs after transformation.
        Set by :func:`fit`.
    min_samples_ : int
        Minimum number of samples needed to use the transformer.
        Set by :func:`fit`.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
        Set by :func:`fit`.
    """

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = False) -> 'EpisodeIndependentLiftingFn':
        # noqa: D102
        # Validate constructor parameters
        self._validate_parameters()
        # Validate fit parameters
        if n_inputs < 0:
            raise ValueError('`n_inputs` must be greater than or equal to 0.')
        # Save presence of episode feature
        self.episode_feature_ = episode_feature
        # Set up array checks. If you have an episode feature, you need at
        # least one other feature!
        self._check_array_params = {
            'ensure_min_features': 2 if episode_feature else 1,
        }
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Set numbre of input features (including episode feature)
        self.n_features_in_ = X.shape[1]
        # Extract episode feature
        if self.episode_feature_:
            X = X[:, 1:]
        # Set states and inputs in
        self.n_inputs_in_ = n_inputs
        self.n_states_in_ = X.shape[1] - n_inputs
        # Episode independent lifting functions only need one sample.
        n_x, n_u = self._fit_one_ep(X)
        self.n_states_out_ = n_x
        self.n_inputs_out_ = n_u
        self.n_features_out_ = (self.n_states_out_ + self.n_inputs_out_ +
                                (1 if self.episode_feature_ else 0))
        # Episode-independent lifting functions only ever need one sample.
        self.min_samples_ = 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # noqa: D102
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'{self.__class__.__name__} `fit()` called '
                             f'with {self.n_features_in_} features, but '
                             f'`transform()` called with {X.shape[1]} '
                             'features.')
        return self._apply_transform_or_inverse(X, 'transform')

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        # noqa: D102
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_out_:
            raise ValueError(f'{self.__class__.__name__} `fit()` output '
                             f'{self.n_features_out_} features, but '
                             '`inverse_transform()` called with '
                             f'{X.shape[1]} features.')
        return self._apply_transform_or_inverse(X, 'inverse_transform')

    def n_samples_in(self, n_samples_out: int = 1) -> int:
        # noqa: D102
        # Episode-independent lifting functions have an input for every output.
        return n_samples_out

    def _apply_transform_or_inverse(self, X: np.ndarray,
                                    transform: str) -> np.ndarray:
        """Strip episode feature, apply transform or inverse, then put it back.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        transform : str
            ``'transform'`` to apply transform or ``'inverse_transform'`` to
            apply inverse transform.

        Returns
        -------
        np.ndarray
            Transformed or inverse transformed data matrix.

        Raises
        ------
        ValueError
            If ``transform`` is not  ``'transform'`` or
            ``'inverse_transform'``.
        """
        # Extract episode feature
        if self.episode_feature_:
            X_ep = X[:, [0]]
            X = X[:, 1:]
        # Transform or inverse transform data
        if transform == 'transform':
            Xt = self._transform_one_ep(X)
        elif transform == 'inverse_transform':
            Xt = self._inverse_transform_one_ep(X)
        else:
            raise ValueError("Parameter `transform` must be one of "
                             "['transform', 'inverse_transform']")
        # Put feature back if needed
        if self.episode_feature_:
            Xt = np.hstack((X_ep, Xt))
        return Xt

    @abc.abstractmethod
    def _fit_one_ep(self, X: np.ndarray) -> Tuple[int, int]:
        """Fit lifting function using a single episode.

        Expects and returns data without an episode header. Data is assumed to
        belong to a single episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        Tuple[int, int]
            Tuple containing the number of state features and input features in
            the transformed data.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
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
    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
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


class EpisodeDependentLiftingFn(KoopmanLiftingFn):
    """Base class for Koopman lifting functions that are episode-dependent.

    Episode-dependent lifting functions cannot be applied to a complete data
    matrix. The data matrix must be split into episodes, and the lifting
    function must be applied to each one. The resulting lifted episodes are
    then concatenated.

    For example, when applying delay coordinates to a data matrix, samples from
    different episodes must not be intermingled. While a sample from episode 0
    and a sample from episode 1 are adjacent in the data matrix, they did not
    take place one timestep apart! Episode-dependent lifting functions take
    this requirement into account.

    Attributes
    ----------
    n_features_in_ : int
        Number of features before transformation, including episode feature.
        Set by :func:`fit`.
    n_states_in_ : int
        Number of states before transformation.
        Set by :func:`fit`.
    n_inputs_in_ : int
        Number of inputs before transformation.
        Set by :func:`fit`.
    n_features_out_ : int
        Number of features after transformation, including episode feature.
        Set by :func:`fit`.
    n_states_out_ : int
        Number of states after transformation.
        Set by :func:`fit`.
    n_inputs_out_ : int
        Number of inputs after transformation.
        Set by :func:`fit`.
    min_samples_ : int
        Minimum number of samples needed to use the transformer.
        Set by :func:`fit`.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
        Set by :func:`fit`.

    Notes
    -----
    When :func:`fit` is called with multiple episodes, it only considers the
    first episode. It is assumed that the first episode contains all the
    information needed to properly fit the transformer. Typically, :func:`fit`
    just needs to know the dimensions of the data, so this is a reasonable
    assumption. When :func:`transform` and :func:`inverse_transform` are
    called, they apply the fit transformer to each episode individually.
    """

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = False) -> 'EpisodeDependentLiftingFn':
        # noqa: D102
        # Validate constructor parameters
        self._validate_parameters()
        # Validate fit parameters
        if n_inputs < 0:
            raise ValueError('`n_inputs` must be greater than or equal to 0.')
        # Save presence of episode feature
        self.episode_feature_ = episode_feature
        # Set up array checks. If you have an episode feature, you need at
        # least one other feature!
        self._check_array_params = {
            'ensure_min_features': 2 if episode_feature else 1,
        }
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Set number of input features (including episode feature)
        self.n_features_in_ = X.shape[1]
        # Split episodes
        episodes = split_episodes(X, episode_feature=self.episode_feature_)
        first_episode = episodes[0]
        X_first = first_episode[1]
        # Set states and inputs in
        self.n_inputs_in_ = n_inputs
        self.n_states_in_ = X_first.shape[1] - n_inputs
        n_x, n_u, n_k = self._fit_one_ep(X_first)
        self.n_states_out_ = n_x
        self.n_inputs_out_ = n_u
        self.n_features_out_ = (self.n_states_out_ + self.n_inputs_out_ +
                                (1 if self.episode_feature_ else 0))
        self.min_samples_ = n_k
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # noqa: D102
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'{self.__class__.__name__} `fit()` called '
                             f'with {self.n_features_in_} features, but '
                             f'`transform()` called with {X.shape[1]} '
                             'features.')
        return self._apply_transform_or_inverse(X, 'transform')

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        # noqa: D102
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
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
            ``'transform'`` to apply transform or ``'inverse_transform'`` to
            apply inverse transform.

        Returns
        -------
        np.ndarray
            Transformed or inverse transformed data matrix.

        Raises
        ------
        ValueError
            If ``transform`` is not  ``'transform'`` or
            ``'inverse_transform'``.
        """
        # Split episodes
        episodes = split_episodes(X, episode_feature=self.episode_feature_)
        # Transform each episode
        transformed_episodes = []
        for (i, X_i) in episodes:
            # Apply transform or inverse to individual episodes
            if transform == 'transform':
                transformed_episode = self._transform_one_ep(X_i)
            elif transform == 'inverse_transform':
                transformed_episode = self._inverse_transform_one_ep(X_i)
            else:
                raise ValueError("Parameter `transform` must be one of "
                                 "['transform', 'inverse_transform']")
            # Add new episode feature back if needed. This is necessary because
            # some transformations may modify the episode length.
            transformed_episodes.append((i, transformed_episode))
        # Concatenate the transformed episodes
        Xt = combine_episodes(transformed_episodes,
                              episode_feature=self.episode_feature_)
        return Xt

    @abc.abstractmethod
    def _fit_one_ep(self, X: np.ndarray) -> Tuple[int, int, int]:
        """Fit lifting function using a single episode.

        Expects and returns data without an episode header. Data is assumed to
        belong to a single episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        Tuple[int, int, int]
            Tuple containing the number of state features in the transformed
            data, the number of input features in the transformed data, and
            the minimum number of samples required to use the transformer.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
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
    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
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


class KoopmanRegressor(sklearn.base.BaseEstimator,
                       sklearn.base.RegressorMixin,
                       metaclass=abc.ABCMeta):
    """Base class for Koopman regressors.

    All attributes with a trailing underscore are set by :func:`fit`.

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

    # Array check parameters for :func:`fit` when ``X`` and ``y` are given
    _check_X_y_params: Dict[str, Any] = {
        'multi_output': True,
        'y_numeric': True,
    }

    # Array check parameters for :func:`predict` and :func:`fit` when only
    # ``X`` is given
    _check_array_params: Dict[str, Any] = {
        'dtype': 'numeric',
    }

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = False) -> 'KoopmanRegressor':
        """Fit the regressor.

        If only ``X`` is specified, the regressor will compute its unshifted
        and shifted versions. If ``X`` and ``y`` are specified, ``X`` is
        treated as the unshifted data matrix, while ``y`` is treated as the
        shifted data matrix.

        Parameters
        ----------
        X : np.ndarray
            Full data matrix if ``y=None``. Unshifted data matrix if ``y`` is
            specified.
        y : np.ndarray
            Optional shifted data matrix. If ``None``, shifted data matrix is
            computed using ``X``.
        n_inputs : int
            Number of input features at the end of ``X``.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.

        Returns
        -------
        KoopmanRegressor
            Instance of itself.

        Raises
        -----
        ValueError
            If constructor or fit parameters are incorrect.
        """
        # Check ``X`` differently depending on whether ``y`` is given
        if y is None:
            X = sklearn.utils.validation.check_array(
                X, **self._check_array_params)
        else:
            X, y = sklearn.utils.validation.check_X_y(X, y,
                                                      **self._check_X_y_params)
        # Validate constructor parameters
        self._validate_parameters()
        # Compute fit attributes
        self.n_features_in_ = X.shape[1]
        self.n_inputs_in_ = n_inputs
        self.n_states_in_ = (X.shape[1] - n_inputs -
                             (1 if episode_feature else 0))
        self.episode_feature_ = episode_feature
        # Split ``X`` if needed
        if y is None:
            X_unshifted, X_shifted = shift_episodes(
                X,
                n_inputs=self.n_inputs_in_,
                episode_feature=self.episode_feature_)
        else:
            X_unshifted = X
            X_shifted = y
        # Strip episode feature if present
        if self.episode_feature_:
            X_unshifted_noep = X_unshifted[:, 1:]
            X_shifted_noep = X_shifted[:, 1:]
        else:
            X_unshifted_noep = X_unshifted
            X_shifted_noep = X_shifted
        # Call fit from subclass
        self.coef_ = self._fit_regressor(X_unshifted_noep, X_shifted_noep)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform a single-step prediction for each state in each episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Predicted data matrix.
        """
        sklearn.utils.validation.check_is_fitted(self)
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Split episodes
        episodes = split_episodes(X, episode_feature=self.episode_feature_)
        # Predict for each episode
        predictions = []
        for (i, X_i) in episodes:
            predictions.append((i, X_i @ self.coef_))
        # Combine and return
        X_pred = combine_episodes(
            predictions,
            episode_feature=self.episode_feature_,
        )
        return X_pred

    @abc.abstractmethod
    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        """Fit the regressor using shifted and unshifted data matrices.

        The input data matrices must not have episode features.

        Parameters
        ----------
        X_unshifted : np.ndarray
            Unshifted data matrix without episode feature.
        X_shifted : np.ndarray
            Shifted data matrix without episode feature.

        Returns
        -------
        np.ndarray
            Fit coefficient matrix.
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

    # Extra estimator tags
    # https://scikit-learn.org/stable/developers/develop.html#estimator-tags
    def _more_tags(self):
        return {
            'multioutput': True,
            'multioutput_only': True,
        }


class SplitPipeline(metaestimators._BaseComposition, KoopmanLiftingFn):
    """Meta-estimator for lifting states and inputs separately.

    Only works with episode-independent lifting functions! It's too complicated
    to make this work with :class:`DelayLiftingFn`, especially when you can
    just set ``n_delays_input=0``.

    Attributes
    ----------
    lifting_functions_state_: List[Tuple[str, EpisodeIndependentLiftingFn]]
        Fit state lifting functions (and their names).
    lifting_functions_input_: List[Tuple[str, EpisodeIndependentLiftingFn]]
        Fit input lifting functions (and their names).
    n_features_in_ : int
        Number of features before transformation, including episode feature.
    n_states_in_ : int
        Number of states before transformation.
    n_inputs_in_ : int
        Number of inputs before transformation.
    n_features_out_ : int
        Number of features after transformation, including episode feature.
    n_states_out_ : int
        Number of states after transformation.
    n_inputs_out_ : int
        Number of inputs after transformation.
    min_samples_ : int
        Minimum number of samples needed to use the transformer.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.

    Examples
    --------
    Apply split pipeline to mass-spring-damper data

    >>> kp = pykoop.SplitPipeline(
    ...     lifting_functions_state=[
    ...         ('pl', pykoop.PolynomialLiftingFn(order=2))
    ...     ],
    ...     lifting_functions_input=None,
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)
    SplitPipeline(lifting_functions_state=[('pl',
    PolynomialLiftingFn(order=2))])
    >>> Xt_msd = kp.transform(X_msd[:2, :])
    """

    # Array check parameters for :func:`predict` and :func:`fit` when only
    # ``X`` is given
    _check_array_params = {
        'dtype': 'numeric',
    }

    def __init__(
        self,
        lifting_functions_state: List[Tuple[
            str, EpisodeIndependentLiftingFn]] = None,
        lifting_functions_input: List[Tuple[
            str, EpisodeIndependentLiftingFn]] = None,
    ) -> None:
        """Instantiate :class:`SplitPipeline`.

        Parameters
        ----------
        lifting_functions_state : List[Tuple[str, EpisodeIndependentLiftingFn]]
            Lifting functions to apply to the state features (and their names).
        lifting_functions_input : List[Tuple[str, EpisodeIndependentLiftingFn]]
            Lifting functions to apply to the input features (and their names).
        """
        self.lifting_functions_state = lifting_functions_state
        self.lifting_functions_input = lifting_functions_input

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = False) -> 'SplitPipeline':
        # noqa: D102
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Save state of episode feature
        self.episode_feature_ = episode_feature
        # Set number of features
        self.n_features_in_ = X.shape[1]
        self.n_states_in_ = (X.shape[1] - n_inputs -
                             (1 if episode_feature else 0))
        self.n_inputs_in_ = n_inputs
        # Clone state lifting functions
        used_keys = []
        self.lifting_functions_state_ = []
        if self.lifting_functions_state is not None:
            for key, lf in self.lifting_functions_state:
                used_keys.append(key)
                self.lifting_functions_state_.append(
                    tuple((key, sklearn.base.clone(lf))))
        # Clone input lifting functions
        self.lifting_functions_input_ = []
        if self.lifting_functions_input is not None:
            for key, lf in self.lifting_functions_input:
                used_keys.append(key)
                self.lifting_functions_input_.append(
                    tuple((key, sklearn.base.clone(lf))))
        # Check names
        self._validate_names(used_keys)
        # Separate episode feature
        if self.episode_feature_:
            X_ep = X[:, [0]]
            X = X[:, 1:]
        # Split state and input
        X_state = X[:, :self.n_states_in_]
        X_input = X[:, self.n_states_in_:]
        # Put back episode feature if needed
        if self.episode_feature_:
            X_state = np.hstack((
                X_ep,
                X_state,
            ))
            X_input = np.hstack((
                X_ep,
                X_input,
            ))
        # Fit and transform states
        X_out_state = X_state
        for _, lf in self.lifting_functions_state_:
            X_out_state = lf.fit_transform(
                X_out_state,
                n_inputs=0,
                episode_feature=self.episode_feature_,
            )
        # Fit and transform inputs
        X_out_input = X_input
        for _, lf in self.lifting_functions_input_:
            X_out_input = lf.fit_transform(
                X_out_input,
                n_inputs=0,
                episode_feature=self.episode_feature_,
            )
        # Compute output dimensions for states
        if len(self.lifting_functions_state_) > 0:
            # Compute number of output states
            last_tf = self.lifting_functions_state_[-1][1]
            if last_tf.n_inputs_out_ != 0:
                raise RuntimeError(f'Lifting function {last_tf} was called '
                                   'with `n_inputs=0` but `n_inputs_out_` is '
                                   'not 0. Is it implemented correctly?')
            self.n_states_out_ = last_tf.n_states_out_
        else:
            self.n_states_out_ = self.n_states_in_
        # Compute output dimensions for inputs
        if len(self.lifting_functions_input_) > 0:
            # Compute number of output states
            last_tf = self.lifting_functions_input_[-1][1]
            if last_tf.n_inputs_out_ != 0:
                raise RuntimeError(f'Lifting function {last_tf} was called '
                                   'with `n_inputs=0` but `n_inputs_out_` is '
                                   'not 0. Is it implemented correctly?')
            self.n_inputs_out_ = last_tf.n_states_out_
        else:
            self.n_inputs_out_ = self.n_inputs_in_
        # Compute number of features and minimum samples needed
        self.n_features_out_ = (self.n_states_out_ + self.n_inputs_out_ +
                                (1 if self.episode_feature_ else 0))
        # Since all lifting functions are episode-independent, we only ever
        # need one sample.
        self.min_samples_ = 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # noqa: D102
        sklearn.utils.validation.check_is_fitted(self)
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'{self.__class__.__name__} `fit()` called '
                             f'with {self.n_features_in_} features, but '
                             f'`transform()` called with {X.shape[1]} '
                             'features.')
        # Separate episode feature
        if self.episode_feature_:
            X_ep = X[:, [0]]
            X = X[:, 1:]
        # Split state and input
        X_state = X[:, :self.n_states_in_]
        X_input = X[:, self.n_states_in_:]
        # Put back episode feature if needed
        if self.episode_feature_:
            X_state = np.hstack((
                X_ep,
                X_state,
            ))
            X_input = np.hstack((
                X_ep,
                X_input,
            ))
        # Fit and transform states
        X_out_state = X_state
        for _, lf in self.lifting_functions_state_:
            X_out_state = lf.transform(X_out_state)
        # Fit and transform inputs
        X_out_input = X_input
        for _, lf in self.lifting_functions_input_:
            X_out_input = lf.transform(X_out_input)
        if self.episode_feature_:
            Xt = np.hstack((
                X_out_state,
                X_out_input[:, 1:],
            ))
        else:
            Xt = np.hstack((
                X_out_state,
                X_out_input,
            ))
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        # noqa: D102
        sklearn.utils.validation.check_is_fitted(self)
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_out_:
            raise ValueError(f'{self.__class__.__name__} `fit()` output '
                             f'{self.n_features_out_} features, but '
                             '`inverse_transform()` called with '
                             f'{X.shape[1]} features.')
        if self.episode_feature_:
            X_ep = X[:, [0]]
            X = X[:, 1:]
        # Split state and input
        X_state = X[:, :self.n_states_out_]
        X_input = X[:, self.n_states_out_:]
        # Put back episode feature if needed
        if self.episode_feature_:
            X_state = np.hstack((
                X_ep,
                X_state,
            ))
            X_input = np.hstack((
                X_ep,
                X_input,
            ))
        # Fit and inverse transform states
        X_out_state = X_state
        for _, lf in self.lifting_functions_state_[::-1]:
            X_out_state = lf.inverse_transform(X_out_state)
        # Fit and transform inputs
        X_out_input = X_input
        for _, lf in self.lifting_functions_input_[::-1]:
            X_out_input = lf.inverse_transform(X_out_input)
        if self.episode_feature_:
            Xt = np.hstack((
                X_out_state,
                X_out_input[:, 1:],
            ))
        else:
            Xt = np.hstack((
                X_out_state,
                X_out_input,
            ))
        return Xt

    def n_samples_in(self, n_samples_out: int = 1) -> int:
        # noqa: D102
        # Since this pipeline only works with episode-independent lifting
        # functions, we know ``n_samples_in == n_samples_out``.
        return n_samples_out

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        # noqa: D102
        # A bit inefficient to do this twice but it's not the end of the world.
        state = self._get_params('lifting_functions_state', deep=deep)
        input = self._get_params('lifting_functions_input', deep=deep)
        state.update(input)
        return state

    def set_params(self, **kwargs) -> 'SplitPipeline':
        # noqa: D102
        # A bit inefficient to do this twice but it's not the end of the world.
        self._set_params('lifting_functions_state', **kwargs)
        self._set_params('lifting_functions_input', **kwargs)
        return self


class KoopmanPipeline(metaestimators._BaseComposition,
                      sklearn.base.BaseEstimator,
                      sklearn.base.TransformerMixin, _LiftRetractMixin):
    """Meta-estimator for chaining lifting functions with an estimator.

    Attributes
    ----------
    liting_functions_ : List[Tuple[str, KoopmanLiftingFn]]
        Fit lifting functions (and their names).
    regressor_ : KoopmanRegressor
        Fit regressor.
    transformers_fit_ : bool
        True if lifting functions have been fit.
    regressor_fit_ : bool
        True if regressor has been fit.
    n_features_in_ : int
        Number of features before transformation, including episode feature.
    n_states_in_ : int
        Number of states before transformation.
    n_inputs_in_ : int
        Number of inputs before transformation.
    n_features_out_ : int
        Number of features after transformation, including episode feature.
    n_states_out_ : int
        Number of states after transformation.
    n_inputs_out_ : int
        Number of inputs after transformation.
    min_samples_ : int
        Minimum number of samples needed to use the transformer.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.

    Examples
    --------
    Apply a basic Koopman pipeline to mass-spring-damper data

    >>> kp = pykoop.KoopmanPipeline(
    ...     regressor=pykoop.Edmd(),
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)
    KoopmanPipeline(regressor=Edmd())

    Apply more sophisticated Koopman pipeline to mass-spring-damper data

    >>> kp = KoopmanPipeline(
    ...     lifting_functions=[
    ...         ('ma', pykoop.SkLearnLiftingFn(
    ...                    sklearn.preprocessing.MaxAbsScaler())),
    ...         ('pl', pykoop.PolynomialLiftingFn(order=2)),
    ...         ('ss', pykoop.SkLearnLiftingFn(
    ...                    sklearn.preprocessing.StandardScaler())),
    ...     ],
    ...     regressor=pykoop.Edmd(),
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)
    KoopmanPipeline(lifting_functions=[('ma',
    SkLearnLiftingFn(transformer=MaxAbsScaler())),
    ('pl', PolynomialLiftingFn(order=2)),
    ('ss', SkLearnLiftingFn(transformer=StandardScaler()))],
    regressor=Edmd())
    >>> Xt_msd = kp.transform(X_msd[:2, :])

    Apply bilinear Koopman pipeline to mass-spring-damper data

    >>> kp = KoopmanPipeline(
    ...     lifting_functions=[
    ...         ('ma', pykoop.SkLearnLiftingFn(
    ...                    sklearn.preprocessing.MaxAbsScaler())),
    ...         ('sp', pykoop.SplitPipeline(
    ...             lifting_functions_state=[
    ...                 ('pl', pykoop.PolynomialLiftingFn(order=2)),
    ...             ],
    ...             lifting_functions_input=None,
    ...         )),
    ...         ('bi', pykoop.BilinearInputLiftingFn()),
    ...         ('ss', pykoop.SkLearnLiftingFn(
    ...                    sklearn.preprocessing.StandardScaler())),
    ...     ],
    ...     regressor=pykoop.Edmd(),
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)
    KoopmanPipeline(lifting_functions=[('ma',
    SkLearnLiftingFn(transformer=MaxAbsScaler())),
    ('sp', SplitPipeline(lifting_functions_state=[('pl',
    PolynomialLiftingFn(order=2))])),
    ('bi', BilinearInputLiftingFn()),
    ('ss', SkLearnLiftingFn(transformer=StandardScaler()))],
    regressor=Edmd())
    """

    # Array check parameters for :func:`predict` and :func:`fit` when only
    # ``X`` is given
    _check_array_params = {
        'dtype': 'numeric',
    }

    def __init__(
        self,
        lifting_functions: List[Tuple[str, KoopmanLiftingFn]] = None,
        regressor: KoopmanRegressor = None,
    ) -> None:
        """Instantiate for :class:`KoopmanPipeline`.

        Parameters
        ----------
        lifting_functions : List[Tuple[str, KoopmanLiftingFn]]
            List of names and lifting function objects.
        regressor : KoopmanRegressor
            Koopman regressor.
        """
        self.lifting_functions = lifting_functions
        self.regressor = regressor

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = False) -> 'KoopmanPipeline':
        """Fit the Koopman pipeline.

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
        KoopmanPipeline
            Instance of itself.

        Raises
        -----
        ValueError
            If constructor or fit parameters are incorrect.
        """
        X = sklearn.utils.validation.check_array(
            X,
            ensure_min_samples=2,
            **self._check_array_params,
        )
        if self.regressor is None:
            raise ValueError(
                '`regressor` must be specified in order to use `fit()`.')
        # Clone regressor
        self.regressor_ = sklearn.base.clone(self.regressor)
        # Fit transformers and transform input
        self.fit_transformers(
            X,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        Xt = self.transform(X)
        # Fit the regressor
        self.regressor_.fit(
            Xt,
            n_inputs=self.n_inputs_out_,
            episode_feature=self.episode_feature_,
        )
        self.regressor_fit_ = True
        return self

    def fit_transformers(self,
                         X: np.ndarray,
                         y: np.ndarray = None,
                         n_inputs: int = 0,
                         episode_feature: bool = False) -> 'KoopmanPipeline':
        """Fit only the lifting functions in the pipeline.

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
        KoopmanPipeline
            Instance of itself.

        Raises
        -----
        ValueError
            If constructor or fit parameters are incorrect.
        """
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Save state of episode feature
        self.episode_feature_ = episode_feature
        # Set number of features
        self.n_features_in_ = X.shape[1]
        self.n_states_in_ = (X.shape[1] - n_inputs -
                             (1 if episode_feature else 0))
        self.n_inputs_in_ = n_inputs
        used_keys = []
        self.lifting_functions_ = []
        if self.lifting_functions is not None:
            for key, lf in self.lifting_functions:
                used_keys.append(key)
                self.lifting_functions_.append(
                    tuple((key, sklearn.base.clone(lf))))
        # Check names
        self._validate_names(used_keys)
        # Fit and transform lifting functions
        X_out = X
        n_inputs_out = n_inputs
        for _, lf in self.lifting_functions_:
            X_out = lf.fit_transform(X_out,
                                     n_inputs=n_inputs_out,
                                     episode_feature=episode_feature)
            n_inputs_out = lf.n_inputs_out_
        # Set output dimensions
        if len(self.lifting_functions_) > 0:
            # Find the last transformer and use it to get output dimensions
            last_pp = self.lifting_functions_[-1][1]
            self.n_features_out_ = last_pp.n_features_out_
            self.n_states_out_ = last_pp.n_states_out_
            self.n_inputs_out_ = last_pp.n_inputs_out_
            # Compute minimum number of samples needed by transformer.
            # Each transformer knows how many input samples it needs to produce
            # a given number of output samples.  Knowing we just want one
            # sample at the output, we work backwards to figure out how many
            # samples we need at the beginning of the pipeline.
            n_samples_out = 1
            for _, tf in self.lifting_functions_[::-1]:
                n_samples_out = tf.n_samples_in(n_samples_out)
            self.min_samples_ = n_samples_out
        else:
            # Fall back on input dimensions
            self.n_features_out_ = self.n_features_in_
            self.n_states_out_ = self.n_states_in_
            self.n_inputs_out_ = self.n_inputs_in_
            self.min_samples_ = 1
        self.transformers_fit_ = True
        return self

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
        sklearn.utils.validation.check_is_fitted(self, 'transformers_fit_')
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'{self.__class__.__name__} `fit()` called '
                             f'with {self.n_features_in_} features, but '
                             f'`transform()` called with {X.shape[1]} '
                             'features.')
        # Apply lifting functions
        X_out = X
        for _, lf in self.lifting_functions_:
            X_out = lf.transform(X_out)
        return X_out

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
        sklearn.utils.validation.check_is_fitted(self, 'transformers_fit_')
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_out_:
            raise ValueError(f'{self.__class__.__name__} `fit()` output '
                             f'{self.n_features_out_} features, but '
                             '`inverse_transform()` called with '
                             f'{X.shape[1]} features.')
        # Apply inverse lifting functions in reverse order
        X_out = X
        for _, lf in self.lifting_functions_[::-1]:
            X_out = lf.inverse_transform(X_out)
        return X_out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform a single-step prediction for each state in each episode.

        Lifts the state, preforms a single-step prediction in the lifted space,
        then retracts to the original state space.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Predicted data matrix.
        """
        sklearn.utils.validation.check_is_fitted(self, 'regressor_fit_')
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Lift data matrix
        X_trans = self.transform(X)
        # Predict in lifted space
        X_pred = self.regressor_.predict(X_trans)
        # Pad inputs wth zeros to do inverse
        if self.n_inputs_out_ != 0:
            X_pred_pad = np.hstack((
                X_pred,
                np.zeros((X_pred.shape[0], self.n_inputs_out_)),
            ))
        else:
            X_pred_pad = X_pred
        # Invert lifting functions
        X_pred_pad_inv = self.inverse_transform(X_pred_pad)
        # Strip zero inputs
        if self.n_inputs_in_ != 0:
            X_pred_inv = X_pred_pad_inv[:, :(self.n_features_in_
                                             - self.n_inputs_in_)]
        else:
            X_pred_inv = X_pred_pad_inv
        return X_pred_inv

    def score(self, X: np.ndarray, y: np.ndarray = None) -> float:
        """Calculate prediction score.

        For more flexible scoring, see :func:`make_scorer`.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.

        Returns
        -------
        float
            Mean squared error prediction score.
        """
        sklearn.utils.validation.check_is_fitted(self, 'regressor_fit_')
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        scorer = KoopmanPipeline.make_scorer()
        score = scorer(self, X, None)
        return score

    @deprecated('Use `predict_trajectory` instead')
    def predict_multistep(self, X: np.ndarray) -> np.ndarray:
        """Perform a multi-step prediction for the first state of each episode.

        This function takes the first ``min_samples_`` states of the input,
        along with all of its inputs, and predicts the next ``X.shape[0]``
        states of the system. This action is performed on a per-episode basis.
        The state features of ``X`` (other than the first ``min_samples_``
        features) are not used at all.

        If prediction fails numerically, missing predictions are filled with
        ``np.nan``.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Predicted data matrix.

        Raises
        ------
        ValueError
            If an episode is shorter than ``min_samples_``.
        """
        sklearn.utils.validation.check_is_fitted(self, 'regressor_fit_')
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Split episodes
        episodes = split_episodes(X, episode_feature=self.episode_feature_)
        # Loop over episodes
        predictions = []
        for (i, X_i) in episodes:
            # Check length of episode.
            if X_i.shape[0] < self.min_samples_:
                raise ValueError(f'Episode {i} has {X_i.shape[0]} samples but '
                                 f'`min_samples_`={self.min_samples_} samples '
                                 'are required.')
            # Index where prediction blows up (if it does)
            crash_index = None
            # Extract initial state and input
            x0 = X_i[:self.min_samples_, :self.n_states_in_]
            u = X_i[:, self.n_states_in_:]
            # Create array to hold predicted states
            X_pred_i = np.zeros((X_i.shape[0], self.n_states_in_))
            # Set the initial condition
            X_pred_i[:self.min_samples_, :] = x0
            # Predict all time steps
            for k in range(self.min_samples_, X_i.shape[0]):
                # Stack episode feature, previous predictions, and input
                X_ik = combine_episodes(
                    [(i,
                      np.hstack((
                          X_pred_i[(k - self.min_samples_):k, :],
                          X_i[(k - self.min_samples_):k, self.n_states_in_:],
                      )))],
                    episode_feature=self.episode_feature_)
                # Predict next step
                try:
                    X_pred_ik = self.predict(X_ik)[[-1], :]
                except ValueError:
                    crash_index = k
                    break
                # Extract data matrix from prediction
                X_pred_i[[k], :] = split_episodes(
                    X_pred_ik, episode_feature=self.episode_feature_)[0][1]
            if crash_index is not None:
                X_pred_i[crash_index:, :] = np.nan
            predictions.append((i, X_pred_i))
        # Combine episodes
        X_p = combine_episodes(predictions,
                               episode_feature=self.episode_feature_)
        return X_p

    def predict_trajectory(
        self,
        X_initial: np.ndarray,
        U: np.ndarray,
        relift_state: bool = True,
        return_lifted: bool = False,
        return_input: bool = False,
        episode_feature: bool = None,
    ) -> np.ndarray:
        """Predict state trajectory given input for each episode.

        Parameters
        ----------
        X_initial : np.ndarray
            Initial state.
        U : np.ndarray
            Input. Length of prediction is governed by length of input.
        relift_state : bool
            If true, retract and re-lift state between prediction steps
            (default). Otherwise, only retract the state after all predictions
            are made. Correspond to the local and global error definitions of
            [MAM22]_.
        return_lifted : bool
            If true, return the lifted state. If false, return the original
            state (default).
        return_input : bool
            If true, return the input as well as the state. If false, return
            only the original state (default).
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            Predicted state. If ``return_input``, input is appended to the
            array. If ``return_lifted``, the predicted state (and input) are
            returned in the lifted space.

        Raises
        ------
        ValueError
            If an episode is shorter than ``min_samples_``.
        """
        # Check fit
        sklearn.utils.validation.check_is_fitted(self, 'regressor_fit_')
        # Set episode feature if unspecified
        if episode_feature is None:
            episode_feature = self.episode_feature_
        # Get Koopman ``A`` and ``B`` matrices
        koop_mat = self.regressor_.coef_.T
        A = koop_mat[:, :koop_mat.shape[0]]
        B = koop_mat[:, koop_mat.shape[0]:]
        # Split episodes
        ep_X0 = split_episodes(X_initial, episode_feature=episode_feature)
        ep_U = split_episodes(U, episode_feature=episode_feature)
        episodes = [(ex[0], ex[1], eu[1]) for (ex, eu) in zip(ep_X0, ep_U)]
        # Predict for each episode
        predictions: List[Tuple[float, np.ndarray]] = []
        for (i, X0_i, U_i) in episodes:
            # Check length of episode.
            if X0_i.shape[0] != self.min_samples_:
                raise ValueError(f'Initial condition in episode {i} has '
                                 f'{X0_i.shape[0]} samples but `min_samples_`='
                                 f'{self.min_samples_} samples are required.')
            if U_i.shape[0] < self.min_samples_:
                raise ValueError(f'Input in episode {i} has {U_i.shape[0]} '
                                 'samples but at least `min_samples_`='
                                 f'{self.min_samples_} samples are required.')
            crash_index = None
            # Iterate over episode and make predictions
            if relift_state:
                # Number of steps in episode
                n_steps_i = U_i.shape[0]
                # Initial conditions
                X_i = np.zeros((n_steps_i, self.n_states_in_))
                X_i[:self.min_samples_, :] = X0_i
                for k in range(self.min_samples_, n_steps_i):
                    try:
                        # Lift state and input
                        window = np.s_[(k - self.min_samples_):k]
                        Theta_ikm1 = self.lift_state(
                            X_i[window, :],
                            episode_feature=False,
                        )
                        Upsilon_ikm1 = self.lift_input(
                            np.hstack((X_i[window, :], U_i[window, :])),
                            episode_feature=False,
                        )
                        # Predict
                        Theta_ik = Theta_ikm1 @ A.T + Upsilon_ikm1 @ B.T
                        # Retract. If more than one sample is returned by
                        # ``retract_state``, take only the last one. This will
                        # happen if there's a delay lifting function.
                        X_i[[k], :] = self.retract_state(
                            Theta_ik,
                            episode_feature=False,
                        )[[-1], :]
                    except ValueError as ve:
                        if (np.all(np.isfinite(Theta_ikm1))
                                and np.all(np.isfinite(X_i))
                                and np.all(np.isfinite(U_i))
                                and np.all(np.isfinite(Upsilon_ikm1))
                                and np.all(np.isfinite(Theta_ik))):
                            raise ve
                        else:
                            crash_index = k - 1
                            X_i[crash_index:, :] = 0
                            break
                Theta_i = self.lift_state(X_i, episode_feature=False)
                Upsilon_i = self.lift_input(
                    np.hstack((X_i, U_i)),
                    episode_feature=False,
                )
            else:
                # Number of steps in episode
                n_steps_i = U_i.shape[0] - self.min_samples_
                # Initial conditions
                Theta_i = np.zeros((n_steps_i, self.n_states_out_))
                Upsilon_i = np.zeros((n_steps_i, self.n_inputs_out_))
                Theta_i[[0], :] = self.lift_state(X0_i, episode_feature=False)
                for k in range(1, n_steps_i + 1):
                    try:
                        X_ikm1 = self.retract_state(
                            Theta_i[[k - 1], :],
                            episode_feature=False,
                        )
                        window = np.s_[k:(k + self.min_samples_)]
                        Upsilon_i[[k - 1], :] = self.lift_input(
                            np.hstack((X_ikm1, U_i[window, :])),
                            episode_feature=False,
                        )
                        # Predict
                        if k < n_steps_i:
                            Theta_i[[k], :] = (Theta_i[[k - 1], :] @ A.T
                                               + Upsilon_i[[k - 1], :] @ B.T)
                    except ValueError as ve:
                        if (np.all(np.isfinite(X_ikm1))
                                and np.all(np.isfinite(Theta_i))
                                and np.all(np.isfinite(Upsilon_i))
                                and np.all(np.isfinite(U_i))):
                            raise ve
                        else:
                            crash_index = k - 1
                            Theta_i[crash_index:, :] = 0
                            Upsilon_i[crash_index:, :] = 0
                            break
                X_i = self.retract_state(Theta_i, episode_feature=False)
            # If prediction crashed, set remaining entries to NaN
            if crash_index is not None:
                log.warning(f'Prediction diverged at index {crash_index}. '
                            'Remaining entries set to `NaN`.')
                # Don't set ``U_i`` to NaN since it's a known input
                X_i[crash_index:, :] = np.nan
                Theta_i[crash_index:, :] = np.nan
                Upsilon_i[crash_index:, :] = np.nan
            # Choose what to return
            if return_lifted:
                if return_input:
                    predictions.append((i, np.hstack((Theta_i, Upsilon_i))))
                else:
                    predictions.append((i, Theta_i))
            else:
                if return_input:
                    predictions.append((i, np.hstack((X_i, U_i))))
                else:
                    predictions.append((i, X_i))
        # Combine episodes
        combined_episodes = combine_episodes(
            predictions,
            episode_feature=episode_feature,
        )
        return combined_episodes

    @staticmethod
    def make_scorer(
        n_steps: int = None,
        discount_factor: float = 1,
        regression_metric: str = 'neg_mean_squared_error',
        multistep: bool = True,
        relift_state: bool = True,
    ) -> Callable[['KoopmanPipeline', np.ndarray, Optional[np.ndarray]],
                  float]:
        """Make a Koopman pipeline scorer.

        A ``scikit-learn`` scorer accepts the parameters ``(estimator, X, y)``
        and returns a float representing the prediction quality of
        ``estimator`` on ``X`` with reference to ``y``. Uses existing
        ``scikit-learn`` regression metrics [#sc]_. Higher numbers are better.
        Metrics corresponding to losses are negated.

        Technically, the scorer will predict the entire episode, regardless of
        how ``n_steps`` is set. It will then assign a zero weight to all errors
        beyond ``n_steps``.

        Parameters
        ----------
        n_steps : int
            Number of steps ahead to predict. If ``None`` or longer than the
            episode, will score the entire episode.

        discount_factor : float
            Discount factor used to weight the error timeseries. Should be
            positive, with magnitude 1 or slightly less. The error at each
            timestep is weighted by ``discount_factor**k``, where ``k`` is the
            timestep.

        regression_metric : str
            Regression metric to use. One of

            - ``'explained_variance'``,
            - ``'neg_mean_absolute_error'``,
            - ``'neg_mean_squared_error'``,
            - ``'neg_mean_squared_log_error'``,
            - ``'neg_median_absolute_error'``,
            - ``'r2'``, or
            - ``'neg_mean_absolute_percentage_error'``,

            which are existing ``scikit-learn`` regression metrics [#sc]_.

        multistep : bool
            If true, predict using :func:`predict_trajectory`. Otherwise,
            predict using :func:`predict` (one-step-ahead prediction).
            Multistep prediction is highly recommended unless debugging. If
            one-step-ahead prediciton is used, `n_steps` and `discount_factor`
            are ignored.

        relift_state : bool
            If true, retract and re-lift state between prediction steps
            (default). Otherwise, only retract the state after all predictions
            are made. Correspond to the local and global error definitions of
            [MAM22]_. Ignored if ``multistep`` is false.

        Returns
        -------
        Callable[[KoopmanPipeline, np.ndarray, Optional[np.ndarray]], float]
            Scorer compatible with ``scikit-learn``.

        Raises
        ------
        ValueError
            If ``discount_factor`` is negative or greater than one.

        References
        ----------
        .. [#sc] https://scikit-learn.org/stable/modules/model_evaluation.html
        """

        def koopman_pipeline_scorer(
            estimator: KoopmanPipeline,
            X: np.ndarray,
            y: np.ndarray = None,
        ) -> float:
            # Shift episodes
            X_unshifted, X_shifted = shift_episodes(
                X,
                n_inputs=estimator.n_inputs_in_,
                episode_feature=estimator.episode_feature_,
            )
            # Predict
            if multistep:
                # Get initial conditions for each episode
                x0 = extract_initial_conditions(
                    X_unshifted,
                    min_samples=estimator.min_samples_,
                    n_inputs=estimator.n_inputs_in_,
                    episode_feature=estimator.episode_feature_,
                )
                # Get inputs for each episode
                u = extract_input(
                    X_unshifted,
                    n_inputs=estimator.n_inputs_in_,
                    episode_feature=estimator.episode_feature_,
                )
                # Predict state for each episode
                X_predicted = estimator.predict_trajectory(
                    x0,
                    u,
                    relift_state=relift_state,
                )
                # Score prediction
                score = score_trajectory(
                    X_predicted,
                    X_shifted,
                    n_steps=n_steps,
                    discount_factor=discount_factor,
                    regression_metric=regression_metric,
                    min_samples=estimator.min_samples_,
                    episode_feature=estimator.episode_feature_,
                )
            else:
                # Warn about ignored non-default arguments
                if not relift_state:
                    log.info('Ignoring `relift_state` since `multistep` is '
                             'false.')
                if n_steps is not None:
                    log.info('Ignoring `n_steps` since `multistep` is false.')
                if discount_factor != 1:
                    log.info('Ignoring `discount_factor` since `multistep` is '
                             'false.')
                # Perform single-step prediction
                X_predicted = estimator.predict(X_unshifted)
                # Score prediction
                score = score_trajectory(
                    X_predicted,
                    X_shifted,
                    n_steps=None,
                    discount_factor=1,
                    regression_metric=regression_metric,
                    min_samples=estimator.min_samples_,
                    episode_feature=estimator.episode_feature_,
                )
            return score

        return koopman_pipeline_scorer

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        # noqa: D102
        return self._get_params('lifting_functions', deep=deep)

    def set_params(self, **kwargs) -> 'KoopmanPipeline':
        # noqa: D102
        self._set_params('lifting_functions', **kwargs)
        return self


def score_trajectory(
    X_predicted: np.ndarray,
    X_expected: np.ndarray,
    n_steps: int = None,
    discount_factor: float = 1,
    regression_metric: str = 'neg_mean_squared_error',
    min_samples: int = 1,
    episode_feature: bool = False,
) -> float:
    """Score a predicted data matrix compared to an expected data matrix.

    Parameters
    ----------
    X_predicted : np.ndarray
        Predicted state data matrix.

    X_expected : np.ndarray
        Expected state data matrix.

    n_steps : int
        Number of steps ahead to predict. If ``None`` or longer than the
        episode, will score the entire episode.

    discount_factor : float
        Discount factor used to weight the error timeseries. Should be
        positive, with magnitude 1 or slightly less. The error at each
        timestep is weighted by ``discount_factor**k``, where ``k`` is the
        timestep.

    regression_metric : str
        Regression metric to use. One of

        - ``'explained_variance'``,
        - ``'neg_mean_absolute_error'``,
        - ``'neg_mean_squared_error'``,
        - ``'neg_mean_squared_log_error'``,
        - ``'neg_median_absolute_error'``,
        - ``'r2'``, or
        - ``'neg_mean_absolute_percentage_error'``,

        which are existing ``scikit-learn`` regression metrics [#sc]_.

    min_samples : int
        Number of samples in initial condition.

    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    float
        Score (greater is better).

    References
    ----------
    .. [#sc] https://scikit-learn.org/stable/modules/model_evaluation.html
    """
    # Valid ``regression_metric`` values:
    regression_metrics = {
        'explained_variance':
        sklearn.metrics.explained_variance_score,
        'r2':
        sklearn.metrics.r2_score,
        'neg_mean_absolute_error':
        sklearn.metrics.mean_absolute_error,
        'neg_mean_squared_error':
        sklearn.metrics.mean_squared_error,
        'neg_mean_squared_log_error':
        sklearn.metrics.mean_squared_log_error,
        'neg_median_absolute_error':
        sklearn.metrics.median_absolute_error,
        'neg_mean_absolute_percentage_error':
        sklearn.metrics.mean_absolute_percentage_error,
    }
    # Scores that do not need inversion
    greater_is_better = ['explained_variance', 'r2']
    # Strip episode feature and initial conditions
    X_expected = strip_initial_conditions(
        X_expected,
        min_samples=min_samples,
        episode_feature=episode_feature,
    )
    X_predicted = strip_initial_conditions(
        X_predicted,
        min_samples=min_samples,
        episode_feature=episode_feature,
    )
    # Compute weights
    weights = _weights_from_data_matrix(
        X_expected,
        n_steps=n_steps,
        discount_factor=discount_factor,
        episode_feature=episode_feature,
    )
    # Strip episode feature if present
    if episode_feature:
        X_expected = X_expected[:, 1:]
        X_predicted = X_predicted[:, 1:]
    # Calculate score
    score = regression_metrics[regression_metric](
        X_expected,
        X_predicted,
        sample_weight=weights,
        multioutput='uniform_average',
    )
    # Invert losses
    if regression_metric not in greater_is_better:
        score *= -1
    return score


def extract_initial_conditions(
    X: np.ndarray,
    min_samples: int = 1,
    n_inputs: int = 0,
    episode_feature: bool = False,
) -> np.ndarray:
    """Extract initial conditions from each episode.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    min_samples : int
        Number of samples in initial condition.
    n_inputs : int
        Number of input features at the end of ``X``.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    np.ndarray
        Initial conditions from each episode.
    """
    episodes = split_episodes(X, episode_feature=episode_feature)
    # Strip each episode
    initial_conditions = []
    for (i, X_i) in episodes:
        if n_inputs == 0:
            initial_condition = X_i[:min_samples, :]
        else:
            initial_condition = X_i[:min_samples, :-n_inputs]
        initial_conditions.append((i, initial_condition))
    # Concatenate the initial conditions
    X0 = combine_episodes(initial_conditions, episode_feature=episode_feature)
    return X0


def extract_input(
    X: np.ndarray,
    n_inputs: int = 0,
    episode_feature: bool = False,
) -> np.ndarray:
    """Extract input from a data matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    n_inputs : int
        Number of input features at the end of ``X``.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    np.ndarray
        Input extracted from data matrix.
    """
    episodes = split_episodes(X, episode_feature=episode_feature)
    # Strip each episode
    inputs = []
    for (i, X_i) in episodes:
        if n_inputs == 0:
            input_ = np.zeros((X_i.shape[0], 0))
        else:
            n_states = X_i.shape[1] - n_inputs
            input_ = X_i[:, n_states:]
        inputs.append((i, input_))
    # Concatenate the inputs
    u = combine_episodes(inputs, episode_feature=episode_feature)
    return u


def strip_initial_conditions(X: np.ndarray,
                             min_samples: int = 1,
                             episode_feature: bool = False) -> np.ndarray:
    """Strip initial conditions from each episode.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    min_samples : int
        Number of samples in initial condition.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    np.ndarray
        Data matrix with initial conditions removed.
    """
    episodes = split_episodes(X, episode_feature=episode_feature)
    # Strip each episode
    stripped_episodes = []
    for (i, X_i) in episodes:
        stripped_episode = X_i[min_samples:, :]
        stripped_episodes.append((i, stripped_episode))
    # Concatenate the stripped episodes
    Xs = combine_episodes(stripped_episodes, episode_feature=episode_feature)
    return Xs


def shift_episodes(
        X: np.ndarray,
        n_inputs: int = 0,
        episode_feature: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Shift episodes and truncate shifted inputs.

    The Koopman matrix ``K`` approximately satisfies::

        Theta_+ = Psi @ K.T

    where ``Psi`` contains the unshifted states and inputs, and ``Theta_+``
    contains the shifted states.

    The regressors used in :class:`KoopmanPipeline` expect ``Psi`` as their
    ``X`` and ``Theta_+`` as their ``y``. This function breaks its input (also
    named ``X``) into ``Psi`` and ``Theta_+`` for use with these regressors.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    n_inputs : int
        Number of input features at the end of ``X``.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple whose first element is the unshifted array and whose second
        element is the shifted array with its inputs truncated. Both arrays
        have the same number of samples. Their episode features are stripped if
        present.
    """
    # Split episodes
    episodes = split_episodes(X, episode_feature=episode_feature)
    # Shift each episode
    unshifted_episodes = []
    shifted_episodes = []
    for i, X_i in episodes:
        # Get unshifted episode
        X_i_unshifted = X_i[:-1, :]
        # Get shifted episode. Strip input if present.
        if n_inputs == 0:
            X_i_shifted = X_i[1:, :]
        else:
            X_i_shifted = X_i[1:, :-n_inputs]
        # Append to episode list
        unshifted_episodes.append((i, X_i_unshifted))
        shifted_episodes.append((i, X_i_shifted))
    # Recombine and return
    X_unshifted = combine_episodes(unshifted_episodes,
                                   episode_feature=episode_feature)
    X_shifted = combine_episodes(shifted_episodes,
                                 episode_feature=episode_feature)
    return (X_unshifted, X_shifted)


def split_episodes(
        X: np.ndarray,
        episode_feature: bool = False) -> List[Tuple[float, np.ndarray]]:
    """Split a data matrix into episodes.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    List[Tuple[float, np.ndarray]]
        List of episode tuples. The first element of each tuple contains the
        episode index. The second element contains the episode data.
    """
    # Extract episode feature
    if episode_feature:
        X_ep = X[:, 0]
        X = X[:, 1:]
    else:
        X_ep = np.zeros((X.shape[0], ))
    # Split X into list of episodes. Each episode is a tuple containing
    # its index and its associated data matrix.
    episodes = []
    # ``pandas.unique`` is faster than ``np.unique`` and preserves order.
    for i in pandas.unique(X_ep):
        episodes.append((i, X[X_ep == i, :]))
    # Return list of episodes
    return episodes


def combine_episodes(episodes: List[Tuple[float, np.ndarray]],
                     episode_feature: bool = False) -> np.ndarray:
    """Combine episodes into a data matrix.

    Parameters
    ----------
    episodes : List[Tuple[float, np.ndarray]]
        List of episode tuples. The first element of each tuple contains the
        episode index. The second element contains the episode data.
    episode_feature : bool
        True if first feature of output should indicate which episode a
        timestep is from.

    Returns
    -------
    np.ndarray
        Combined data matrix.
    """
    combined_episodes = []
    for (i, X) in episodes:
        if episode_feature:
            combined_episodes.append(
                np.hstack((i * np.ones((X.shape[0], 1)), X)))
        else:
            combined_episodes.append(X)
    # Concatenate the combined episodes
    Xc = np.vstack(combined_episodes)
    return Xc


def _weights_from_data_matrix(
    X: np.ndarray,
    n_steps: int = None,
    discount_factor: float = 1,
    episode_feature: bool = False,
) -> np.ndarray:
    """Create an array of scoring weights from a data matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix
    n_steps : int
        Number of steps ahead to predict. If ``None`` or longer than the
        episode, will weight the entire episode.
    discount_factor : float
        Discount factor used to weight the error timeseries. Should be
        positive, with magnitude 1 or slightly less. The error at each
        timestep is weighted by ``discount_factor**k``, where ``k`` is the
        timestep.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    np.ndarray
        Array of weights use for scoring.

    Raises
    ------
    ValueError
        If ``discount_factor`` is not in [0, 1].
    """
    # Check discount factor
    if (discount_factor < 0) or (discount_factor > 1):
        raise ValueError('`discount_factor` must be positive and less '
                         'than one.')
    weights_list = []
    episodes = split_episodes(X, episode_feature=episode_feature)
    for i, X_i in episodes:
        # Compute number of nonzero weights needed
        n_samples_i = X_i.shape[0]
        if n_steps is None:
            n_nonzero_weights_i = n_samples_i
        else:
            n_nonzero_weights_i = min(n_steps, n_samples_i)
        # Compute weights. Weights after ``n_steps`` are 0.
        weights_i = np.array(
            [discount_factor**k for k in range(n_nonzero_weights_i)]
            + [0] * (n_samples_i - n_nonzero_weights_i))
        weights_list.append(weights_i)
    weights = np.concatenate(weights_list)
    return weights
