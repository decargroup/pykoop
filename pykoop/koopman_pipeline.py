"""Koopman pipeline meta-estimators and related interfaces.

Since the Koopman regression problem operates on timeseries data, it has
additional requirements that preclude the use of ``scikit-learn``
:class:`Pipeline` objects:

1. The original state must be kept at the beginning of the lifted state.
2. The input-dependent lifted states must be kept at the end of the lifted
   state.
3. The number of input-independent and input-dependent lifting functions must
   be tracked throughout the pipeline.
4. Samples must not be reordered or subsampled (this would corrupt delay-based
   lifting functions).
5. Concatenated data from different training epidodes must not be mixed (even
   though the states are adjacent in the array, they may not be sequential in
   time).

To meet these requirements, each lifting function, described by the
:class:`KoopmanLiftingFn` interface, supports a feature that indicates which
episode each sample belongs to. Furthermore, each lifting function stores the
number of input-dependent and input-independent features at its input and
output.

The data matrices provided to :func:`fit` (as well as :func:`transform`
and :func:`inverse_transform`) must obey the following format:

1. If ``episode_feature`` is true, the first feature must indicate
   which episode each timestep belongs to.
2. The last ``n_inputs`` features must be exogenous inputs.
3. The remaining features are considered to be states (input-independent).

Consider an example data matrix where the :func:`fit` parameters are
``episode_feature=True`` and ``n_inputs=1``:

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

In the above matrix, there are three distinct episodes with different
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

Koopman regressors, which implement the interface defined in
:class:`KoopmanRegressor` are distinct from ``scikit-learn`` regressors in that
they support the episode feature and state tracking attributes used by the
lifting function objects. Koopman regressors also support being fit with a
single data matrix, which they will split and time-shift according to the
episode feature.
"""

import abc
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas
import sklearn.base
import sklearn.metrics

from ._sklearn_metaestimators import metaestimators


class KoopmanLiftingFn(sklearn.base.BaseEstimator,
                       sklearn.base.TransformerMixin,
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
        X_pred = combine_episodes(predictions,
                                  episode_feature=self.episode_feature_)
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
    to make this work with :class:``DelayLiftingFn``, especially when you can
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
                      sklearn.base.TransformerMixin):
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
        X = sklearn.utils.validation.check_array(X,
                                                 ensure_min_samples=2,
                                                 **self._check_array_params)
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

    @staticmethod
    def make_scorer(
        n_steps: int = None,
        discount_factor: float = 1,
        regression_metric: str = 'neg_mean_squared_error',
        multistep: bool = True
    ) -> Callable[['KoopmanPipeline', np.ndarray, Optional[np.ndarray]],
                  float]:
        """Make a Koopman pipeline scorer.

        A ``scikit-learn`` scorer accepts the parameters ``(estimator, X, y)``
        and returns a float representing the prediction quality of
        ``estimator`` on ``X`` with reference to ``y``. Higher numbers are
        better. Losses are negated [scorers]_.

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
            Regression metric to use. One of ['explained_variance',
            'neg_mean_absolute_error', 'neg_mean_squared_error',
            'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2',
            'neg_mean_absolute_percentage_error']. See [scorers]_.
        multistep : bool
            If true, predict using :func:`predict_multistep`. Otherwise,
            predict using :func:`predict` (one-step-ahead prediction).
            Multistep prediction is highly recommended unless debugging. If
            one-step-ahead prediciton is used, `n_steps` and `discount_factor`
            are ignored.

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
        .. [scorers] https://scikit-learn.org/stable/modules/model_evaluation.html  # noqa: E501
        """
        # Check discount factor
        if (discount_factor < 0) or (discount_factor > 1):
            raise ValueError('`discount_factor` must be positive and less '
                             'than one.')

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

        def koopman_pipeline_scorer(estimator: KoopmanPipeline,
                                    X: np.ndarray,
                                    y: np.ndarray = None) -> float:
            # Shift episodes
            X_unshifted, X_shifted = shift_episodes(
                X,
                n_inputs=estimator.n_inputs_in_,
                episode_feature=estimator.episode_feature_)
            # Predict
            if multistep:
                X_predicted = estimator.predict_multistep(X_unshifted)
            else:
                X_predicted = estimator.predict(X_unshifted)
            # Strip episode feature and initial conditions
            X_shifted = strip_initial_conditions(X_shifted,
                                                 estimator.min_samples_,
                                                 estimator.episode_feature_)
            X_predicted = strip_initial_conditions(X_predicted,
                                                   estimator.min_samples_,
                                                   estimator.episode_feature_)
            # Strip episode feature if present
            if estimator.episode_feature_:
                X_shifted = X_shifted[:, 1:]
                X_predicted = X_predicted[:, 1:]
            # Compute weights
            weights: Optional[np.ndarray]
            if multistep:
                # Compute number of weights needed
                n_samples = X_shifted.shape[0]
                if (n_steps is None) or (n_steps > n_samples):
                    n_weights = n_samples
                else:
                    n_weights = n_steps
                # Compute weights. Weights after ``n_steps`` are 0.
                weights = np.array(
                    [discount_factor**k for k in range(n_weights)]
                    + [0] * (n_samples - n_weights))
            else:
                weights = None
            # Calculate score
            score = regression_metrics[regression_metric](
                X_shifted,
                X_predicted,
                sample_weight=weights,
                multioutput='uniform_average',
            )
            # Invert losses
            if regression_metric not in greater_is_better:
                score *= -1
            return score

        return koopman_pipeline_scorer

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        # noqa: D102
        return self._get_params('lifting_functions', deep=deep)

    def set_params(self, **kwargs) -> 'KoopmanPipeline':
        # noqa: D102
        self._set_params('lifting_functions', **kwargs)
        return self


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
        episode_feature: bool = False) -> List[Tuple[int, np.ndarray]]:
    """Split a data matrix into episodes.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    List[Tuple[int, np.ndarray]]
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


def combine_episodes(episodes: List[Tuple[int, np.ndarray]],
                     episode_feature: bool = False) -> np.ndarray:
    """Combine episodes into a data matrix.

    Parameters
    ----------
    episodes : List[Tuple[int, np.ndarray]]
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
