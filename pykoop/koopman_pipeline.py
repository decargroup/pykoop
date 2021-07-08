"""Koopman pipeline meta-estimator and related interfaces.

Since the Koopman regression problem operates on timeseries data, it has
additional requirements that preclude the use of ``scikit-learn`` pipelines:

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

All Koopman lifting functions and preprocessors are ancestors of
:class:`KoopmanLiftingFn`. However, they differ slightly in their indended
usage.  When predicting using a Koopman pipeline, lifting functions are applied
and inverted. Preprocessors are applied at the beginning of the pipeline but
never inverted.

For example, preprocessing angles by replacing them with ``cos`` and ``sin`` of
their values is typically not inverted, since it's more convenient to work with
``cos`` and ``sin`` when scoring and cross-validating.

Koopman regressors, which implement the interface defined in
:class:`KoopmanRegressor` are distinct from ``scikit-learn`` regressors in that
they support the episode feature and state tracking attributes used by the
lifting function objects. Koopman regressors also support being fit with a
single data matrix, which they will split and time-shift according to the
episode feature.
"""

import abc
from typing import Optional
from collections.abc import Callable

import numpy as np
import pandas
import sklearn.base
import sklearn.metrics


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
    def _fit_one_ep(self, X: np.ndarray) -> tuple[int, int]:
        """Fit lifting function using a single episode.

        Expects and returns data without an episode header. Data is assumed to
        belong to a single episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        tuple[int, int]
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
        episodes = _split_episodes(X, episode_feature=self.episode_feature_)
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

    @abc.abstractmethod
    def n_samples_in(self, n_samples_out: int = 1) -> int:
        raise NotImplementedError()

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
        episodes = _split_episodes(X, episode_feature=self.episode_feature_)
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
        Xt = _combine_episodes(transformed_episodes,
                               episode_feature=self.episode_feature_)
        return Xt

    @abc.abstractmethod
    def _fit_one_ep(self, X: np.ndarray) -> tuple[int, int, int]:
        """Fit lifting function using a single episode.

        Expects and returns data without an episode header. Data is assumed to
        belong to a single episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        tuple[int, int, int]
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
    _check_X_y_params = {
        'multi_output': True,
        'y_numeric': True,
    }

    # Array check parameters for :func:`predict` and :func:`fit` when only
    # ``X`` is given
    _check_array_params = {
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
        # Compute fit attributes
        self.n_features_in_ = X.shape[1]
        self.n_inputs_in_ = n_inputs
        self.n_states_in_ = (X.shape[1] - n_inputs -
                             (1 if episode_feature else 0))
        self.episode_feature_ = episode_feature
        # Split ``X`` if needed
        if y is None:
            X_unshifted, X_shifted = _shift_episodes(
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
        episodes = _split_episodes(X, episode_feature=self.episode_feature_)
        # Predict for each episode
        predictions = []
        for (i, X_i) in episodes:
            predictions.append((i, X_i @ self.coef_))
        # Combine and return
        X_pred = _combine_episodes(predictions,
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


class KoopmanPipeline(sklearn.base.BaseEstimator):
    """Meta-estimator for chaining lifting functions with an estimator.

    Attributes
    ----------
    min_samples_ : int
        Minimum number of samples needed to use the transformer.
    """

    def __init__(
        self,
        preprocessors: list[tuple[str, KoopmanLiftingFn]] = None,
        lifting_functions: list[tuple[str, KoopmanLiftingFn]] = None,
        regressor: tuple[str, KoopmanRegressor] = None,
    ) -> None:
        """Instantiate for :class:`KoopmanPipeline`.

        While both ``preprocessors`` and ``lifting_functions`` contain
        :class:`KoopmanLiftingFn` objects, their purposes differ. Lifting
        functions are inverted in :func:`inverse_transform`, while
        preprocessors are applied once and not inverted.

        As much error checking as possible is delegated to the sub-estimators.

        Parameters
        ----------
        preprocessors : list[tuple[str, KoopmanLiftingFn]]
            List of tuples containing preprocessor objects and their names.
        lifting_functions : list[tuple[str, KoopmanLiftingFn]]
            List of tuples containing lifting function objects and their names.
        regressor : tuple[str, sklearn.base.RegressorMixin]
            Tuple containing a regressor object and its name.
        """
        self.preprocessors = preprocessors
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
        if self.regressor is None:
            raise ValueError('`regressor` must be specified to use `fit()`.')
        # Clone regressor
        self.regressor_ = (
            self.regressor[0],
            sklearn.base.clone(self.regressor[1]),
        )
        # Fit transformers and transform input
        self.fit_transformers(
            X,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        Xt = self.transform(X)
        # Fit the regressor
        self.regressor_[1].fit(
            Xt,
            n_inputs=self.n_inputs_out_,
            episode_feature=self.episode_feature_,
        )
        return self

    def fit_transformers(self,
                         X: np.ndarray,
                         y: np.ndarray = None,
                         n_inputs: int = 0,
                         episode_feature: bool = False) -> 'KoopmanPipeline':
        """Fit only the preprocessors and lifting functions in the pipeline.

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
        if self.preprocessors is None:
            self.preprocessors = []
        if self.lifting_functions is None:
            self.lifting_functions = []
        # Save state of episode feature
        self.episode_feature_ = episode_feature
        # Set number of features
        self.n_features_in_ = X.shape[1]
        self.n_states_in_ = (X.shape[1] - n_inputs -
                             (1 if episode_feature else 0))
        self.n_inputs_in_ = n_inputs
        # Clone preprocessors and lifting functions
        self.preprocessors_ = []
        for name, pp in self.preprocessors:
            self.preprocessors_.append((name, sklearn.base.clone(pp)))
        self.lifting_functions_ = []
        for name, lf in self.lifting_functions:
            self.lifting_functions_.append((name, sklearn.base.clone(lf)))
        # Fit and transform preprocessors and lifting functions
        X_out = X
        n_inputs_out = n_inputs
        for name, pp in self.preprocessors_:
            X_out = pp.fit_transform(X_out,
                                     n_inputs=n_inputs_out,
                                     episode_feature=episode_feature)
            n_inputs_out = pp.n_inputs_out_
        for name, lf in self.lifting_functions_:
            X_out = lf.fit_transform(X_out,
                                     n_inputs=n_inputs_out,
                                     episode_feature=episode_feature)
            n_inputs_out = lf.n_inputs_out_
        # Set output dimensions
        tfs = (self.preprocessors_ + self.lifting_functions_)
        if len(tfs) > 0:
            # Find the last transformer and use it to get output dimensions
            last_tf = tfs[-1][1]
            self.n_features_out_ = last_tf.n_features_out_
            self.n_states_out_ = last_tf.n_states_out_
            self.n_inputs_out_ = last_tf.n_inputs_out_
            # Compute minimum number of samples needed by transformer.
            # Each transformer knows how many input samples it needs to produce
            # a given number of output samples.  Knowing we just want one
            # sample at the output, we work backwards to figure out how many
            # samples we need at the beginning of the pipeline.
            n_samples_out = 1
            for tf in tfs[::-1]:
                n_samples_out = tf[1].n_samples_in(n_samples_out)
            self.min_samples_ = n_samples_out
        else:
            # Fall back on input dimensions
            self.n_features_out_ = self.n_features_in_
            self.n_states_out_ = self.n_states_in_
            self.n_inputs_out_ = self.n_inputs_in_
            self.min_samples_ = 1
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
        # Apply preprocessing transforms, then lifting functions
        X_out = X
        for name, pp in self.preprocessors_:
            X_out = pp.transform(X_out)
        for name, lf in self.lifting_functions_:
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
        # Apply inverse lifting functions in reverse order
        X_out = X
        for name, lf in self.lifting_functions_[::-1]:
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
        # Lift data matrix
        X_trans = self.transform(X)
        # Predict in lifted space
        X_pred = self.regressor_[1].predict(X_trans)
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
            X_pred_inv = X_pred_pad_inv[:, :self.n_states_in_]
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
        # Split episodes
        episodes = _split_episodes(X, episode_feature=self.episode_feature_)
        # Loop over episodes
        predictions = []
        for (i, X_i) in episodes:
            # Check length of episode.
            if X_i.shape[0] < self.min_samples_:
                raise ValueError(f'Episode {i} has {X_i.shape[0]} samples but '
                                 f'`min_samples_`={self.min_samples_} samples '
                                 'are required.')
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
                X_ik = _combine_episodes(
                    [(i,
                      np.hstack((
                          X_pred_i[(k - self.min_samples_):k, :],
                          X_i[(k - self.min_samples_):k, self.n_states_in_:],
                      )))],
                    episode_feature=self.episode_feature_)
                # Predict next step
                X_pred_ik = self.predict(X_ik)[[-1], :]
                # Extract data matrix from prediction
                X_pred_i[[k], :] = _split_episodes(
                    X_pred_ik, episode_feature=self.episode_feature_)[0][1]
            predictions.append((i, X_pred_i))
        # Combine episodes
        X_p = _combine_episodes(predictions,
                                episode_feature=self.episode_feature_)
        return X_p

    @staticmethod
    def make_scorer(
        n_steps: int = None,
        discount_factor: float = 1,
        regression_metric: str = 'neg_mean_squared_error',
    ) -> Callable[['KoopmanPipeline', np.ndarray, Optional[np.ndarray]],
                  float]:
        """Make a Koopman pipeline scorer.

        A ``scikit-learn`` scorer accepts the parameters ``(estimator, X, y)``
        and returns a float representing the prediction quality of
        ``estimator`` on ``X`` with reference to ``y``. Higher numbers are
        better. Losses are negated [1]_.

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
            'neg_mean_absolute_percentage_error']. See [1]_.

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
        .. [1] https://scikit-learn.org/stable/modules/model_evaluation.html
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
            X_unshifted, X_shifted = _shift_episodes(
                X,
                n_inputs=estimator.n_inputs_in_,
                episode_feature=estimator.episode_feature_)
            # Predict
            X_predicted = estimator.predict_multistep(X_unshifted)
            # Strip episode feature and initial conditions
            if estimator.episode_feature_:
                X_shifted = X_shifted[estimator.min_samples_:, 1:]
                X_predicted = X_predicted[estimator.min_samples_:, 1:]
            else:
                X_shifted = X_shifted[estimator.min_samples_:, :]
                X_predicted = X_predicted[estimator.min_samples_:, :]
            # Compute number of weights needed
            n_samples = X_shifted.shape[0]
            if (n_steps is None) or (n_steps > n_samples):
                n_weights = n_samples
            else:
                n_weights = n_steps
            # Compute weights. Weights after ``n_steps`` are 0.
            weights = np.array([discount_factor**k for k in range(n_weights)]
                               + [0] * (n_samples - n_weights))
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


def _shift_episodes(
        X: np.ndarray,
        n_inputs: int = 0,
        episode_feature: bool = False) -> tuple[np.ndarray, np.ndarray]:
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
    tuple[np.ndarray, np.ndarray]
        Tuple whose first element is the unshifted array and whose second
        element is the shifted array with its inputs truncated. Both arrays
        have the same number of samples. Their episode features are stripped if
        present.
    """
    # Split episodes
    episodes = _split_episodes(X, episode_feature=episode_feature)
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
    X_unshifted = _combine_episodes(unshifted_episodes,
                                    episode_feature=episode_feature)
    X_shifted = _combine_episodes(shifted_episodes,
                                  episode_feature=episode_feature)
    return (X_unshifted, X_shifted)


def _split_episodes(
        X: np.ndarray,
        episode_feature: bool = False) -> list[tuple[int, np.ndarray]]:
    """Split a data matrix into episodes.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    list[tuple[int, np.ndarray]]
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


def _combine_episodes(episodes: list[tuple[int, np.ndarray]],
                      episode_feature: bool = False) -> np.ndarray:
    """Combine episodes into a data matrix.

    Parameters
    ----------
    episodes : list[tuple[int, np.ndarray]]
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
