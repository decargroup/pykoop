"""Lifting functions and preprocessors for use with :class:``KoopmanPipeline``.

Koopman lifting functions have two particular requirements that distinguish
them from other ``scikit-learn`` transformers:

1. They keep the original state at the top of the lifted state.
2. They keep all input-dependent lifted states at the bottom of the lifted
   state.

All Koopman lifting functions and preprocessors are ancestors of
:class:`LiftingFn`. However, they differ slightly in their indended usage.
When predicting using a Koopman pipeline, lifting functions are applied and
inverted. Preprocessors are applied at the beginning of the pipeline but never
inverted.

For example, preprocessing angles by replacing them with ``cos`` and ``sin`` of
their values is typically not inverted, since it's more convenient to work with
``cos`` and ``sin`` when scoring and cross-validating.

Notes
-----

The data matrices provided to :func:`fit` (as well as :func:`transform`
and :func:`inverse_transform`) must obey the following format:

1. If ``episode_feature`` is true, the first feature must indicate
   which episode each timestep belongs to.
2. The last ``n_inputs`` features must be exogenous inputs.
3. The remaining features are considered to be states.

Example data matrix where ``episode_feature=True`` and ``n_inputs=1``:

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
"""

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
        Number of features before transformation, including episode feature.
        Must be set in subclass :func:`fit`.
    n_states_in_ : int
        Number of states before transformation.
        Must be set in subclass :func:`fit`.
    n_inputs_in_ : int
        Number of inputs before transformation.
        Must be set in subclass :func:`fit`.
    n_features_out_ : int
        Number of features after transformation, including episode feature.
        Must be set in subclass :func:`fit`.
    n_states_out_ : int
        Number of states after transformation.
        Must be set in subclass :func:`fit`.
    n_inputs_out_ : int
        Number of inputs after transformation.
        Must be set in subclass :func:`fit`.
    min_samples_ : int
        Minimum number of samples needed to use the transformer.
        Must be set in subclass :func:`fit`.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
        Must be set in subclass :func:`fit`.
    """

    @abc.abstractmethod
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
    def _fit_one_ep(self, X: np.ndarray) -> None:
        """Fit lifting function using a single episode.

        Expects and returns data without an episode header. Data is assumed to
        belong to a single episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
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


class EpisodeIndependentLiftingFn(LiftingFn):
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
        Must be set by subclass :func:`_fit_one_ep`.
    n_states_out_ : int
        Number of states after transformation.
        Must be set by subclass :func:`_fit_one_ep`.
    n_inputs_out_ : int
        Number of inputs after transformation.
        Must be set by subclass :func:`_fit_one_ep`.
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
        self._check_params = {
            'ensure_min_features': 2 if episode_feature else 1,
        }
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_params)
        # Set numbre of input features (including episode feature)
        self.n_features_in_ = X.shape[1]
        # Extract episode feature
        if self.episode_feature_:
            X = X[:, 1:]
        # Set states and inputs in
        self.n_inputs_in_ = n_inputs
        self.n_states_in_ = X.shape[1] - n_inputs
        # Episode independent lifting functions only need one sample.
        self.min_samples_ = 1
        self._fit_one_ep(X)
        return self

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


class EpisodeDependentLiftingFn(LiftingFn):
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
        Must be set by subclass :func:`_fit_one_ep`.
    n_states_out_ : int
        Number of states after transformation.
        Must be set by subclass :func:`_fit_one_ep`.
    n_inputs_out_ : int
        Number of inputs after transformation.
        Must be set by subclass :func:`_fit_one_ep`.
    min_samples_ : int
        Minimum number of samples needed to use the transformer.
        Set by :func:`fit`.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
        Must be set by subclass :func:`_fit_one_ep`.

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
        self._check_params = {
            'ensure_min_features': 2 if episode_feature else 1,
        }
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_params)
        # Set numbre of input features (including episode feature)
        self.n_features_in_ = X.shape[1]
        # Extract episode feature
        if self.episode_feature_:
            X_ep = X[:, 0]
            X = X[:, 1:]
        else:
            X_ep = np.zeros((X.shape[0], ))
        # Extract first episod eonly
        first_ep_idx = np.unique(X_ep)[0]
        X_first = X[X_ep == first_ep_idx, :]
        # Set states and inputs in
        self.n_inputs_in_ = n_inputs
        self.n_states_in_ = X.shape[1] - n_inputs
        self._fit_one_ep(X_first)
        return self

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
                transformed_episode = self._transform_one_ep(X_i)
            elif transform == 'inverse_transform':
                transformed_episode = self._inverse_transform_one_ep(X_i)
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
    """Preprocessor used to replace angles with their cosines and sines.

    Intended as a preprocessor to be applied once to the input, rather than a
    lifting function that is inverted after prediction.

    Attributes
    ----------
    angle_features : np.ndarray
        Indices of features that are angles.
    unwrap_inverse : bool
        Unwrap inverse by replacing absolute jumps greater than ``pi`` by
        their ``2pi`` complement.
    angles_in_ : np.ndarray
        Boolean array that indicates which input features are angles.
    lin_out_ : np.ndarray
        Boolean array that indicates which output features are linear.
    cos_out_ : np.ndarray
        Boolean array that indicates which output features are cosines.
    sin_out_ : np.ndarray
        Boolean array that indicates which output features are sines.
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

    Warnings
    --------
    The inverse of this preprocessor is not the true inverse unless the data is
    inside ``[-pi, pi]``. Any offsets of ``2pi`` are lost when ``cos`` and
    ``sin`` are applied to the angles.
    """

    def __init__(self,
                 angle_features: np.ndarray = None,
                 unwrap_inverse: bool = False) -> None:
        """Instantiate :class:`AnglePreprocessor`.

        Parameters
        ----------
        angle_features : np.ndarray
            Indices of features that are angles.
        unwrap_inverse : bool
            Unwrap inverse by replacing absolute jumps greater than ``pi`` by
            their ``2pi`` complement.
        """
        self.angle_features = angle_features
        self.unwrap_inverse = unwrap_inverse

    def _fit_one_ep(self, X: np.ndarray) -> None:
        # Compute boolean array with one entry per feature. A ``True`` value
        # indicates that the feature is an angle.
        n_states_inputs_in = self.n_states_in_ + self.n_inputs_in_
        if ((self.angle_features is None)
                or (self.angle_features.shape[0] == 0)):
            self.angles_in_ = np.zeros((n_states_inputs_in, ), dtype=bool)
        else:
            # Create an array with all ```False``.
            angles_bool = np.zeros((n_states_inputs_in, ), dtype=bool)
            # Set indicated entries to ``True``.
            angles_bool[self.angle_features] = True
            self.angles_in_ = angles_bool
        # Figure out how many linear and angular features there are.
        n_lin_states = np.sum(~self.angles_in_[:self.n_states_in_])
        n_lin_inputs = np.sum(~self.angles_in_[self.n_states_in_:])
        n_ang_states = 2 * np.sum(self.angles_in_[:self.n_states_in_])
        n_ang_inputs = 2 * np.sum(self.angles_in_[self.n_states_in_:])
        self.n_states_out_ = n_lin_states + n_ang_states
        self.n_inputs_out_ = n_lin_inputs + n_ang_inputs
        n_states_inputs_out = self.n_states_out_ + self.n_inputs_out_
        self.n_features_out_ = (n_states_inputs_out +
                                (1 if self.episode_feature_ else 0))
        # Create array for linear, cosine, and sine feature indices.
        self.lin_out_ = np.zeros((n_states_inputs_out, ), dtype=bool)
        self.cos_out_ = np.zeros((n_states_inputs_out, ), dtype=bool)
        self.sin_out_ = np.zeros((n_states_inputs_out, ), dtype=bool)
        # Figure out which features are cosines, sines, or linear.
        i = 0
        for j in range(n_states_inputs_in):
            if self.angles_in_[j]:
                self.cos_out_[i] = True
                self.sin_out_[i + 1] = True
                i += 2
            else:
                self.lin_out_[i] = True
                i += 1

    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Create blank array
        n_states_inputs_out = self.n_states_out_ + self.n_inputs_out_
        Xt = np.zeros((X.shape[0], n_states_inputs_out))
        # Apply cos and sin to appropriate features.
        Xt[:, self.lin_out_] = X[:, ~self.angles_in_]
        Xt[:, self.cos_out_] = np.cos(X[:, self.angles_in_])
        Xt[:, self.sin_out_] = np.sin(X[:, self.angles_in_])
        return Xt

    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Create blank array
        n_states_inputs_in = self.n_states_in_ + self.n_inputs_in_
        Xt = np.zeros((X.shape[0], n_states_inputs_in))
        # Put linear features back where they belong
        Xt[:, ~self.angles_in_] = X[:, self.lin_out_]
        # Invert transform for angles. Unwrap if necessary.
        angle_values = np.arctan2(X[:, self.sin_out_], X[:, self.cos_out_])
        if self.unwrap_inverse:
            Xt[:, self.angles_in_] = np.unwrap(angle_values, axis=0)
        else:
            Xt[:, self.angles_in_] = angle_values
        return Xt

    def _validate_parameters(self) -> None:
        pass  # No constructor parameters need validation.


class PolynomialLiftingFn(EpisodeIndependentLiftingFn):
    """Lifting function to generate all monomials of the input features.

    Attributes
    ----------
    order : int
        Order of monomials to generate
    interaction_only : bool
        If ``True``, skip powers of the same feature.
    transformer_ : sklearn.preprocessing.PolynomialFeatures
        Internal transformer generating the polynomial features.
    transform_order_ : np.ndarray
        Indices indicating reordering needed after applying ``transformer_``.
    inverse_transform_order_ : np.ndarray
        Indices indicating reverse ordering needed to reverse
        ``transform_order_``.
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

    def __init__(self, order: int = 1, interaction_only: bool = False) -> None:
        """Instantiate :class:`PolynomialLiftingFn`.

        Parameters
        ----------
        order : int
            Order of monomials to generate
        interaction_only : bool
            If ``True``, skip powers of the same feature.
        """
        self.order = order
        self.interaction_only = interaction_only

    def _fit_one_ep(self, X: np.ndarray) -> None:
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
            # Original states and inputs will have a ``power_`` that's a
            # row of the identity matrix, since they are mononomials of the
            # form ``x_1^1 * x_2^0 * x_3^0``. In that case, the row would be
            # ``[1, 0, 0]``.
            index = np.nonzero(
                np.all(self.transformer_.powers_ == eye[i, :], axis=1))
            # Split up the "original states" from the "original inputs"
            if i < self.n_states_in_:
                orig_states.append(index)
            else:
                orig_inputs.append(index)
        original_states = np.ravel(orig_states).astype(int)
        original_input = np.ravel(orig_inputs).astype(int)
        # Figure out which other lifted states contain inputs. These ones will
        # go at the bottom of the output.
        # Find all input-dependent features:
        all_inputs = np.nonzero(
            np.any(self.transformer_.powers_[:, self.n_states_in_:] != 0,
                   axis=1))[0].astype(int)
        # Do a set difference to remove the unlifted input features:
        other_inputs = np.setdiff1d(all_inputs, original_input)
        # Figure out which other lifted states contain states (but are not the
        # original states themselves. Accomplish this by subtracting off
        # all the other types of features we found. That is, if it's not in
        # ``original_state_features``, ``original_input_features``, or
        # ``other_input_features``, it must be in ``other_state_features``.
        other_states = np.setdiff1d(
            np.arange(self.transformer_.powers_.shape[0]),
            np.union1d(np.union1d(original_states, original_input),
                       other_inputs),
        ).astype(int)
        # Form new ordering of states. Input-dependent states go at the end.
        self.transform_order_ = np.concatenate(
            (original_states, other_states, original_input, other_inputs))
        # Compute how many input-independent lifted states and input-dependent
        # lifted states there are
        self.n_states_out_ = (original_states.shape[0] + other_states.shape[0])
        self.n_inputs_out_ = (original_input.shape[0] + other_inputs.shape[0])
        self.n_features_out_ = (self.n_states_out_ + self.n_inputs_out_ +
                                (1 if self.episode_feature_ else 0))
        # Figure out original order of features
        self.inverse_transform_order_ = np.concatenate((
            np.arange(0, self.n_states_in_),
            np.arange(self.n_states_out_,
                      (self.n_states_out_ + self.n_inputs_in_)),
        ))

    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Transform states
        Xt = self.transformer_.transform(X)
        # Reorder states
        return Xt[:, self.transform_order_]

    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Extract the original features from the lifted features
        return X[:, self.inverse_transform_order_]

    def _validate_parameters(self) -> None:
        if self.order <= 0:
            raise ValueError('`order` must be greater than or equal to 1.')


class DelayLiftingFn(EpisodeDependentLiftingFn):
    """Lifting function to generate delay coordinates for state and input.

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
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.

    Warnings
    --------
    :func:`transform` and :func:`inverse_transform` are not exact inverses
    unless ``n_delays_x`` and ``n_delays_u`` are the same. Only the last
    samples will be the same, since the ``abs(n_delays_x - n_delays_u)``
    earliest samples will need to be dropped to ensure the output array is
    rectangular.
    """

    def __init__(self,
                 n_delays_state: int = 0,
                 n_delays_input: int = 0) -> None:
        """Instantiate :class:`DelayLiftingFn`.

        Parameters
        ----------
        n_delays_state : int
            Number of delays to apply to the state.
        n_delays_input : int
            Number of delays to apply to the input.
        """
        self.n_delays_state = n_delays_state
        self.n_delays_input = n_delays_input

    def _fit_one_ep(self, X: np.ndarray) -> None:
        # Compute number of states and inputs that will be output.
        self.n_states_out_ = self.n_states_in_ * (self.n_delays_state + 1)
        self.n_inputs_out_ = self.n_inputs_in_ * (self.n_delays_input + 1)
        self.n_features_out_ = (self.n_states_out_ + self.n_inputs_out_ +
                                (1 if self.episode_feature_ else 0))
        # Compute the minimum number of samples needed to use this transformer.
        self.n_samples_needed_ = max(self.n_delays_state,
                                     self.n_delays_input) + 1

    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Separate states and inputs
        X_x = X[:, :self.n_states_in_]
        X_u = X[:, self.n_states_in_:]
        # Delay states and inputs separately
        Xd_x = DelayLiftingFn._delay(X_x, self.n_delays_state)
        Xd_u = DelayLiftingFn._delay(X_u, self.n_delays_input)
        # ``Xd_x`` and ``Xd_u`` have different numbers of samples at this
        # point. Truncate the larger one to the same shape.
        n_samples = min(Xd_x.shape[0], Xd_u.shape[0])
        Xd = np.hstack((Xd_x[-n_samples:, :], Xd_u[-n_samples:, :]))
        return Xd

    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Separate states and inputs.
        X_x = X[:, :self.n_states_out_]
        X_u = X[:, self.n_states_out_:]
        # Undelay states and inputs separately.
        Xu_x = DelayLiftingFn._undelay(X_x, self.n_delays_state)
        Xu_u = DelayLiftingFn._undelay(X_u, self.n_delays_input)
        # ``Xu_x`` and ``Xu_u`` have different numbers of samples at this
        # point. Truncate the larger one to the same shape.
        n_samples = min(Xu_x.shape[0], Xu_u.shape[0])
        Xu = np.hstack((Xu_x[-n_samples:, :], Xu_u[-n_samples:, :]))
        return Xu

    def _validate_parameters(self):
        if self.n_delays_state < 0:
            raise ValueError(
                '`n_delays_x` must be greater than or equal to 0.')
        if self.n_delays_input < 0:
            raise ValueError(
                '`n_delays_u` must be greater than or equal to 0.')

    @staticmethod
    def _delay(X: np.ndarray, n_delays: int) -> np.ndarray:
        """Delay an array by a specific number of samples.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        n_delays : int
            Number of delays.

        Returns
        -------
        np.ndarray
            Delayed matrix.
        """
        n_samples_out = X.shape[0] - n_delays
        # Go through matrix with moving window and treat each window as a new
        # set of features.
        delays = []
        for i in range(n_delays, -1, -1):
            delays.append(X[i:(n_samples_out + i), :])
        Xd = np.concatenate(delays, axis=1)
        return Xd

    @staticmethod
    def _undelay(X: np.ndarray, n_delays: int) -> np.ndarray:
        """Undelay an array by a specific number of samples.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        n_delays : int
            Number of delays.

        Returns
        -------
        np.ndarray
            Undelayed matrix.

        Notes
        -----
        To understand how this implementation works, consider a matrix::

            [1 -1]
            [2 -2]
            [3 -3]
            [4 -4]

        and its delayed-by-one version::

            [2 -2 1 -1]
            [3 -3 2 -2]
            [4 -4 3 -3]

        The last two features contain the first three timesteps of the original
        matrix::

            [1 -1]
            [2 -2]
            [3 -3]

        The last timestep contains the last two timesteps of the original
        matrix::

            [4 -4 3 -3]

        By merging these two blocks of the matrix (omitting the last timestep
        of the last two features to avoid duplication), we can reconstruct the
        original matrix.
        """
        # Compute number of features
        n_features = X.shape[1] // (n_delays + 1)
        # Get the first part of the sequence from the last features (omitting
        # the last sample to avoid duplication).
        Xu_1 = [X[:-1, -n_features:]]
        # Get the rest of the sequence from the last sample
        Xu_2 = np.split(X[[-1], :], n_delays + 1, axis=1)[::-1]
        # Put the sequence together
        Xu = np.vstack(Xu_1 + Xu_2)
        return Xu
