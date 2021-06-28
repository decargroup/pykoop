import abc

import numpy as np
import pandas
import sklearn.base
import sklearn.metrics

from .lifting_functions import LiftingFn  # For type hints


class KoopmanPipeline(sklearn.base.BaseEstimator):
    """Meta-estimator for chaining lifting functions with an estimator."""

    def __init__(
        self,
        preprocessors: list[tuple[str, LiftingFn]] = None,
        lifting_functions: list[tuple[str, LiftingFn]] = None,
        regressor: tuple[str, sklearn.base.RegressorMixin] = None,
    ) -> None:
        """Instantiate for :class:`KoopmanPipeline`.

        While both ``preprocessors`` and ``lifting_functions`` contain
        :class:`LiftingFn` objects, their purposes differ. Lifting functions
        are inverted in :func:`inverse_transform`, while preprocessors are
        applied once and not inverted.

        As much error checking as possible is delegated to the sub-estimators.

        Parameters
        ----------
        preprocessors : list[tuple[str, LiftingFn]]
            List of tuples containing preprocessor objects and their names.
        lifting_functions : list[tuple[str, LiftingFn]]
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
            n_inputs=n_inputs,
            episode_feature=episode_feature,
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
        try:
            # Find the last transformer and use it to get output dimensions
            last_tf = (self.preprocessors_ + self.lifting_functions_)[-1][1]
            self.n_features_out_ = last_tf.n_features_out_
            self.n_states_out_ = last_tf.n_states_out_
            self.n_inputs_out_ = last_tf.n_inputs_out_
        except IndexError:
            # Fall back on input dimensions
            self.n_features_out_ = self.n_features_in_
            self.n_states_out_ = self.n_states_in_
            self.n_inputs_out_ = self.n_inputs_in_
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
        """Predict next states from data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Predicted data matrix.
        """
        # TODO HOW TO HANDLE EPISODE FEATURE?
        # TODO ARE EPISODES CORRECTLY CONSIDERED???
        # TODO THIS IS SINGLE STEP PREDICTION. HOW ABOUT MULTI?
        # Lift data matrix
        X_trans = self.transform(X)
        # Predict in lifted space
        X_pred = self.regressor_[1].predict(X_trans)
        # Pad inputs wth zeros to do inverse
        if self.n_inputs_out_ != 0:
            X_pred_pad = np.hstack(
                (X_pred, np.zeros((X_pred.shape[1], self.n_inputs_out_))))
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


class KoopmanRegressor(sklearn.base.BaseEstimator,
                       sklearn.base.RegressorMixin,
                       metaclass=abc.ABCMeta):
    """Base class for Koopman regressors.

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
        Fit coefficient matrix
    """

    # Array check parameters for :func:`fit` when ``X`` and ``y` are given
    _check_X_y_params = {
        'multi_output': True,
        'y_numeric': True,
    }

    # Array check parameters for :func:`predict` and :func:`fit` when only
    # ``X`` is given
    _check_array_params = {}

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
        # Split ``X`` if needed
        if y is None:
            X = sklearn.utils.validation.check_array(
                X, **self._check_array_params)
            X_unshifted, X_shifted = shift_episodes(
                X, n_inputs=n_inputs, episode_feature=episode_feature)
        else:
            X, y = sklearn.utils.validation.check_X_y(X, y,
                                                      **self._check_X_y_params)
            X_unshifted = X
            X_shifted = y
        # Compute fit attributes
        self.n_features_in_ = X.shape[1]
        self.n_inputs_in_ = n_inputs
        self.n_states_in_ = (X.shape[1] - n_inputs -
                             (1 if episode_feature else 0))
        self.episode_feature_ = episode_feature
        # Strip episode feature if present
        if episode_feature:
            X_unshifted_noep = X_unshifted[:, 1:]
            X_shifted_noep = X_shifted[:, 1:]
        else:
            X_unshifted_noep = X_unshifted
            X_shifted_noep = X_shifted
        # Call fit from subclass
        self._fit_regressor(X_unshifted_noep, X_shifted_noep)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next states from data.

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
                       X_shifted: np.ndarray) -> None:
        """Fit the regressor using shifted and unshifted data matrices.

        The input data matrices must not have episode features.

        Parameters
        ----------
        X_unshifted : np.ndarray
            Unshifted data matrix without episode feature.
        X_shifted : np.ndarray
            Shifted data matrix without episode feature.
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


def shift_episodes(
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


def combine_episodes(episodes: list[tuple[int, np.ndarray]],
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
