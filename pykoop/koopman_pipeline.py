import numpy as np
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
        # Split into unshifted and shifted data matrices
        Xt_unshifted, Xt_shifted = shift_episodes(
            Xt,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        # Fit regressor
        # TODO IF EPISODE FEATURES ARENT IN FIT, THEY CANT BE IN PREDICT...
        self.regressor_[1].fit(Xt_unshifted, Xt_shifted)
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
            X_pred_pad = np.hstack((
                X_pred,
                np.zeros((X_pred.shape[1], self.n_inputs_out_))
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
    # Extract episode feature
    if episode_feature:
        X_ep = X[:, 0]
        X = X[:, 1:]
    else:
        X_ep = np.zeros((X.shape[0], ))
    # Split X into list of episodes. Each episode is a tuple containing
    # its index and its associated data matrix.
    episodes = []
    for i in np.unique(X_ep):
        episodes.append((i, X[X_ep == i, :]))
    # Shift each episode
    X_unshifted = []
    X_shifted = []
    for _, ep in episodes:
        X_unshifted.append(ep[:-1, :])
        # Strip input if present
        if n_inputs == 0:
            X_shifted.append(ep[1:, :])
        else:
            X_shifted.append(ep[1:, :-n_inputs])
    # Recombine and return
    X_unshifted_np = np.vstack(X_unshifted)
    X_shifted_np = np.vstack(X_shifted)
    return (X_unshifted_np, X_shifted_np)
