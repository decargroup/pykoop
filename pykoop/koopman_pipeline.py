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
        estimator: tuple[str, sklearn.base.RegressorMixin] = None,
    ) -> None:
        """Constructor for :class:`KoopmanPipeline`.

        While both ``preprocessors`` and ``lifting_functions`` contain
        :class:`LiftingFn` objects, their purposes differ. Lifting functions
        are inverted in :func:`inverse_transform`, while preprocessors are
        applied once and not inverted.

        Parameters
        ----------
        preprocessors : list[tuple[str, LiftingFn]]
            List of tuples containing preprocessor objects and their names.
        lifting_functions : list[tuple[str, LiftingFn]]
            List of tuples containing lifting function objects and their names.
        estimator : tuple[str, sklearn.base.RegressorMixin]
            Tuple containing a regressor object and its name.
        """
        self.preprocessors = preprocessors
        self.lifting_functions = lifting_functions
        self.estimator = estimator

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
        self.fit_transformers(X,
                              n_inputs=n_inputs,
                              episode_feature=episode_feature)

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
        self._validate_parameters()
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
        raise NotImplementedError()

    def _validate_parameters(self) -> None:
        """Validate parameters passed in constructor.

        Raises
        ------
        ValueError
            If constructor parameters are incorrect.
        """
        pass  # No constructor parameters need validation.
