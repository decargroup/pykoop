"""Kernel approximations for corresponding lifting functions."""

import abc

import sklearn.base


class KernelApproximation(
        sklearn.base.BaseEstimator,
        sklearn.base.TransformerMixin,
        metaclass=abc.ABCMeta,
):
    """Base class for all kernel approximations.

    All attributes with a trailing underscore must be set in the subclass'
    :func:`fit`.

    Attributes
    ----------
    n_features_in_ : int
        Number of features input.
    n_features_out_ : int
        Number of features output (i.e., number of components).
    """

    @abc.abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
    ) -> 'KernelApproximation':
        """Fit kernel approximation.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.

        Returns
        -------
        KernelApproximation
            Instance of itself.

        Raises
        ------
        ValueError
            If any of the constructor parameters are incorrect.
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


class RandomFourierKernalApprox(KernelApproximation):
    """Kernel approximation with random Fourier features.

    Attributes
    ----------
    n_features_in_ : int
        Number of features input.
    n_features_out_ : int
        Number of features output (i.e., number of components).
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
    ) -> 'RandomFourierKernelApprox':
        """Fit kernel approximation.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.

        Returns
        -------
        RandomFourierKernelApprox
            Instance of itself.

        Raises
        ------
        ValueError
            If any of the constructor parameters are incorrect.
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


class RandomBinningKernelApprox(KernelApproximation):
    """Kernel approximation with random binning.

    Attributes
    ----------
    n_features_in_ : int
        Number of features input.
    n_features_out_ : int
        Number of features output (i.e., number of components).
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
    ) -> 'RandomBinningKernelApprox':
        """Fit kernel approximation.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.

        Returns
        -------
        RandomBinningKernelApprox
            Instance of itself.

        Raises
        ------
        ValueError
            If any of the constructor parameters are incorrect.
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
