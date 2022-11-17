"""Kernel approximations for corresponding lifting functions."""

import abc

import scipy.stats
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


class RandomFourierKernelApprox(KernelApproximation):
    """Kernel approximation with random Fourier features.

    Attributes
    ----------
    n_features_in_ : int
        Number of features input.
    ift_ : scipy.stats.rv_continuous
        Probability distribution corresponding to inverse Fourier transform of
        chosen kernel.
    """

    def __init__(
        self,
        kernel_or_ift: Union[str, scipy.stats.rv_continuous] = None,
        n_components: int = 100,
        shape: float = 1,
        method: str = 'weight_offset',
        random_state: Union[int, np.random.RandomState] = None,
    ) -> None:
        """Instantiate :class:`RandomFourierKernelApprox`.

        Parameters
        ----------
        kernel_or_ift : Union[str, scipy.stats.rv_continuous]
            Kernel to approximate. Possible options are

                - ``'gaussian'`` -- Gaussian kernel, with inverse Fourier
                  transform ``scipy.stats.norm``,
                - ``'laplacian'`` -- Laplacian kernel, with inverse Fourier
                  transform ``scipy.stats.cauchy``, or
                - ``'Cauchy'`` -- Cauchy kernel, with inverse Fourier transform
                  ``scipy.stats.laplace``.

            Alternatively, a positive, shift-invariant kernel can be implicitly
            specified by providing a univariate probability distrivution
            subclassing ``scipy.stats.rv_continuous``.

        n_components : int
            Number of random samples used to generate features. If
            ``method='weight_offset'``, this corresponds directly to the number
            of features generated. If ``method='weight_only'``, then
            ``2 * n_components`` features are generated.

        shape : float
            Shape parameter. Must be greater than zero. Larger numbers
            correspond to "sharper" kernels. Default is ``1``.

        method : str
            Feature generation method to use. If ``'weight_offset'`` (default),
            each weight corresponds to a feature
            ``cos(weight.T @ x + offset)``. If ``'weight_only'``, then each
            weight corresponds to two features, ``cos(weight.T @ x)`` and
            ``sin(weight.T @ x)``, meaning the number of features generated is
            ``2 * n_components``.

        random_state : Union[int, np.random.RandomState]
            Random seed.
        """
        self.kernel_or_ift = kernel_or_ift
        self.n_components = n_components
        self.shape = shape
        self.method = method
        self.random_state = random_state

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
        # TODO VALIDATE INPUT
        self.n_features_in_ = X.shape[1]
        self.random_weights_ = self.ift_.rvs(
            scale=self.shape,
            size=(self.n_features_in_, self.n_components),
        )
        # TODO MAKE SURE RIGHT SIZE, DONT ALLOW MULTIVARIATE
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
        raise NotImplementedError()


class RandomBinningKernelApprox(KernelApproximation):
    """Kernel approximation with random binning.

    Attributes
    ----------
    n_features_in_ : int
        Number of features input.
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
