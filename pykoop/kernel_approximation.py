"""Kernel approximations for corresponding lifting functions."""

import abc
from typing import Union

import numpy as np
import scipy.stats
import sklearn.base
import sklearn.utils


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
        Number of features output. This attribute is not available in
        estimators from :mod:`sklearn.kernel_approximation`.
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
    n_features_out_ : int
        Number of features output. This attribute is not available in
        estimators from :mod:`sklearn.kernel_approximation`.
    ift_ : scipy.stats.rv_continuous
        Probability distribution corresponding to inverse Fourier transform of
        chosen kernel.
    random_weights_ : np.ndarray, shape (n_feature_in_, n_components)
        Random weights to inner-product with features.
    random_offsets_ : np.ndarray, shape (n_features_in_, )
        Random offsets if ``method`` is ``'weight_offset'``.
    """

    # Laplacian and Cauchy being swapped is not a typo. They are Fourier
    # transforms of each other.
    _ift_lookup = {
        'gaussian': scipy.stats.norm,
        'laplacian': scipy.stats.cauchy,
        'cauchy': scipy.stats.laplace,
    }

    def __init__(
        self,
        kernel_or_ift: Union[str, scipy.stats.rv_continuous] = 'gaussian',
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
                  transform ``scipy.stats.norm`` (default),
                - ``'laplacian'`` -- Laplacian kernel, with inverse Fourier
                  transform ``scipy.stats.cauchy``, or
                - ``'cauchy'`` -- Cauchy kernel, with inverse Fourier transform
                  ``scipy.stats.laplace``.

            Alternatively, a positive, shift-invariant kernel can be implicitly
            specified by providing a univariate probability distribution
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
        X = sklearn.utils.validation.check_array(X)
        # Set inverse Fourier transform
        if isinstance(self.kernel_or_ift, str):
            self.ift_ = self._ift_lookup[self.kernel_or_ift]
        else:
            self.ift_ = self.kernel_or_ift
        # Validate input
        if self.n_components <= 0:
            raise ValueError('`n_components` must be positive.')
        if self.shape <= 0:
            raise ValueError('`shape` must be positive.')
        valid_methods = ['weight_offset', 'weight_only']
        if self.method not in valid_methods:
            raise ValueError(f'`method` must be one of {valid_methods}.')
        # Set number of input and output features
        self.n_features_in_ = X.shape[1]
        if self.method == 'weight_only':
            self.n_features_out_ = 2 * self.n_components
        else:
            self.n_features_out_ = self.n_components
        # Generate random weights
        self.random_weights_ = self.ift_.rvs(
            scale=self.shape,  # TODO THIS MIGHT BE WRONG
            size=(self.n_features_in_, self.n_components),
            random_state=self.random_state,
        )
        # Generate random offsets if needed
        if self.method == 'weight_only':
            self.random_offsets_ = None
        else:
            # Range is [loc, loc + scale]
            self.random_offsets_ = scipy.stats.uniform.rvs(
                loc=0,
                scale=(2 * np.pi),
                size=self.n_components,
                random_state=self.random_state,
            )
        # Easiest way to make sure distribution is univariate is to check the
        # dimension of the output.
        if self.random_weights_.ndim != 2:
            raise ValueError('`kernel_or_ift` must specify a univariate '
                             'probability distribution.')
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
        sklearn.utils.validation.check_is_fitted(self)
        X = sklearn.utils.validation.check_array(X)
        products = X @ self.random_weights_  # (n_samples, n_components)
        if self.method == 'weight_only':
            Xt_unscaled = np.hstack((
                np.cos(products),
                np.sin(products),
            ))
        else:
            Xt_unscaled = np.sqrt(2) * np.cos(products + self.random_offsets_)
        Xt = np.sqrt(1 / self.n_components) * Xt_unscaled
        return Xt


class RandomBinningKernelApprox(KernelApproximation):
    """Kernel approximation with random binning.

    Attributes
    ----------
    n_features_in_ : int
        Number of features input.
    n_features_out_ : int
        Number of features output. This attribute is not available in
        estimators from :mod:`sklearn.kernel_approximation`.
    """

    def __init__(
        self,
        n_components: int = 100,
        random_state: Union[int, np.random.RandomState] = None,
    ) -> None:
        """Instantiate :class:`RandomBinningKernelApprox`.

        Parameters
        ----------
        n_components : int
            Number of random samples used to generate features. The higher the
            number of components, the higher the number of features. Since
            unoccupied bins are eliminated, it's impossible to know the exact
            number of features before fitting.
        random_state : Union[int, np.random.RandomState]
            Random seed.
        """
        self.n_components = n_components
        self.random_state = random_state

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
