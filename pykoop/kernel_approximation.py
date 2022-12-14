"""Kernel approximations for corresponding lifting functions."""

import abc
import logging
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import scipy.stats
import sklearn.base
import sklearn.preprocessing
import sklearn.utils

# Create logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


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

    Similar to :class:`sklearn.kernel_approximation.RBFSampler`, but supports
    kernels other than just the Gaussian.

    For details, see [RR07]_.

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
    random_weights_ : np.ndarray, shape (n_features, n_components)
        Random weights to inner-product with features.
    random_offsets_ : np.ndarray, shape (n_features, )
        Random offsets if ``method`` is ``'weight_offset'``.

    Examples
    --------
    Generate random Fourier features from a Gaussian kernel

    >>> ka = pykoop.RandomFourierKernelApprox(
    ...     kernel_or_ift='gaussian',
    ...     n_components=10,
    ...     shape=1,
    ...     random_state=1234,
    ... )
    >>> ka.fit(X_msd[:, 1:])  # Remove episode feature
    RandomFourierKernelApprox(n_components=10, random_state=1234)
    >>> ka.transform(X_msd[:, 1:])
    array([...])
    """

    _ift_lookup = {
        'gaussian': scipy.stats.norm,
        'laplacian': scipy.stats.cauchy,
        'cauchy': scipy.stats.laplace,
    }
    """Lookup table for inverse Fourier transform of kernel.

    Laplacian and Cauchy being swapped is not a typo. They are Fourier
    transforms of each other.
    """

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
                  transform :class:`scipy.stats.norm` (default),
                - ``'laplacian'`` -- Laplacian kernel, with inverse Fourier
                  transform :class:`scipy.stats.cauchy`, or
                - ``'cauchy'`` -- Cauchy kernel, with inverse Fourier transform
                  :class:`scipy.stats.laplace`.

            Alternatively, a positive, shift-invariant kernel can be implicitly
            specified by providing its inverse Fourier transform as a
            univariate probability distribution subclassing
            :class:`scipy.stats.rv_continuous`.

        n_components : int
            Number of random samples used to generate features. If
            ``method='weight_offset'``, this corresponds directly to the number
            of features generated. If ``method='weight_only'``, then
            ``2 * n_components`` features are generated.

        shape : float
            Shape parameter. Must be greater than zero. Larger numbers
            correspond to "sharper" kernels. Scaled to be consistent with
            ``gamma`` from :class:`sklearn.kernel_approximation.RBFSampler`.
            This can lead to a mysterious factor of ``sqrt(2)`` in other
            kernels. Default is ``1``.

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
            scale=1,
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
        X_scaled = np.sqrt(2 * self.shape) * X
        products = X_scaled @ self.random_weights_  # (n_samples, n_components)
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
    r"""Kernel approximation with random binning.

    Highly experimental! For more details, see [RR07]_.

    Attributes
    ----------
    n_features_in_ : int
        Number of features input.
    n_features_out_ : int
        Number of features output. This attribute is not available in
        estimators from :mod:`sklearn.kernel_approximation`.
    ddot_ : scipy.stats.rv_continuous
        Probability distribution corresponding to ``\delta \ddot{k}(\delta)``.
    pitches_ : np.ndarray, shape (n_features, n_components)
        Grid pitches for each component.
    shifts_ : np.ndarray, shape (n_features, n_components)
        Grid shifts for each component.
    encoder_ : sklearn.preprocessing.OneHotEncoder
        One-hot encoder used for hashing sample coordinates for each component.

    Examples
    --------
    Generate randomly binned features from a Laplacian kernel

    >>> ka = pykoop.RandomBinningKernelApprox(
    ...     kernel_or_ddot='laplacian',
    ...     n_components=10,
    ...     shape=1,
    ...     random_state=1234,
    ... )
    >>> ka.fit(X_msd[:, 1:])  # Remove episode feature
    RandomBinningKernelApprox(n_components=10, random_state=1234)
    >>> ka.transform(X_msd[:, 1:])
    array([...])
    """

    _ddot_lookup = {
        'laplacian': scipy.stats.gamma(a=2),
    }
    r"""Lookup table for ``\delta \ddot{k}(\delta)``."""

    def __init__(
        self,
        kernel_or_ddot: Union[str, scipy.stats.rv_continuous] = 'laplacian',
        n_components: int = 100,
        shape: float = 1,
        encoder_kwargs: Dict[str, Any] = None,
        random_state: Union[int, np.random.RandomState] = None,
    ) -> None:
        r"""Instantiate :class:`RandomBinningKernelApprox`.

        Parameters
        ----------
        kernel_or_ddot : Union[str, scipy.stats.rv_continuous]
            Kernel to approximate. Possible options are

                - ``'laplacian'`` -- Laplacian kernel, with
                  ``\delta \ddot{k}(\delta)`` being :class:`scipy.stats.gamma`
                  with shape parameter ``a=2`` (default).

            Alternatively, a separable, positive, shift-invariant kernel can be
            implicitly specified by providing ``\delta \ddot{k}(\delta)`` as a
            univariate probability distribution subclassing
            :class:`scipy.stats.rv_continuous`.

        n_components : int
            Number of random samples used to generate features. The higher the
            number of components, the higher the number of features. Since
            unoccupied bins are eliminated, it's impossible to know the exact
            number of features before fitting.

        shape : float
            Shape parameter. Must be greater than zero. Larger numbers
            correspond to "sharper" kernels. Scaled to be consistent with
            ``gamma`` from :class:`sklearn.kernel_approximation.RBFSampler`.
            This can lead to a mysterious factor of ``sqrt(2)`` in other
            kernels. Default is ``1``.

        encoder_kwargs : Dict[str, Any]
            Extra keyword arguments for internal
            :class:`sklearn.preprocessing.OneHotEncoder`. For experimental use
            only. The wrong arguments can break everything. Overrides defaults.

        random_state : Union[int, np.random.RandomState]
            Random seed.
        """
        self.kernel_or_ddot = kernel_or_ddot
        self.n_components = n_components
        self.shape = shape
        self.encoder_kwargs = encoder_kwargs
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
        X = sklearn.utils.validation.check_array(X)
        X_scaled = np.sqrt(2 * self.shape) * X
        # Set ``ddot_``
        if isinstance(self.kernel_or_ddot, str):
            self.ddot_ = self._ddot_lookup[self.kernel_or_ddot]
        else:
            self.ddot_ = self.kernel_or_ddot
        # Validate input
        if self.n_components <= 0:
            raise ValueError('`n_components` must be positive.')
        if self.shape <= 0:
            raise ValueError('`shape` must be positive.')
        # Set number of input and output features
        self.n_features_in_ = X_scaled.shape[1]
        # Generate grids
        self.pitches_ = self.ddot_.rvs(
            size=(self.n_features_in_, self.n_components),
            random_state=self.random_state,
        )
        self.shifts_ = scipy.stats.uniform.rvs(
            loc=0,
            scale=self.pitches_,
            random_state=self.random_state,
        )
        # Hash samples and fit one-hot encoder
        X_hashed = self._hash_samples(X_scaled)
        encoder_args = {
            'categories': 'auto',
            'sparse': False,
            'handle_unknown': 'ignore',
        }
        if self.encoder_kwargs is not None:
            encoder_args.update(self.encoder_kwargs)
        self.encoder_ = sklearn.preprocessing.OneHotEncoder(**encoder_args)
        self.encoder_.fit(X_hashed)
        # Get number of output features from the encoder
        self.n_features_out_ = np.concatenate(self.encoder_.categories_).size
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
        X_scaled = np.sqrt(2 * self.shape) * X
        X_hashed = self._hash_samples(X_scaled)
        Xt = self.encoder_.transform(X_hashed) / np.sqrt(self.n_components)
        return Xt

    def _hash_samples(self, X: np.ndarray) -> np.ndarray:
        """Calculate bin coordinates of each sample and hash them.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data matrix.

        Returns
        -------
        np.ndarray, shape (n_samples, n_components)
            Hashed bin coordinates for each component.
        """
        # Calculate bin coordinates of each sample
        # Shape is (n_samples, n_features, n_components)
        coords = np.floor((X[:, :, np.newaxis] - self.shifts_) / self.pitches_)
        # Convert each coordinate to tuple (so it's immutable), then compute
        # its hash.
        coords_hashed = np.array(
            [[
                hash(tuple(coords[sample, :, component].astype(int)))
                for component in range(coords.shape[2])
            ] for sample in range(coords.shape[0])],
            dtype=int,
        )
        return coords_hashed
