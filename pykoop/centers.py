"""Center generation from data for radial basis functions."""

import abc
import logging
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import sklearn.base
import sklearn.cluster
import sklearn.mixture
from scipy import stats

log = logging.getLogger(__name__)


class Centers(sklearn.base.BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for all center generation estimators.

    All attributes with a trailing underscore must be set in the subclass'
    :func:`fit`.

    Attributes
    ----------
    centers_ : np.ndarray
        Centers, shape (n_centers, n_features).
    n_centers_ : int
        Number of centers generated.
    n_features_in_ : int
        Number of features input.
    """

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'Centers':
        """Generate centers from data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.

        Returns
        -------
        Centers
            Instance of itself.

        Raises
        ------
        ValueError
            If any of the constructor parameters are incorrect.
        """
        raise NotImplementedError()


class GridCenters(Centers):
    """Centers generated on a uniform grid.

    Attributes
    ----------
    centers_ : np.ndarray
        Centers, shape (n_centers, n_features).
    n_centers_ : int
        Number of centers generated.
    n_features_in_ : int
        Number of features input.
    range_max_ : np.ndarray
        Maximum value of each feature used to generate grid.
    range_min_ : np.ndarray
        Minimum value of each feature used to generate grid.

    Examples
    --------
    Generate centers on a grid

    >>> grid = pykoop.GridCenters(n_points_per_feature=4)
    >>> grid.fit(X_msd[:, 1:])  # Remove episode feature
    GridCenters(n_points_per_feature=4)
    >>> grid.centers_
    array([...])
    """

    def __init__(
        self,
        n_points_per_feature: int = 2,
        symmetric_range: bool = False,
    ) -> None:
        """Instantiate :class:`GridCenters`.

        Parameters
        ----------
        n_points_per_feature : int
            Number of points in grid for each feature.
        symmetric_range : bool
            If true, the grid range for a given feature is forced to be
            symmetric about zero (i.e., ``[-max(abs(x)), max(abs(x))]``).
            Otherwise, the grid range is taken directly on the data
            (i.e., ``[min(x), max(x)]``). Default is false.
        """
        self.n_points_per_feature = n_points_per_feature
        self.symmetric_range = symmetric_range

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'GridCenters':
        # noqa: D102
        X = sklearn.utils.validation.check_array(X)
        self.n_features_in_ = X.shape[1]
        # Validate parameters
        if (self.n_points_per_feature is not None
                and self.n_points_per_feature <= 0):
            raise ValueError('`n_points_per_feature` must be at least one.')
        # Calculate ranges of each feature
        self.range_min_, self.range_max_ = _feature_range(
            X, self.symmetric_range)
        # Generate linspaces for each feature
        linspaces = [
            np.linspace(self.range_min_[i], self.range_max_[i],
                        self.n_points_per_feature)
            for i in range(self.n_features_in_)
        ]
        # Generate centers
        self.centers_ = np.array(np.meshgrid(*linspaces)).reshape(
            self.n_features_in_, -1).T
        self.n_centers_ = self.centers_.shape[0]
        return self


class UniformRandomCenters(Centers):
    """Centers sampled from a uniform distribution.

    Inspired by center generation approach used in [DTK20]_.

    Attributes
    ----------
    centers_ : np.ndarray
        Centers, shape (n_centers, n_features).
    n_centers_ : int
        Number of centers generated.
    n_features_in_ : int
        Number of features input.
    range_max_ : np.ndarray
        Maximum value of each feature used to generate grid.
    range_min_ : np.ndarray
        Minimum value of each feature used to generate grid.

    Examples
    --------
    Generate centers from a uniform distribution

    >>> rand = pykoop.UniformRandomCenters(n_centers=10)
    >>> rand.fit(X_msd[:, 1:])  # Remove episode feature
    UniformRandomCenters(n_centers=10)
    >>> rand.centers_
    array([...])
    """

    def __init__(
        self,
        n_centers: int = 100,
        symmetric_range: bool = False,
        random_state: Union[int, np.random.RandomState] = None,
    ) -> None:
        """Instantiate :class:`UniformRandomCenters`.

        Parameters
        ----------
        n_centers : int
            Number of centers to generate.
        symmetric_range : bool
            If true, the grid range for a given feature is forced to be
            symmetric about zero (i.e., ``[-max(abs(x)), max(abs(x))]``).
            Otherwise, the grid range is taken directly on the data
            (i.e., ``[min(x), max(x)]``). Default is false.
        random_state : Union[int, np.random.RandomState]
            Random seed.
        """
        self.n_centers = n_centers
        self.symmetric_range = symmetric_range
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
    ) -> 'UniformRandomCenters':
        # noqa: D102
        X = sklearn.utils.validation.check_array(X)
        self.n_features_in_ = X.shape[1]
        # Validate parameters
        if self.n_centers <= 0:
            raise ValueError('`n_centers` must be greater than zero.')
        self.n_centers_ = self.n_centers
        # Calculate ranges of each feature
        self.range_min_, self.range_max_ = _feature_range(
            X, self.symmetric_range)
        # Generate centers in range ``[loc, loc + scale]``.
        self.centers_ = np.array([
            stats.uniform.rvs(
                loc=self.range_min_[i],
                scale=(self.range_max_[i] - self.range_min_[i]),
                size=self.n_centers_,
                random_state=self.random_state,
            ) for i in range(self.n_features_in_)
        ]).T
        return self


class GaussianRandomCenters(Centers):
    """Centers sampled from a Gaussian distribution.

    Inspired by center generation approach used in [CHH19]_.

    Attributes
    ----------
    centers_ : np.ndarray
        Centers, shape (n_centers, n_features).
    n_centers_ : int
        Number of centers generated.
    n_features_in_ : int
        Number of features input.
    mean_ : np.ndarray
        Mean feature.
    cov_ : np.ndarray
        Covariance matrix.

    Examples
    --------
    Generate centers from a Gaussian distribution

    >>> rand = pykoop.GaussianRandomCenters(n_centers=10)
    >>> rand.fit(X_msd[:, 1:])  # Remove episode feature
    GaussianRandomCenters(n_centers=10)
    >>> rand.centers_
    array([...])
    """

    def __init__(
        self,
        n_centers: int = 100,
        random_state: Union[int, np.random.RandomState] = None,
    ) -> None:
        """Instantiate :class:`GaussianRandomCenters`.

        Parameters
        ----------
        n_centers : int
            Number of centers to generate.
        random_state : Union[int, np.random.RandomState]
            Random seed.
        """
        self.n_centers = n_centers
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
    ) -> 'GaussianRandomCenters':
        # noqa: D102
        X = sklearn.utils.validation.check_array(X, ensure_min_samples=2)
        self.n_features_in_ = X.shape[1]
        # Validate parameters
        if self.n_centers <= 0:
            raise ValueError('`n_centers` must be greater than zero.')
        self.n_centers_ = self.n_centers
        # Calculate mean and covariance
        self.mean_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X, rowvar=False)
        # Generate centers
        self.centers_ = stats.multivariate_normal.rvs(
            mean=self.mean_,
            cov=self.cov_,
            size=self.n_centers_,
            random_state=self.random_state,
        )
        return self


class QmcCenters(Centers):
    """Centers generated with Quasi-Monte Carlo sampling.

    Attributes
    ----------
    centers_ : np.ndarray
        Centers, shape (n_centers, n_features).
    n_centers_ : int
        Number of centers generated.
    n_features_in_ : int
        Number of features input.
    range_max_ : np.ndarray
        Maximum value of each feature used to generate grid.
    range_min_ : np.ndarray
        Minimum value of each feature used to generate grid.
    qmc_ : stats.qmc.QMCEngine
        Quasi-Monte Carlo sampler instantiated from ``qmc``.

    Examples
    --------
    Generate centers using Latin hypercube sampling (default)

    >>> qmc = pykoop.QmcCenters(n_centers=10)
    >>> qmc.fit(X_msd[:, 1:])  # Remove episode feature
    QmcCenters(n_centers=10)
    >>> qmc.centers_
    array([...])

    Generate centers using a Sobol sequence

    >>> qmc = pykoop.QmcCenters(n_centers=8, qmc=scipy.stats.qmc.Sobol)
    >>> qmc.fit(X_msd[:, 1:])  # Remove episode feature
    QmcCenters(n_centers=8, qmc=<class 'scipy.stats._qmc.Sobol'>)
    >>> qmc.centers_
    array([...])
    """

    def __init__(
        self,
        n_centers: int = 100,
        symmetric_range: bool = False,
        qmc: Callable[..., stats.qmc.QMCEngine] = None,
        qmc_kw: Dict[str, Any] = None,
        random_state: Union[int, np.random.RandomState] = None,
    ) -> None:
        """Instantiate :class:`QmcCenters`.

        Parameters
        ----------
        n_centers : int
            Number of centers to generate.

        symmetric_range : bool
            If true, the grid range for a given feature is forced to be
            symmetric about zero (i.e., ``[-max(abs(x)), max(abs(x))]``).
            Otherwise, the grid range is taken directly on the data
            (i.e., ``[min(x), max(x)]``). Default is false.

        qmc : Callable[..., stats.qmc.QMCEngine]
            Quasi-Monte Carlo method from :mod:`scipy.stats.qmc` to use.
            Argument is the desired subclass of
            :class:`scipy.stats.qmc.QMCEngine` to use. Accepts the class
            itself, not an instance of the class. Possible values are

            - :class:`scipy.stats.qmc.Sobol` -- Sobol sequence,
            - :class:`scipy.stats.qmc.Halton` -- Halton sequence,
            - :class:`scipy.stats.qmc.LatinHypercube` -- Latin hypercube
              sampling (LHS),
            - :class:`scipy.stats.qmc.PoissonDisk` -- Poisson disk sampling
              (requires ``scipy`` v1.9.0.),
            - :class:`scipy.stats.qmc.MultinomialQMC` -- Multinomial
              distribution, and
            - :class:`scipy.stats.qmc.MultivariateNormalQMC` -- Multivariate
              normal distribution.

            If ``None``, defaults to Latin hypercube sampling.

        qmc_kw : Dict[str, Any]
            Additional keyword arguments passed when instantiating ``qmc``. If
            ``seed`` is specified here, it takes precedence over
            ``random_state``.

        random_state : Union[int, np.random.RandomState]
            Random seed.
        """
        self.n_centers = n_centers
        self.symmetric_range = symmetric_range
        self.qmc = qmc
        self.qmc_kw = qmc_kw
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
    ) -> 'QmcCenters':
        # noqa: D102
        X = sklearn.utils.validation.check_array(X, ensure_min_samples=2)
        self.n_features_in_ = X.shape[1]
        # Validate parameters
        if self.n_centers <= 0:
            raise ValueError('`n_centers` must be greater than zero.')
        self.n_centers_ = self.n_centers
        # Calculate ranges of each feature
        self.range_min_, self.range_max_ = _feature_range(
            X, self.symmetric_range)
        # Set default QMC method
        qmc = stats.qmc.LatinHypercube if self.qmc is None else self.qmc
        # Set QMC arguments
        qmc_args = {} if self.qmc_kw is None else self.qmc_kw
        if 'seed' not in qmc_args.keys():
            qmc_args['seed'] = self.random_state
        # Instantiate QMC
        self.qmc_ = qmc(self.n_features_in_, **qmc_args)
        # Generate and scale samples
        unscaled_centers = self.qmc_.random(self.n_centers_)
        self.centers_ = stats.qmc.scale(unscaled_centers, self.range_min_,
                                        self.range_max_)
        return self


class ClusterCenters(Centers):
    """Centers generated from a clustering algorithm.

    Also supports taking centers from the means of a Gaussian mixture model.

    Inspired by center generation approach used in [DTK20]_.

    Attributes
    ----------
    centers_ : np.ndarray
        Centers, shape (n_centers, n_features).
    n_centers_ : int
        Number of centers generated.
    n_features_in_ : int
        Number of features input.
    estimator_ : sklearn.base.BaseEstimator
        Fit clustering estimator or Gaussian mixture model.

    Examples
    --------
    Generate centers using K-means clustering

    >>> kmeans = pykoop.ClusterCenters(sklearn.cluster.KMeans(n_clusters=3))
    >>> kmeans.fit(X_msd[:, 1:])  # Remove episode feature
    ClusterCenters(estimator=KMeans(n_clusters=3))
    >>> kmeans.centers_
    array([...])
    """

    def __init__(self, estimator: sklearn.base.BaseEstimator = None) -> None:
        """Instantiate :class:`ClusterCenters`.

        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator
            Clustering estimator or Gaussian mixture model. Must provide
            ``cluster_centers_`` or ``means_`` once fit. Possible algorithms
            include

            - :class:`sklearn.cluster.KMeans`,
            - :class:`sklearn.cluster.AffinityPropagation`,
            - :class:`sklearn.cluster.MeanShift`,
            - :class:`sklearn.cluster.BisectingKMeans`,
            - :class:`sklearn.cluster.MiniBatchKMeans`,
            - :class:`sklearn.mixture.GaussianMixture`, or
            - :class:`sklearn.mixture.BayesianGaussianMixture`.

            The number of centers generated is controlled by the chosen
            estimator. If a random seed is desired, it must be set in the
            chosen estimator. Defaults to :class:`sklearn.cluster.KMeans`.
        """
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'ClusterCenters':
        # noqa: D102
        X = sklearn.utils.validation.check_array(X)
        self.n_features_in_ = X.shape[1]
        # Clone and fit estimator
        self.estimator_ = (sklearn.base.clone(self.estimator) if self.estimator
                           is not None else sklearn.cluster.KMeans())
        self.estimator_.fit(X)
        # Set centers
        if hasattr(self.estimator_, 'cluster_centers_'):
            self.centers_ = self.estimator_.cluster_centers_
        elif hasattr(self.estimator_, 'means_'):
            self.centers_ = self.estimator_.means_
        else:
            raise ValueError('`estimator` must provide either '
                             '`cluster_centers_` or `means_` after fit.')
        self.n_centers_ = self.centers_.shape[0]
        return self


class GaussianMixtureRandomCenters(Centers):
    """Centers generated from sampling a Gaussian mixture model.

    Attributes
    ----------
    centers_ : np.ndarray
        Centers, shape (n_centers, n_features).
    n_centers_ : int
        Number of centers generated.
    n_features_in_ : int
        Number of features input.
    estimator_ : sklearn.base.BaseEstimator
        Fit Gaussian mixture model.

    Examples
    --------
    Generate centers by sampling a Gaussian mixture model

    >>> gmm = pykoop.GaussianMixtureRandomCenters(n_centers=100,
    ...     estimator=sklearn.mixture.GaussianMixture(n_components=3))
    >>> gmm.fit(X_msd[:, 1:])  # Remove episode feature
    GaussianMixtureRandomCenters(estimator=GaussianMixture(n_components=3))
    >>> gmm.centers_
    array([...])
    """

    def __init__(
        self,
        n_centers: int = 100,
        estimator: sklearn.base.BaseEstimator = None,
    ) -> None:
        """Instantiate :class:`GaussianMixtureRandomCenters`.

        Parameters
        ----------
        n_centers : int
            Number of centers to generate.

        estimator : sklearn.base.BaseEstimator
            Gaussian mixture model. Possible algorithms include

            - :class:`sklearn.mixture.GaussianMixture`, or
            - :class:`sklearn.mixture.BayesianGaussianMixture`.

            If a random seed is desired, it must be set in the
            chosen estimator. Defaults to
            :class:`sklearn.mixture.GaussianMixture(n_components=2)`.
        """
        self.n_centers = n_centers
        self.estimator = estimator

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
    ) -> 'GaussianMixtureRandomCenters':
        # noqa: D102
        X = sklearn.utils.validation.check_array(X)
        self.n_features_in_ = X.shape[1]
        # Clone and fit estimator
        self.estimator_ = (sklearn.base.clone(self.estimator) if self.estimator
                           is not None else sklearn.mixture.GaussianMixture(
                               n_components=2))
        self.estimator_.fit(X)
        # Sample fit distribution
        self.centers_ = self.estimator_.sample(self.n_centers)[0]
        self.n_centers_ = self.centers_.shape[0]
        return self


class DataCenters(Centers):
    """Centers taken from raw data upon instantiation or fit.

    Attributes
    ----------
    centers_ : np.ndarray
        Centers, shape (n_centers, n_features).
    n_centers_ : int
        Number of centers generated.
    n_features_in_ : int
        Number of features input.
    """

    def __init__(self, centers: np.ndarray = None):
        """Instantiate :class:`DataCenters`.

        Parameters
        ----------
        centers : np.ndarray
            Array of centers, shape (n_centers, n_features). If ``None``, then
            centers are taken from fit.
        """
        self.centers = centers

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DataCenters':
        # noqa: D102
        X = sklearn.utils.validation.check_array(X)
        self.n_features_in_ = X.shape[1]
        # Set centers
        if self.centers is None:
            self.centers_ = X
        else:
            if self.centers.shape[1] != self.n_features_in_:
                raise ValueError('`centers` must have shape '
                                 '(n_centers, n_features).')
            self.centers_ = self.centers
        # Set center shape
        self.n_centers_ = self.centers_.shape[0]
        return self


def _feature_range(
    X: np.ndarray,
    symmetric_range: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get range of features from data matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    symmetric_range : bool
        If true, the grid range for a given feature is forced to be symmetric
        about zero (i.e., ``[-max(abs(x)), max(abs(x))]``). Otherwise, the grid
        range is taken directly on the data (i.e., ``[min(x), max(x)]``).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Minumum and maximum values of each feature.
    """
    if symmetric_range:
        range_max = np.max(np.abs(X), axis=0)
        range_min = -range_max
    else:
        range_max = np.max(X, axis=0)
        range_min = np.min(X, axis=0)
    return (range_min, range_max)
