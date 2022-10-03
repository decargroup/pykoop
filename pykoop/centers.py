"""Center generation from data for radial basis functions."""

import logging
from typing import Any, Callable, Dict, ParamSpecKwargs, Tuple, Union

import numpy as np
import sklearn.base
from scipy import stats

log = logging.getLogger(__name__)

# TODO Add citations to each method
# TODO Consider Gaussian mixture model


class GridCenters(sklearn.base.BaseEstimator):
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
        """Generate centers from a uniform grid.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.

        Returns
        -------
        GridCenters
            Instance of itself.

        Raises
        ------
        ValueError
            If any of the constructor parameters are incorrect.
        """
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


class UniformRandomCenters(sklearn.base.BaseEstimator):
    """Centers sampled from a uniform distribution.

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
        """Generate centers from a uniform distribution.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.

        Returns
        -------
        UniformRandomCenters
            Instance of itself.

        Raises
        ------
        ValueError
            If any of the constructor parameters are incorrect.
        """
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


class GaussianRandomCenters(sklearn.base.BaseEstimator):
    """Centers sampled from a Gaussian distribution.

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
        """Generate centers from a Gaussian distribution.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.

        Returns
        -------
        GaussianRandomCenters
            Instance of itself.

        Raises
        ------
        ValueError
            If any of the constructor parameters are incorrect.
        """
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


class QmcCenters(sklearn.base.BaseEstimator):
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
    """

    def __init__(
        self,
        n_centers: int = 100,
        symmetric_range: bool = False,
        qmc: Callable[[int, ParamSpecKwargs], stats.qmc.QMCEngine] = None,
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

        qmc : Callable[[int, ParamSpecKwargs], stats.qmc.QMCEngine]
            Quasi-Monte Carlo method from :mod:`scipy.stats.qmc` to use.
            Argument is the desired subclass of
            :class:`scipy.stats.qmc.QMCEngine` to use. Accepts the class
            itself, not an instance of the class. Possible values are

            - :class:`scipy.stats.qmc.Sobol` -- Sobol sequence,
            - :class:`scipy.stats.qmc.Halton` -- Halton sequence,
            - :class:`scipy.stats.qmc.LatinHypercube` -- Latin hypercube
              sampling (LHS),
            - :class:`scipy.stats.qmc.PoissonDisk` -- Poisson disk sampling,
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
        """Generate centers using Quasi-Monte Carlo sampling.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.

        Returns
        -------
        QmcCenters
            Instance of itself.

        Raises
        ------
        ValueError
            If any of the constructor parameters are incorrect.
        """
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


class ClusterCenters(sklearn.base.BaseEstimator):
    """Centers generated from a clustering algorithm.

    Attributes
    ----------
    centers_ : np.ndarray
        Centers, shape (n_centers, n_features).
    n_centers_ : int
        Number of centers generated.
    n_features_in_ : int
        Number of features input.
    """

    pass


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
