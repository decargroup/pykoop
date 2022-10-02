"""Center generation from data for radial basis functions."""

import logging
from typing import Union

import numpy as np
import sklearn.base
from scipy import stats

log = logging.getLogger(__name__)

# TODO Add citations to each method


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
        if self.symmetric_range:
            self.range_max_ = np.max(np.abs(X), axis=0)
            self.range_min_ = -self.range_max_
        else:
            self.range_max_ = np.max(X, axis=0)
            self.range_min_ = np.min(X, axis=0)
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
            Seed for sampling from uniform distribution.
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
        if self.symmetric_range:
            self.range_max_ = np.max(np.abs(X), axis=0)
            self.range_min_ = -self.range_max_
        else:
            self.range_max_ = np.max(X, axis=0)
            self.range_min_ = np.min(X, axis=0)
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
            Seed for sampling from normal distribution.
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
        self.cov_ = np.cov(X.T)
        # Generate centers
        self.centers_ = stats.multivariate_normal.rvs(
            mean=self.mean_,
            cov=self.cov_,
            size=self.n_centers_,
            random_state=self.random_state,
        )
        return self


class LhsCenters(sklearn.base.BaseEstimator):
    """Centers generated with Latin hypercube sampling.

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
