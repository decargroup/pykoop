"""Center generation from data for radial basis functions."""

import logging

import numpy as np
import sklearn.base

log = logging.getLogger(__name__)


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
        range_scale: float = 1,
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
        range_scale : float
            Scale factor to apply to grid range. Can be used to pad out the
            size of the grid.
        """
        self.n_points_per_feature = n_points_per_feature
        self.symmetric_range = symmetric_range
        self.range_scale = range_scale

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
        if self.range_scale == 0:
            raise ValueError('`range_scale` cannot be zero.')
        # Calculate ranges of each feature
        if self.symmetric_range:
            self.range_max_ = np.max(np.abs(X), axis=0) * self.range_scale
            self.range_min_ = -feat_max
        else:
            self.range_max_ = np.max(X, axis=0) * self.range_scale
            self.range_min_ = np.min(X, axis=0) * self.range_scale
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


class RandomCenters(sklearn.base.BaseEstimator):
    """Centers sampled from a uniform or normal distribution.

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
