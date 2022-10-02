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
    """

    def __init__(
        self,
        points_per_feature: int = None,
        symmetric_range: bool = False,
        range_scale: float = 1,
    ) -> None:
        """Instantiate :class:`GridCenters`.

        Parameters
        ----------
        points_per_feature : int
            Number of points in grid for each feature. If ``None``, chosen
            automatically to generate around 100 centers.
        symmetric_range : bool
            If true, the grid range for a given feature is forced to be
            symmetric about zero (i.e., ``[-max(abs(x)), max(abs(x))]``).
            Otherwise, the grid range is taken directly on the data
            (i.e., ``[min(x), max(x)]``). Default is false.
        range_scale : float
            Scale factor to apply to grid range. Can be used to pad out the
            size of the grid.
        """
        self.points_per_feature = points_per_feature
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
        if (self.points_per_feature is not None
                and self.points_per_feature <= 0):
            raise ValueError('`points_per_feature` must be at least one.')
        if self.range_scale == 0:
            raise ValueError('`range_scale` cannot be zero.')
        # TODO
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
