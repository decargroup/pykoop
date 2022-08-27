"""Truncated singular value decomposition."""

import logging

import numpy as np
import optht
import sklearn.base
from scipy import linalg

log = logging.getLogger(__name__)


class Tsvd(sklearn.base.BaseEstimator):
    """Truncated singular value decomposition.

    Attributes
    ----------
    left_singular_vectors_ : np.ndarray
        Left singular vectors.
    singular_values_ : np.ndarray
        Singular values.
    right_singular_vectors_ : np.ndarray
        Right singular vectors.
    n_features_in_ : int
        Number of features input.
    """

    def __init__(self,
                 truncation: str = 'economy',
                 truncation_param: float = None) -> None:
        """Instantiate :class:`Tsvd`.

        Parameters
        ----------
        truncation : str
            Truncation method. Possible values are

            - ``'economy'`` -- do not truncate (use economy SVD),
            - ``'unknown_noise'``-- truncate using optimal hard truncation
              [GD14]_ with unknown noise,
            - ``'known_noise'`` -- truncate using optimal hard truncation
              [GD14]_ with known noise,
            - ``'cutoff'`` -- truncate singular values smaller than a cutoff,
              or
            - ``'rank'`` -- truncate singular values to a fixed rank.

        truncation_param : float
            Parameter whose interpretation is based on the truncation method.
            For each truncation method, ``truncation_param`` is interpreted as

            - ``'economy'`` -- ignored,
            - ``'unknown_noise'``-- ignored,
            - ``'known_noise'`` -- known noise magnitude,
            - ``'cutoff'`` -- singular value cutoff, or
            - ``'rank'`` -- desired rank.

        Notes
        -----
        Optimal hard truncation [GD14]_ assumes the noisy measured matrix
        ``X_measured`` is composed of::

            X_measured = X_true + noise_magnitude * X_noise

        where ``X_noise`` is composed of i.i.d. Gaussian variables with zero
        mean and unit variance.

        Warnings
        --------
        Does not consider episode features!

        Examples
        --------
        SVD with no truncation

        >>> tsvd = pykoop.Tsvd()
        >>> tsvd.fit(X_msd)
        Tsvd()
        >>> tsvd.singular_values_
        array([...])

        SVD with cutoff truncation

        >>> tsvd = pykoop.Tsvd(truncation='cutoff', truncation_param=1e-3)
        >>> tsvd.fit(X_msd)
        Tsvd(truncation='cutoff', truncation_param=0.001)
        >>> tsvd.singular_values_
        array([...])

        SVD with manual rank truncation

        >>> tsvd = pykoop.Tsvd(truncation='rank', truncation_param=2)
        >>> tsvd.fit(X_msd)
        Tsvd(truncation='rank', truncation_param=2)
        >>> tsvd.singular_values_
        array([...])
        """
        self.truncation = truncation
        self.truncation_param = truncation_param

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'Tsvd':
        """Compute the truncated singular value decomposition.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.

        Returns
        -------
        Tsvd
            Instance of itself.

        Raises
        ------
        ValueError
            If any of the constructor parameters are incorrect.
        """
        X = sklearn.utils.validation.check_array(X)
        self.n_features_in_ = X.shape[1]
        # Check param value
        valid_methods_noth = ['economy', 'unknown_noise']
        valid_methods_th = ['known_noise', 'cutoff', 'rank']
        valid_methods = valid_methods_noth + valid_methods_th
        if ((self.truncation_param is None)
                and (self.truncation in valid_methods_th)):
            raise ValueError('`truncation_param` must be specified for method '
                             f'{self.truncation}.')
        if (self.truncation_param is not None) and (self.truncation_param < 0):
            raise ValueError('`truncation_param` must be greater than zero.')
        if self.truncation not in valid_methods:
            raise ValueError(f'`method` must be one of {valid_methods}.')
        # Compute SVDs
        Q, sig, Zh = linalg.svd(X, full_matrices=False)
        # Transpose notation to make checking math easier
        Z = Zh.T
        # Truncate SVD
        if self.truncation == 'economy':
            rank = sig.shape[0]
        elif self.truncation == 'unknown_noise':
            rank = optht.optht(X.T, sig)
        elif self.truncation == 'known_noise':
            rank = optht.optht(X.T, sig, self.truncation_param)
        elif self.truncation == 'cutoff':
            greater_than_cutoff = np.where(sig > self.truncation_param)
            if greater_than_cutoff[0].size > 0:
                rank = np.max(greater_than_cutoff) + 1
            else:
                rank = 0
        elif self.truncation == 'rank':
            rank = self.truncation_param
        else:
            # Already checked
            assert False
        Q_r = Q[:, :rank]
        sig_r = sig[:rank]
        Z_r = Z[:, :rank]
        stats = {
            'method': self.truncation,
            'shape': X.shape,
            'rank': sig.shape[0],
            'reduced_rank': rank,
            'max_sv': f'{np.max(sig):.2e}',
            'min_sv': f'{np.min(sig):.2e}',
            'reduced_min_sv': f'{np.min(sig_r):.2e}',
        }
        log.info(f'``Tsvd.fit()`` stats: {stats}')
        self.left_singular_vectors_ = Q_r
        self.singular_values_ = sig_r
        self.right_singular_vectors_ = Z_r
        return self
