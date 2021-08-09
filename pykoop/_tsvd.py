"""Compute truncated singular value decomposition."""

import logging

import numpy as np
import optht
from scipy import linalg

log = logging.getLogger(__name__)


def _tsvd(
    X: np.ndarray,
    method: str,
    param: float = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute truncated singular value decomposition.

    Parameters
    ----------
    X : np.ndarray
        Data matrix

    method : str
        Method name. One of

        - ``'economy'`` -- use economy-sized SVD,
        - ``'unknown_noise'`` -- use optimal hard truncation [optht]_ with
          unknown noise,
        - ``'known_noise'`` -- use optimal hard truncation [optht]_ with known
          noise ``param``,
        - ``'threshold'`` -- use singular values larger than ``param``, or
        - ``'rank'`` -- keep ``param`` singular values.

    param : Union[float, int]
        Noise, threshold, or rank depending on ``method``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Left singular vectors ``Q``, singular values ``sig``, and right
        singular vectors ``Z`` (not transposed).
    """
    # Check param value
    valid_methods = [
        'economy', 'unknown_noise', 'known_noise', 'cutoff', 'rank'
    ]
    if (param is None) and (method not in valid_methods[:2]):
        raise ValueError(f'`param` must be specified for method {method}.')
    if (param is not None) and (param < 0):
        raise ValueError('`param` must be greater than zero.')
    # Compute SVDs
    Q, sig, Zh = linalg.svd(X, full_matrices=False)
    # Transpose notation to make checking math easier
    Z = Zh.T
    # Truncate SVD
    if method == 'economy':
        rank = sig.shape[0]
    elif method == 'unknown_noise':
        rank = optht.optht(X.T, sig)
    elif method == 'known_noise':
        rank = optht.optht(X.T, sig, param)
    elif method == 'cutoff':
        greater_than_cutoff = np.where(sig > param)
        if greater_than_cutoff[0].size > 0:
            rank = np.max(greater_than_cutoff) + 1
        else:
            rank = 0
    elif method == 'rank':
        rank = param
    else:
        raise ValueError(f'`method` must be one of {valid_methods}.')
    Q_r = Q[:, :rank]
    sig_r = sig[:rank]
    Z_r = Z[:, :rank]
    stats = {
        'method': method,
        'shape': X.shape,
        'rank': sig.shape[0],
        'reduced_rank': rank,
        'max_sv': f'{np.max(sig):.2e}',
        'min_sv': f'{np.min(sig):.2e}',
        'reduced_min_sv': f'{np.min(sig_r):.2e}',
    }
    log.info(f'`_tsvd()` stats: {stats}')
    return (Q_r, sig_r, Z_r)
