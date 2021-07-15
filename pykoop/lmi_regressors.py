"""Collection of LMI-based Koopman regressors.

Warning
-------
Importing this module has side effects! When imported, the module creates a
temporary directory with the prefix ``pykoop_``, which is used to memoize long
computations that may be repreated frequently. It also catches ``SIGINT`` so
that long regressions can be stopped politely.
"""

import logging
import signal
import tempfile
from typing import Any

import joblib
import numpy as np
import picos
from scipy import linalg

from . import koopman_pipeline

# Create logger
log = logging.getLogger(__name__)

# Create temporary cache directory for memoized computations
cachedir = tempfile.TemporaryDirectory(prefix='pykoop_')
log.info(f'Temporary directory created at `{cachedir.name}`')
memory = joblib.Memory(cachedir.name, verbose=0)

# Create signal handler to politely stop computations
polite_stop = False


def sigint_handler(sig, frame):
    """Signal handler for ^C."""
    print('Stop requested. Regression will terminate at next iteration...')
    global polite_stop
    polite_stop = True


signal.signal(signal.SIGINT, sigint_handler)


# TODO Make it all one class...
# TODO Hinf and the like only really reuse create_problem and extract_solution.
#      just pull them out completely...
class LmiEdmd(koopman_pipeline.KoopmanRegressor):
    """LMI-based EDMD with Tikhonov regularization.

    Attributes
    ----------
    alpha_tikhonov_ : float
        Tikhonov regularization coefficient. Must be saved separately for
        derived classes with mixed regularizers to use.
    solver_params_ : dict[str, Any]
        Solver parameters used (defaults merged with constructor input).
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    coef_ : np.ndarray
        Fit coefficient matrix.
    """

    # Default solver parameters
    _default_solver_params: dict[str, Any] = {
        'primals': None,
        'duals': None,
        'dualize': True,
        'abs_bnb_opt_tol': None,
        'abs_dual_fsb_tol': None,
        'abs_ipm_opt_tol': None,
        'abs_prim_fsb_tol': None,
        'integrality_tol': None,
        'markowitz_tol': None,
        'rel_bnb_opt_tol': None,
        'rel_dual_fsb_tol': None,
        'rel_ipm_opt_tol': None,
        'rel_prim_fsb_tol': None,
    }

    # Override since PICOS only works with ``float64``.
    _check_X_y_params: dict[str, Any] = {
        'multi_output': True,
        'y_numeric': True,
        'dtype': 'float64',
    }

    def __init__(self,
                 alpha: float = 0,
                 inv_method: str = 'svd',
                 picos_eps: float = 0,
                 solver_params: dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiEdmd`.

        Parameters
        ----------
        alpha : float
            Tikhonov regularization coefficient. Can be zero without
            introducing numerical problems.

        inv_method : str
            Method to handle or avoid inversion of the ``H`` matrix when
            forming the LMI problem. Possible values are

            - ``'inv'`` -- invert ``H`` directly,
            - ``'pinv'`` -- apply the Moore-Penrose pseudoinverse to ``H``,
            - ``'eig'`` -- split ``H`` using an eigendecomposition,
            - ``'ldl'`` -- split ``H`` using an LDL decomposition,
            - ``'chol'`` -- split ``H`` using a Cholesky decomposition,
            - ``'sqrt'`` -- split ``H`` using :func:`scipy.linalg.sqrtm()`, or
            - ``'svd'`` -- split ``H`` using a singular value decomposition.

        picos_eps : float
            Tolerance used for strict LMIs. If nonzero, should be larger than
            solver tolerance.

        solver_params : dict[str, Any]
            Parameters passed to PICOS
            :func:`picos.Problem.solve()`.
        """
        # TODO Add truncated svd inv_method (dimension, cutoff, or optht)
        self.alpha = alpha
        self.inv_method = inv_method
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Save Tikhonov coefficient
        self.alpha_tikhonov_ = self.alpha
        # Form optimization problem
        problem = self._create_problem(X_unshifted, X_shifted)
        # Solve optimiztion problem
        problem.solve(**self.solver_params_)
        # Save solution status
        self.solution_status_ = problem.last_solution.claimedStatus
        # Extract solution from ``Problem`` object
        coef = self._extract_solution(problem)
        return coef

    def _validate_parameters(self) -> None:
        # Validate ``alpha``
        if self.alpha < 0:
            raise ValueError('Parameter `alpha` must be positive or zero.')
        # Validate ``inv_method``
        valid_inv_methods = [
            'inv', 'pinv', 'eig', 'ldl', 'chol', 'sqrt', 'svd'
        ]
        if self.inv_method not in valid_inv_methods:
            raise ValueError('`inv_method` must be one of: '
                             f'{", ".join(valid_inv_methods)}.')
        # Validate ``picos_eps``
        if self.picos_eps < 0:
            raise ValueError('Parameter `picos_eps` must be positive or zero.')

    def _create_problem(self, X_unshifted: np.ndarray,
                        X_shifted: np.ndarray) -> picos.Problem:
        """Create optimization problem.

        Parameters
        ----------
        X_unshifted : np.ndarray
            Unshifted data matrix.
        X_shifted : np.ndarray
            Shifted data matrix.

        Returns
        -------
        picos.Problem
            Optimization problem.
        """
        c, G, H, _ = _calc_c_G_H(X_unshifted, X_shifted, self.alpha_tikhonov_)
        # Optimization problem
        problem = picos.Problem()
        # Constants
        G_T = picos.Constant('G^T', G.T)
        # Variables
        U = picos.RealVariable('U', (G.shape[0], H.shape[0]))
        Z = picos.SymmetricVariable('Z', (G.shape[0], G.shape[0]))
        # Constraints
        problem.add_constraint(Z >> self.picos_eps)
        # Choose method to handle inverse of H
        if self.inv_method == 'inv':
            H_inv = picos.Constant('H^-1', _calc_Hinv(H))
            problem.add_constraint(picos.block([
                [Z, U],
                [U.T, H_inv],
            ]) >> 0)
        elif self.inv_method == 'pinv':
            H_inv = picos.Constant('H^+', _calc_Hpinv(H))
            problem.add_constraint(picos.block([
                [Z, U],
                [U.T, H_inv],
            ]) >> 0)
        elif self.inv_method == 'eig':
            VsqrtLmb = picos.Constant('(V Lambda^(1/2))', _calc_VsqrtLmb(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * VsqrtLmb],
                    [VsqrtLmb.T * U.T, 'I'],
                ]) >> 0)
        elif self.inv_method == 'ldl':
            LsqrtD = picos.Constant('(L D^(1/2))', _calc_LsqrtD(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * LsqrtD],
                    [LsqrtD.T * U.T, 'I'],
                ]) >> 0)
        elif self.inv_method == 'chol':
            L = picos.Constant('L', _calc_L(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * L],
                    [L.T * U.T, 'I'],
                ]) >> 0)
        elif self.inv_method == 'sqrt':
            sqrtH = picos.Constant('sqrt(H)', _calc_sqrtH(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * sqrtH],
                    [sqrtH.T * U.T, 'I'],
                ]) >> 0)
        elif self.inv_method == 'svd':
            QSig = picos.Constant(
                'Q Sigma', _calc_QSig(X_unshifted, self.alpha_tikhonov_))
            problem.add_constraint(
                picos.block([
                    [Z, U * QSig],
                    [QSig.T * U.T, 'I'],
                ]) >> 0)
        # Set objective
        obj = c - 2 * picos.trace(U * G_T) + picos.trace(Z)
        problem.set_objective('min', obj)
        return problem

    @staticmethod
    def _extract_solution(problem: picos.Problem) -> np.ndarray:
        """Extract solution from an optimization problem.

        Parameters
        ----------
        problem : picos.Problem
            Solved optimization problem.

        Returns
        -------
        np.ndarray
            Solution.
        """
        return np.array(problem.get_valued_variable('U'), ndmin=2).T


@memory.cache
def _calc_c_G_H(
    X_unshifted: np.ndarray,
    X_shifted: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute ``c``, ``G``, and ``H``.

    Parameters
    ----------
    X_unshifted : np.ndarray
        Unshifted data matrix.
    X_shifted: np.ndarray
        Shifted data matrix.
    alpha: float
        Tikhonov regularization coefficient.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, dict[str, Any]]
        Tuple containing ``c``, ``G``, and ``H``, along with numerical
        statistics.
    """
    # Compute G and H
    Psi = X_unshifted.T
    Theta_p = X_shifted.T
    p, q = Psi.shape
    # Compute G and Tikhonov-regularized H
    G = (Theta_p @ Psi.T) / q
    H_unreg = (Psi @ Psi.T) / q
    H_reg = H_unreg + (alpha * np.eye(p)) / q
    # Compute c
    c = np.trace(Theta_p @ Theta_p.T) / q
    # Check condition number and rank of G and H
    cond_G = np.linalg.cond(G)
    rank_G = np.linalg.matrix_rank(G)
    shape_G = G.shape
    cond_H_unreg = np.linalg.cond(H_unreg)
    rank_H_unreg = np.linalg.matrix_rank(H_unreg)
    shape_H_unreg = H_unreg.shape
    cond_H_reg = np.linalg.cond(H_reg)
    rank_H_reg = np.linalg.matrix_rank(H_reg)
    shape_H_reg = H_reg.shape
    stats = {
        'cond_G': cond_G,
        'rank_G': rank_G,
        'shape_G': shape_G,
        'cond_H_unreg': cond_H_unreg,
        'rank_H_unreg': rank_H_unreg,
        'shape_H_unreg': shape_H_unreg,
        'cond_H_reg': cond_H_reg,
        'rank_H_reg': rank_H_reg,
        'shape_H_reg': shape_H_reg,
    }
    stats_str = {}
    for key in stats:
        if 'cond' in key:
            stats_str[key] = f'{stats[key]:.2e}'
        else:
            stats_str[key] = stats[key]
    log.info(f'_calc_c_G_H() stats: {stats_str}')
    return c, G, H_reg, stats


@memory.cache
def _calc_Hinv(H: np.ndarray) -> np.ndarray:
    """Compute inverse of ``H``."""
    return linalg.inv(H)


@memory.cache
def _calc_Hpinv(H: np.ndarray) -> np.ndarray:
    """Compute Moore-Penrose pseudoinverse of ``H``."""
    return linalg.pinv(H)


@memory.cache
def _calc_VsqrtLmb(H: np.ndarray) -> np.ndarray:
    """Split ``H`` using its eigendecomposition."""
    lmb, V = linalg.eigh(H)
    return V @ np.diag(np.sqrt(lmb))


@memory.cache
def _calc_LsqrtD(H: np.ndarray) -> np.ndarray:
    """Split ``H`` using its LDL decomposition."""
    L, D, _ = linalg.ldl(H)
    return L @ np.sqrt(D)


@memory.cache
def _calc_L(H: np.ndarray) -> np.ndarray:
    """Split ``H`` using its Cholesky decomposition."""
    return linalg.cholesky(H, lower=True)


@memory.cache
def _calc_sqrtH(H: np.ndarray) -> np.ndarray:
    """Split ``H`` using ``scipy.linalg.sqrtm``."""
    # Since H is symmetric, its square root is symmetric.
    # Otherwise, this would not work!
    return linalg.sqrtm(H)


@memory.cache
def _calc_QSig(X: np.ndarray, alpha: float, r: int = None) -> np.ndarray:
    """Split ``H`` using the truncated SVD of ``X``.

    Parameters
    ----------
    X : np.ndarray
        ``X``, where ``H = X @ X.T``.
    alpha : float
        Tikhonov regularization coefficient.
    r : int
        Singular value truncation index.
    """
    # SVD
    Q, s, _ = linalg.svd(X.T, full_matrices=False)
    # Truncate
    Qr = Q[:, :r]
    sr = s[:r]
    # Regularize
    q = X.shape[0]
    sr_reg = np.sqrt((sr**2 + alpha) / q)
    Sr_reg = np.diag(sr_reg)
    # Multiply with Q and return
    QSig = Qr @ Sr_reg
    log.info(f'_calc_QSig() stats: r={r}, alpha={alpha}, len(s)={len(s)}, '
             f's[0]={s[0]}, s[-1]={s[-1]}')
    return QSig
