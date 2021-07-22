"""Collection of experimental LMI-based Koopman regressors.

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
from typing import Any, Optional, Union

import joblib
import numpy as np
import optht
import picos
from scipy import linalg

from . import koopman_pipeline

# Create logger
log = logging.getLogger(__name__)

# Create temporary cache directory for memoized computations
_cachedir = tempfile.TemporaryDirectory(prefix='pykoop_')
log.info(f'Temporary directory created at `{_cachedir.name}`')
memory = joblib.Memory(_cachedir.name, verbose=0)

# Create signal handler to politely stop computations
polite_stop = False


def _sigint_handler(sig, frame):
    """Signal handler for ^C."""
    print('Stop requested. Regression will terminate at next iteration...')
    global polite_stop
    polite_stop = True


signal.signal(signal.SIGINT, _sigint_handler)


class LmiEdmd(koopman_pipeline.KoopmanRegressor):
    """LMI-based EDMD with regularization.

    Supports Tikhonov regularization, optionally mixed with matrix two-norm
    regularization or nuclear norm regularization.

    Attributes
    ----------
    self.alpha_tikhonov_ : float
        Tikhonov regularization coefficient used.
    self.alpha_other_ : float
        Matrix two norm or nuclear norm regularization coefficient used.
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
                 ratio: float = 1,
                 reg_method: str = 'tikhonov',
                 inv_method: str = 'svd',
                 tsvd_method: Union[str, tuple[str, ...]] = 'economy',
                 picos_eps: float = 0,
                 solver_params: dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiEdmd`.

        To disable regularization, use ``alpha=0`` paired with
        ``reg_method='tikhonov'``.

        Parameters
        ----------
        alpha : float
            Regularization coefficient. Can only be zero if
            ``reg_method='tikhonov'``.

        ratio : float
            Ratio of matrix two-norm or nuclear norm to use in mixed
            regularization. If ``ratio=1``, no Tikhonov regularization is
            used. Cannot be zero. Ignored if ``reg_method='tikhonov'``.

        reg_method : str
            Regularization method to use. Possible values are

            - ``'tikhonov'`` -- pure Tikhonov regularization (``ratio``
              is ignored),
            - ``'twonorm'`` -- matrix two-norm regularization mixed with
              Tikhonov regularization, or
            - ``'nuclear'`` -- nuclear norm regularization mixed with Tikhonov
              regularization.

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

         tsvd_method : Union[str, tuple[str, ...]]
            Singular value truncation method if ``inv_method='svd'``. Very
            experimental. Possible values are

            - ``'economy'`` or ``('economy', )`` -- use economy SVD without
              truncating singular values
            - ``'unknown_noise'`` or ``('unknown_noise', )`` -- use optimal
              hard truncation [optht]_ with unknown noise to truncate,
            - ``('known_noise', sigma)`` -- use optimal hard truncation
              [optht]_ with known noise ``sigma`` to truncate, or
            - ``('manual', rank)`` -- manually truncate SVDs of ``X_unshifted``
            to ``rank``.

        picos_eps : float
            Tolerance used for strict LMIs. If nonzero, should be larger than
            solver tolerance.

        solver_params : dict[str, Any]
            Parameters passed to PICOS :func:`picos.Problem.solve()`. By
            default, allows chosen solver to select its own tolerances.

        References
        ----------
        .. [optht] Gavish, Matan, and David L. Donoho. "The optimal hard
           threshold for singular values is 4/sqrt(3)" IEEE Transactions on
           Information Theory 60.8 (2014): 5040-5053.
           http://arxiv.org/abs/1305.5870
        """
        self.alpha = alpha
        self.ratio = ratio
        self.reg_method = reg_method
        self.inv_method = inv_method
        self.tsvd_method = tsvd_method
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Compute regularization coefficients
        if self.reg_method == 'tikhonov':
            self.alpha_tikhonov_ = self.alpha
            self.alpha_other_ = 0.0
        else:
            self.alpha_tikhonov_ = self.alpha * (1.0 - self.ratio)
            self.alpha_other_ = self.alpha * self.ratio
        # Form optimization problem. Regularization coefficients must be scaled
        # because of how G and H are defined.
        q = X_unshifted.shape[0]
        problem = self._create_base_problem(X_unshifted, X_shifted,
                                            self.alpha_tikhonov_ / q,
                                            self.inv_method, self.tsvd_method,
                                            self.picos_eps)
        if self.reg_method == 'twonorm':
            problem = _add_twonorm(problem, problem.variables['U'],
                                   self.alpha_other_ / q, self.picos_eps)
        elif self.reg_method == 'nuclear':
            problem = _add_nuclear(problem, problem.variables['U'],
                                   self.alpha_other_ / q, self.picos_eps)
        # Solve optimization problem
        problem.solve(**self.solver_params_)
        # Save solution status
        self.solution_status_ = problem.last_solution.claimedStatus
        # Extract solution from ``Problem`` object
        coef = self._extract_solution(problem)
        return coef

    def _validate_parameters(self) -> None:
        # Check problem creation parameters
        self._validate_problem_parameters(self.alpha, self.inv_method,
                                          self.tsvd_method, self.picos_eps)
        # Check regularization methods
        valid_reg_methods = ['tikhonov', 'twonorm', 'nuclear']
        if self.reg_method not in valid_reg_methods:
            raise ValueError('`reg_method` must be one of '
                             f'{valid_reg_methods}.')
        # Check ratio
        if (self.ratio <= 0) or (self.ratio > 1):
            raise ValueError('`ratio` must be positive and less than one.')
        # Check regularization method
        if self.reg_method != 'tikhonov' and self.alpha == 0:
            raise ValueError(
                "`alpha` cannot be zero if `reg_method='twonorm'` or "
                "`reg_method='nuclear'`.")

    @staticmethod
    def _create_base_problem(
        X_unshifted: np.ndarray,
        X_shifted: np.ndarray,
        alpha_tikhonov: float,
        inv_method: str,
        tsvd_method: Union[str, tuple[str, ...]],
        picos_eps: float,
    ) -> picos.Problem:
        """Create optimization problem."""
        q = X_unshifted.shape[0]
        LmiEdmd._validate_problem_parameters(alpha_tikhonov, inv_method,
                                             tsvd_method, picos_eps)
        c, G, H, _ = _calc_c_G_H(X_unshifted, X_shifted, alpha_tikhonov)
        # Optimization problem
        problem = picos.Problem()
        # Constants
        G_T = picos.Constant('G^T', G.T)
        # Variables
        U = picos.RealVariable('U', (G.shape[0], H.shape[0]))
        Z = picos.SymmetricVariable('Z', (G.shape[0], G.shape[0]))
        # Constraints
        problem.add_constraint(Z >> picos_eps)
        # Choose method to handle inverse of H
        if inv_method == 'inv':
            H_inv = picos.Constant('H^-1', _calc_Hinv(H))
            problem.add_constraint(picos.block([
                [Z, U],
                [U.T, H_inv],
            ]) >> 0)
        elif inv_method == 'pinv':
            H_inv = picos.Constant('H^+', _calc_Hpinv(H))
            problem.add_constraint(picos.block([
                [Z, U],
                [U.T, H_inv],
            ]) >> 0)
        elif inv_method == 'eig':
            VsqrtLmb = picos.Constant('(V Lambda^(1/2))', _calc_VsqrtLmb(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * VsqrtLmb],
                    [VsqrtLmb.T * U.T, 'I'],
                ]) >> 0)
        elif inv_method == 'ldl':
            LsqrtD = picos.Constant('(L D^(1/2))', _calc_LsqrtD(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * LsqrtD],
                    [LsqrtD.T * U.T, 'I'],
                ]) >> 0)
        elif inv_method == 'chol':
            L = picos.Constant('L', _calc_L(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * L],
                    [L.T * U.T, 'I'],
                ]) >> 0)
        elif inv_method == 'sqrt':
            sqrtH = picos.Constant('sqrt(H)', _calc_sqrtH(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * sqrtH],
                    [sqrtH.T * U.T, 'I'],
                ]) >> 0)
        elif inv_method == 'svd':
            QSig = picos.Constant(
                'Q Sigma', _calc_QSig(X_unshifted, alpha_tikhonov,
                                      tsvd_method))
            problem.add_constraint(
                picos.block([
                    [Z, U * QSig],
                    [QSig.T * U.T, 'I'],
                ]) >> 0)
        else:
            # Should never, ever get here.
            assert False
        # Set objective
        obj = c - 2 * picos.trace(U * G_T) + picos.trace(Z)
        problem.set_objective('min', obj)
        return problem

    @staticmethod
    def _extract_solution(problem: picos.Problem) -> np.ndarray:
        """Extract solution from an optimization problem."""
        return np.array(problem.get_valued_variable('U'), ndmin=2).T

    @staticmethod
    def _validate_problem_parameters(alpha: float, inv_method: str,
                                     tsvd_method: Union[str, tuple[str, ...]],
                                     picos_eps: float) -> None:
        """Validate parameters involved in problem creation."""
        # Validate ``alpha``
        if alpha < 0:
            raise ValueError('Parameter `alpha` must be positive or zero.')
        # Validate ``inv_method``
        valid_inv_methods = [
            'inv', 'pinv', 'eig', 'ldl', 'chol', 'sqrt', 'svd'
        ]
        if inv_method not in valid_inv_methods:
            raise ValueError('`inv_method` must be one of '
                             f'{valid_inv_methods}.')
        # Check number of singular values to keep.
        _validate_tsvd_method(tsvd_method, manual_len=2)
        # Validate ``picos_eps``
        if picos_eps < 0:
            raise ValueError('Parameter `picos_eps` must be positive or zero.')


class LmiDmdc(koopman_pipeline.KoopmanRegressor):
    """LMI-based DMDc."""

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
                 ratio: float = 1,
                 tsvd_method: Union[str, tuple[str, ...]] = 'economy',
                 reg_method: str = 'tikhonov',
                 picos_eps: float = 0,
                 solver_params: dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiDmdc`."""
        self.alpha = alpha
        self.ratio = ratio
        self.tsvd_method = tsvd_method
        self.reg_method = reg_method
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Compute regularization coefficients
        if self.reg_method == 'tikhonov':
            self.alpha_tikhonov_ = self.alpha
            self.alpha_other_ = 0.0
        else:
            self.alpha_tikhonov_ = self.alpha * (1.0 - self.ratio)
            self.alpha_other_ = self.alpha * self.ratio
        # Get needed sizes
        q, p = X_unshifted.shape
        p_theta = X_shifted.shape[1]
        # Compute SVDs
        Q_tld, sigma_tld, Zh_tld = linalg.svd(X_unshifted.T,
                                              full_matrices=False)
        Q_hat, sigma_hat, Zh_hat = linalg.svd(X_shifted.T, full_matrices=False)
        # Transpose notation to make checking math easier
        Z_tld = Zh_tld.T
        Z_hat = Zh_hat.T
        # Truncate SVD
        if ((self.tsvd_method == 'economy')
                or (self.tsvd_method[0] == 'economy')):
            r_tld = sigma_tld.shape[0]
            r_hat = sigma_hat.shape[0]
        elif ((self.tsvd_method == 'unknown_noise')
              or (self.tsvd_method[0] == 'unknown_noise')):
            r_tld = optht.optht(X_unshifted.T, sigma_tld)
            r_hat = optht.optht(X_shifted.T, sigma_hat)
        elif self.tsvd_method[0] == 'known_noise':
            variance = self.tsvd_method[1]
            r_tld = optht.optht(X_unshifted.T, sigma_tld, variance)
            r_hat = optht.optht(X_shifted.T, sigma_hat, variance)
        elif self.tsvd_method[0] == 'manual':
            r_tld = self.tsvd_method[1]
            r_hat = self.tsvd_method[2]
        else:
            # Already checked
            assert False
        Q_tld = Q_tld[:, :r_tld]
        sigma_tld = sigma_tld[:r_tld]
        Z_tld = Z_tld[:, :r_tld]
        Q_hat = Q_hat[:, :r_hat]
        sigma_hat = sigma_hat[:r_hat]
        Z_hat = Z_hat[:, :r_hat]
        # Form optimization problem
        problem = self._create_base_problem(Q_tld, sigma_tld, Z_tld, Q_hat,
                                            sigma_hat, Z_hat,
                                            self.alpha_tikhonov_,
                                            self.picos_eps)
        if self.reg_method == 'twonorm':
            problem = _add_twonorm(problem, problem.variables['U_hat'],
                                   self.alpha_other_, self.picos_eps)
        elif self.reg_method == 'nuclear':
            problem = _add_nuclear(problem, problem.variables['U_hat'],
                                   self.alpha_other_, self.picos_eps)
        # Solve optimization problem
        problem.solve(**self.solver_params_)
        # Save solution status
        self.solution_status_ = problem.last_solution.claimedStatus
        # Extract solution from ``Problem`` object
        U_hat = self._extract_solution(problem).T
        U = Q_hat @ U_hat @ linalg.block_diag(Q_hat, np.eye(p - p_theta)).T
        return U.T

    def _validate_parameters(self) -> None:
        # Check problem creation parameters
        self._validate_problem_parameters(self.alpha, self.picos_eps)
        _validate_tsvd_method(self.tsvd_method, manual_len=3)
        # Check regularization methods
        valid_reg_methods = ['tikhonov', 'twonorm', 'nuclear']
        if self.reg_method not in valid_reg_methods:
            raise ValueError('`reg_method` must be one of '
                             f'{valid_reg_methods}.')
        # Check ratio
        if (self.ratio <= 0) or (self.ratio > 1):
            raise ValueError('`ratio` must be positive and less than one.')
        # Check regularization method
        if self.reg_method != 'tikhonov' and self.alpha == 0:
            raise ValueError(
                "`alpha` cannot be zero if `reg_method='twonorm'` or "
                "`reg_method='nuclear'`.")

    @staticmethod
    def _create_base_problem(
        Q_tld: np.ndarray,
        sigma_tld: np.ndarray,
        Z_tld: np.ndarray,
        Q_hat: np.ndarray,
        sigma_hat: np.ndarray,
        Z_hat: np.ndarray,
        alpha_tikhonov: float,
        picos_eps: float,
    ) -> picos.Problem:
        """Create optimization problem."""
        LmiDmdc._validate_problem_parameters(alpha_tikhonov, picos_eps)
        # Compute needed sizes
        p, r_tld = Q_tld.shape
        p_theta, r_hat = Q_hat.shape
        # Compute Q_hat
        Q_bar = linalg.block_diag(Q_hat, np.eye(p - p_theta)).T @ Q_tld
        # Create optimization problem
        problem = picos.Problem()
        # Constants
        Sigma_hat_sq = picos.Constant('Sigma_hat^2', np.diag(sigma_hat**2))
        Sigma_hat = np.diag(sigma_hat)
        Sigma_tld = np.diag(sigma_tld)
        big_constant = picos.Constant(
            'Q_bar Sigma_tld Z_tld.T Z_hat Sigma_Hat',
            Q_bar @ Sigma_tld @ Z_tld.T @ Z_hat @ Sigma_hat,
        )
        Q_bar_Sigma_tld = picos.Constant(
            'Q_bar Sigma_tld',
            Q_bar @ Sigma_tld,
        )
        m1 = picos.Constant('-1', -1 * np.eye(r_tld))
        Sigma_hat = picos.Constant('Sigma_hat', Sigma_hat)
        # Variables
        U_hat = picos.RealVariable('U_hat', (r_hat, r_hat + p - p_theta))
        W_hat = picos.SymmetricVariable('W_hat', (r_hat, r_hat))
        # Constraints
        problem.add_constraint(W_hat >> picos_eps)
        problem.add_constraint(
            picos.block([
                [
                    -W_hat + Sigma_hat_sq - U_hat * big_constant
                    - big_constant.T * U_hat.T, U_hat * Q_bar_Sigma_tld
                ],
                [Q_bar_Sigma_tld.T * U_hat.T, m1],
            ]) << picos_eps)
        problem.set_objective('min', picos.trace(W_hat))
        return problem

    @staticmethod
    def _extract_solution(problem: picos.Problem) -> np.ndarray:
        """Extract solution from an optimization problem."""
        return np.array(problem.get_valued_variable('U_hat'), ndmin=2).T

    @staticmethod
    def _validate_problem_parameters(alpha: float, picos_eps: float) -> None:
        """Validate parameters involved in problem creation."""
        # Validate ``alpha``
        if alpha < 0:
            raise ValueError('Parameter `alpha` must be positive or zero.')
        # Validate ``picos_eps``
        if picos_eps < 0:
            raise ValueError('Parameter `picos_eps` must be positive or zero.')


class LmiEdmdSpectralRadiusConstr(koopman_pipeline.KoopmanRegressor):
    """LMI-based EDMD with spectral radius constraint.

    Optionally supports Tikhonov regularization.

    Attributes
    ----------
    objective_log_ : list[float]
        Objective function history.
    stop_reason_ : str
        Reason iteration stopped.
    n_iter_ : int
        Number of iterations
    Gamma_ : np.ndarray
        ``Gamma`` matrix, for debugging.
    P_ : np.ndarray
        ``P`` matrix, for debugging.
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
                 spectral_radius: float = 1.0,
                 max_iter: int = 100,
                 iter_tol: float = 1e-6,
                 alpha: float = 0,
                 inv_method: str = 'svd',
                 tsvd_method: Union[str, tuple[str, ...]] = 'economy',
                 picos_eps: float = 0,
                 solver_params: dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiEdmdSpectralRadiusConstr`.

        To disable regularization, use ``alpha=0``.

        Parameters
        ----------
        spectral_radius : float
            Maximum spectral radius.

        max_iter : int
            Maximum number of solver iterations.

        iter_tol : float
            Absolute tolerance for change in objective function value. When the
            change in objective function is less than ``iter_tol``, the
            iteration will stop.

        alpha : float
            Regularization coefficient. Can only be zero if
            ``reg_method='tikhonov'``.

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

         tsvd_method : Union[str, tuple[str, ...]]
            Singular value truncation method if ``inv_method='svd'``. Very
            experimental. Possible values are

            - ``'economy'`` or ``('economy', )`` -- use economy SVD without
              truncating singular values
            - ``'unknown_noise'`` or ``('unknown_noise', )`` -- use optimal
              hard truncation [optht]_ with unknown noise to truncate,
            - ``('known_noise', sigma)`` -- use optimal hard truncation
              [optht]_ with known noise ``sigma`` to truncate, or
            - ``('manual', rank)`` -- manually truncate SVDs of ``X_unshifted``
            to ``rank``.

        picos_eps : float
            Tolerance used for strict LMIs. If nonzero, should be larger than
            solver tolerance.

        solver_params : dict[str, Any]
            Parameters passed to PICOS :func:`picos.Problem.solve()`. By
            default, allows chosen solver to select its own tolerances.
        """
        self.spectral_radius = spectral_radius
        self.max_iter = max_iter
        self.iter_tol = iter_tol
        self.alpha = alpha
        self.inv_method = inv_method
        self.tsvd_method = tsvd_method
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Get needed sizes
        p = X_unshifted.shape[1]
        p_theta = X_shifted.shape[1]
        # Make initial guesses and iterate
        Gamma = np.eye(p_theta)
        # Set scope of other variables
        U = np.zeros((p_theta, p))
        P = np.zeros((p_theta, p_theta))
        self.objective_log_ = []
        for k in range(self.max_iter):
            # Formulate Problem A
            problem_a = self._create_problem_a(X_unshifted, X_shifted, Gamma)
            # Solve Problem A
            if polite_stop:
                self.stop_reason_ = 'User requested stop.'
                log.warn(self.stop_reason_)
                break
            log.info(f'Solving problem A{k}')
            problem_a.solve(**self.solver_params_)
            solution_status_a = problem_a.last_solution.claimedStatus
            if solution_status_a != 'optimal':
                self.stop_reason_ = (
                    'Unable to solve `problem_a`. Used last valid `U`. '
                    f'Solution status: `{solution_status_a}`.')
                log.warn(self.stop_reason_)
                break
            U = np.array(problem_a.get_valued_variable('U'), ndmin=2)
            P = np.array(problem_a.get_valued_variable('P'), ndmin=2)
            # Check stopping condition
            self.objective_log_.append(problem_a.value)
            if len(self.objective_log_) > 1:
                diff = np.absolute(self.objective_log_[-2]
                                   - self.objective_log_[-1])
                if (diff < self.iter_tol):
                    self.stop_reason_ = f'Reached tolerance {diff}'
                    break
            # Formulate Problem B
            problem_b = self._create_problem_b(U, P)
            # Solve Problem B
            if polite_stop:
                self.stop_reason_ = 'User requested stop.'
                log.warn(self.stop_reason_)
                break
            log.info(f'Solving problem B{k}')
            problem_b.solve(**self.solver_params_)
            solution_status_b = problem_b.last_solution.claimedStatus
            if solution_status_b != 'optimal':
                self.stop_reason_ = (
                    'Unable to solve `problem_b`. Used last valid `U`. '
                    f'Solution status: `{solution_status_b}`.')
                log.warn(self.stop_reason_)
                break
            Gamma = np.array(problem_b.get_valued_variable('Gamma'), ndmin=2)
        else:
            self.stop_reason_ = f'Reached maximum iterations {self.max_iter}'
            log.warn(self.stop_reason_)
        self.n_iter_ = k + 1
        coef = U.T
        # Only useful for debugging
        self.Gamma_ = Gamma
        self.P_ = P
        return coef

    def _validate_parameters(self) -> None:
        # Check problem creation parameters
        LmiEdmd._validate_problem_parameters(self.alpha, self.inv_method,
                                             self.tsvd_method, self.picos_eps)
        # Check spectral radius
        if self.spectral_radius <= 0:
            raise ValueError('`spectral_radius` must be positive.')
        if self.max_iter <= 0:
            raise ValueError('`max_iter` must be positive.')
        if self.iter_tol <= 0:
            raise ValueError('`iter_tol` must be positive.')

    def _create_problem_a(self, X_unshifted: np.ndarray, X_shifted: np.ndarray,
                          Gamma: np.ndarray) -> picos.Problem:
        """Create first problem in iteration scheme."""
        q = X_unshifted.shape[0]
        problem_a = LmiEdmd._create_base_problem(X_unshifted, X_shifted,
                                                 self.alpha / q,
                                                 self.inv_method,
                                                 self.tsvd_method,
                                                 self.picos_eps)
        # Extract information from problem
        U = problem_a.variables['U']
        # Get needed sizes
        p_theta = U.shape[0]
        # Add new constraints
        rho_bar_sq = picos.Constant('rho_bar^2', self.spectral_radius**2)
        Gamma = picos.Constant('Gamma', Gamma)
        P = picos.SymmetricVariable('P', p_theta)
        problem_a.add_constraint(P >> self.picos_eps)
        problem_a.add_constraint(
            picos.block([
                [rho_bar_sq * P, U[:, :p_theta].T * Gamma],
                [Gamma.T * U[:, :p_theta], Gamma + Gamma.T - P],
            ]) >> self.picos_eps)
        return problem_a

    def _create_problem_b(self, U: np.ndarray, P: np.ndarray) -> picos.Problem:
        """Create second problem in iteration scheme."""
        # Create optimization problem
        problem_b = picos.Problem()
        # Get needed sizes
        p_theta = U.shape[0]
        # Create constants
        rho_bar_sq = picos.Constant('rho_bar^2', self.spectral_radius**2)
        U = picos.Constant('U', U)
        P = picos.Constant('P', P)
        # Create variables
        Gamma = picos.RealVariable('Gamma', P.shape)
        # Add constraints
        problem_b.add_constraint(
            picos.block([
                [rho_bar_sq * P, U[:, :p_theta].T * Gamma],
                [Gamma.T * U[:, :p_theta], Gamma + Gamma.T - P],
            ]) >> self.picos_eps)
        # Set objective
        problem_b.set_objective('find')
        return problem_b


class LmiEdmdHinfReg(koopman_pipeline.KoopmanRegressor):
    """LMI-based EDMD with H-infinity norm regularization.

    Optionally supports additional Tikhonov regularization.

    Attributes
    ----------
    objective_log_ : list[float]
        Objective function history.
    stop_reason_ : str
        Reason iteration stopped.
    n_iter_ : int
        Number of iterations
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
                 alpha: float = 1,
                 ratio: float = 1,
                 weight: tuple[str, np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray] = None,
                 max_iter: int = 100,
                 iter_tol: float = 1e-6,
                 inv_method: str = 'svd',
                 tsvd_method: Union[str, tuple[str, ...]] = 'economy',
                 picos_eps: float = 0,
                 solver_params: dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiEdmdHinfReg`.

        Supports cascading the plant with an LTI weighting function.

        Parameters
        ----------
        alpha : float
            Regularization coefficient. Cannot be zero.

        ratio : float
            Ratio of H-infinity norm to use in mixed regularization. If
            ``ratio=1``, no Tikhonov regularization is used. Cannot be zero.

        weight : tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing weight type (``'pre'`` or ``'post'``), and the
            weight state space matrices (``A``, ``B``, ``C``, and ``D``). If
            ``None``, no weighting is used.

        max_iter : int
            Maximum number of solver iterations.

        iter_tol : float
            Absolute tolerance for change in objective function value. When the
            change in objective function is less than ``iter_tol``, the
            iteration will stop.

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

         tsvd_method : Union[str, tuple[str, ...]]
            Singular value truncation method if ``inv_method='svd'``. Very
            experimental. Possible values are

            - ``'economy'`` or ``('economy', )`` -- use economy SVD without
              truncating singular values
            - ``'unknown_noise'`` or ``('unknown_noise', )`` -- use optimal
              hard truncation [optht]_ with unknown noise to truncate,
            - ``('known_noise', sigma)`` -- use optimal hard truncation
              [optht]_ with known noise ``sigma`` to truncate, or
            - ``('manual', rank)`` -- manually truncate SVDs of ``X_unshifted``
            to ``rank``.

        picos_eps : float
            Tolerance used for strict LMIs. If nonzero, should be larger than
            solver tolerance.

        solver_params : dict[str, Any]
            Parameters passed to PICOS :func:`picos.Problem.solve()`. By
            default, allows chosen solver to select its own tolerances.
        """
        self.alpha = alpha
        self.ratio = ratio
        self.weight = weight
        self.max_iter = max_iter
        self.iter_tol = iter_tol
        self.inv_method = inv_method
        self.tsvd_method = tsvd_method
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Set regularization coefficients
        self.alpha_tikhonov_ = self.alpha * (1 - self.ratio)
        self.alpha_other_ = self.alpha * self.ratio
        # Get needed sizes
        p = X_unshifted.shape[1]
        p_theta = X_shifted.shape[1]
        # Check that at least one input is present
        if p_theta == p:
            # If you remove the `{p} features(s)` part of this message,
            # the scikit-learn estimator_checks will fail!
            raise ValueError('LmiEdmdHinfReg() requires an input to function.'
                             '`X` and `y` must therefore have different '
                             'numbers of features. `X and y` both have '
                             f'{p} feature(s).')
        # Set up weights
        if self.weight is None:
            P = np.eye(p_theta)
        elif self.weight[0] == 'pre':
            n_u = p - p_theta
            P = np.eye(p_theta + n_u * self.weight[1].shape[0])
        elif self.weight[0] == 'post':
            n_x = p_theta
            P = np.eye(p_theta + n_x * self.weight[1].shape[0])
        else:
            # Already checked. Should never get here.
            assert False
        # Solve optimization problem iteratively
        U = np.zeros((p_theta, p))
        gamma = np.zeros((1, ))
        self.objective_log_ = []
        for k in range(self.max_iter):
            # Formulate Problem A
            problem_a = self._create_problem_a(X_unshifted, X_shifted, P)
            # Solve Problem A
            if polite_stop:
                self.stop_reason_ = 'User requested stop.'
                log.warn(self.stop_reason_)
                break
            log.info(f'Solving problem A{k}')
            problem_a.solve(**self.solver_params_)
            solution_status_a = problem_a.last_solution.claimedStatus
            if solution_status_a != 'optimal':
                self.stop_reason_ = (
                    'Unable to solve `problem_a`. Used last valid `U`. '
                    f'Solution status: `{solution_status_a}`.')
                log.warn(self.stop_reason_)
                break
            U = np.array(problem_a.get_valued_variable('U'), ndmin=2)
            gamma = np.array(problem_a.get_valued_variable('gamma'))
            self.objective_log_.append(problem_a.value)
            if len(self.objective_log_) > 1:
                diff = np.absolute(self.objective_log_[-2]
                                   - self.objective_log_[-1])
                if (diff < self.iter_tol):
                    self.stop_reason_ = f'Reached tolerance {diff}'
                    break
            # Formulate Problem B
            problem_b = self._create_problem_b(U, gamma)
            # Solve Problem B
            if polite_stop:
                self.stop_reason_ = 'User requested stop.'
                log.warn(self.stop_reason_)
                break
            log.info(f'Solving problem B{k}')
            problem_b.solve(**self.solver_params_)
            solution_status_b = problem_b.last_solution.claimedStatus
            if solution_status_b != 'optimal':
                self.stop_reason_ = (
                    'Unable to solve `problem_b`. Used last valid `U`. '
                    'Solution status: f`{solution_status_b}`.')
                log.warn(self.stop_reason_)
                break
            P = np.array(problem_b.get_valued_variable('P'), ndmin=2)
        else:
            self.stop_reason_ = f'Reached maximum iterations {self.max_iter}'
            log.warn(self.stop_reason_)
        self.n_iter_ = k + 1
        coef = U.T
        # Only useful for debugging
        self.P_ = P
        self.gamma_ = gamma
        return coef

    def _validate_parameters(self) -> None:
        # Check problem creation parameters
        LmiEdmd._validate_problem_parameters(self.alpha, self.inv_method,
                                             self.tsvd_method, self.picos_eps)
        # Check other parameters
        if self.alpha <= 0:
            # Check alpha again because ``_validate_problem_parameters`` allows
            # zero alpha.
            raise ValueError('`alpha` must be positive.')
        if (self.ratio <= 0) or (self.ratio > 1):
            raise ValueError('`ratio` must be positive and less than one.')
        valid_weight_types = ['pre', 'post']
        if self.weight is not None:
            if self.weight[0] not in valid_weight_types:
                raise ValueError('First element of the `weight` must be one'
                                 f'of {valid_weight_types}.')
        if self.max_iter <= 0:
            raise ValueError('`max_iter` must be positive.')
        if self.iter_tol <= 0:
            raise ValueError('`iter_tol` must be positive.')

    def _create_problem_a(self, X_unshifted: np.ndarray, X_shifted: np.ndarray,
                          P: np.ndarray) -> picos.Problem:
        """Create first problem in iteration scheme."""
        q = X_unshifted.shape[0]
        problem_a = LmiEdmd._create_base_problem(X_unshifted, X_shifted,
                                                 self.alpha_tikhonov_ / q,
                                                 self.inv_method,
                                                 self.tsvd_method,
                                                 self.picos_eps)
        # Extract information from problem
        U = problem_a.variables['U']
        direction = problem_a.objective.direction
        objective = problem_a.objective.function
        # Get needed sizes
        p_theta = U.shape[0]
        # Add new constraint
        P = picos.Constant('P', P)
        gamma = picos.RealVariable('gamma', 1)
        # Get weighted state space matrices
        A, B, C, D = self._create_ss(U)
        gamma_33 = picos.diag(gamma, D.shape[1])
        gamma_44 = picos.diag(gamma, D.shape[0])
        problem_a.add_constraint(
            picos.block([
                [P,          A * P,    B,         0],
                [P.T * A.T,  P,        0,         P * C.T],
                [B.T,        0,        gamma_33,  D.T],
                [0,          C * P.T,  D,         gamma_44],
            ]) >> self.picos_eps)  # yapf: disable
        # Add term to cost function
        alpha_scaled = picos.Constant('alpha_inf/q', self.alpha_other_ / q)
        objective += alpha_scaled * gamma
        problem_a.set_objective(direction, objective)
        return problem_a

    def _create_problem_b(self, U: np.ndarray,
                          gamma: np.ndarray) -> picos.Problem:
        """Create second problem in iteration scheme."""
        # Create optimization problem
        problem_b = picos.Problem()
        # Get needed sizes
        p_theta = U.shape[0]
        # Create constants
        U = picos.Constant('U', U)
        gamma = picos.Constant('gamma', gamma)
        # Get weighted state space matrices
        A, B, C, D = self._create_ss(U)
        # Create variables
        P = picos.SymmetricVariable('P', A.shape[0])
        # Add constraints
        problem_b.add_constraint(P >> self.picos_eps)
        gamma_33 = picos.diag(gamma, D.shape[1])
        gamma_44 = picos.diag(gamma, D.shape[0])
        problem_b.add_constraint(
            picos.block([
                [P,          A * P,    B,         0],
                [P.T * A.T,  P,        0,         P * C.T],
                [B.T,        0,        gamma_33,  D.T],
                [0,          C * P.T,  D,         gamma_44],
            ]) >> self.picos_eps)  # yapf: disable
        # Set objective
        problem_b.set_objective('find')
        return problem_b

    def _create_ss(
        self, U: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Augment Koopman system with weight if present.

        Parameters
        ----------
        U : np.ndarray
            Koopman matrix containing ``A`` and ``B`` concatenated
            horizontally.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Weighted state space matrices (``A``, ``B``, ``C``, ``D``).
        """
        p_theta = U.shape[0]
        if self.weight is None:
            A = U[:, :p_theta]
            B = U[:, p_theta:]
            C = picos.Constant('C', np.eye(p_theta))
            D = picos.Constant('D', np.zeros((C.shape[0], B.shape[1])))
        else:
            Am = U[:, :p_theta]
            Bm = U[:, p_theta:]
            Cm = picos.Constant('Cm', np.eye(p_theta))
            Dm = picos.Constant('Dm', np.zeros((Cm.shape[0], Bm.shape[1])))
            if self.weight_type_ == 'pre':
                n_u = Bm.shape[1]
                Aw_blk = linalg.block_diag(*([self.weight[1]] * n_u))
                Bw_blk = linalg.block_diag(*([self.weight[2]] * n_u))
                Cw_blk = linalg.block_diag(*([self.weight[3]] * n_u))
                Dw_blk = linalg.block_diag(*([self.weight[4]] * n_u))
                Aw = picos.Constant('Aw', Aw_blk)
                Bw = picos.Constant('Bw', Bw_blk)
                Cw = picos.Constant('Cw', Cw_blk)
                Dw = picos.Constant('Dw', Dw_blk)
                A = picos.block([
                    [Aw, 0],
                    [Bm * Cw, Am],
                ])
                B = picos.block([
                    [Bw],
                    [Bm * Dw],
                ])
                C = picos.block([
                    [Dm * Cw, Cm],
                ])
                D = Dm * Dw
            elif self.weight_type_ == 'post':
                n_x = Bm.shape[0]
                Aw_blk = linalg.block_diag(*([self.weight[1]] * n_x))
                Bw_blk = linalg.block_diag(*([self.weight[2]] * n_x))
                Cw_blk = linalg.block_diag(*([self.weight[3]] * n_x))
                Dw_blk = linalg.block_diag(*([self.weight[4]] * n_x))
                Aw = picos.Constant('Aw', Aw_blk)
                Bw = picos.Constant('Bw', Bw_blk)
                Cw = picos.Constant('Cw', Cw_blk)
                Dw = picos.Constant('Dw', Dw_blk)
                A = picos.block([
                    [Am, 0],
                    [Bw * Cm, Aw],
                ])
                B = picos.block([
                    [Bm],
                    [Bw * Dm],
                ])
                C = picos.block([
                    [Dw * Cm, Cw],
                ])
                D = Dw * Dm
            else:
                # Already checked, should not get here.
                assert False
        return (A, B, C, D)


def _add_twonorm(problem: picos.Problem, U: picos.RealVariable,
                 alpha_other: float, picos_eps: float) -> picos.Problem:
    """Add matrix two norm regularizer to an optimization problem.

    Parameters
    ----------
    problem : picos.Problem
        Optimization problem.
    U : picos.RealVariable
        Koopman matrix variable.
    alpha_other : float
        Regularization coefficient (already divided by ``q`` if applicable).
    picos_eps : float
        Tolerance used for strict LMIs.

    Returns
    -------
    picos.Problem
        Optimization problem with regularizer added.
    """
    # Extract information from problem
    direction = problem.objective.direction
    objective = problem.objective.function
    # Get needed sizes
    p_theta, p = U.shape
    # Add new constraint
    gamma = picos.RealVariable('gamma', 1)
    problem.add_constraint(
        picos.block([[picos.diag(gamma, p), U.T],
                     [U, picos.diag(gamma, p_theta)]]) >> 0)
    # Add term to cost function
    alpha_scaled = picos.Constant('alpha_scaled_2', alpha_other)
    objective += alpha_scaled * gamma
    problem.set_objective(direction, objective)
    return problem


def _add_nuclear(problem: picos.Problem, U: picos.RealVariable,
                 alpha_other: float, picos_eps: float) -> picos.Problem:
    """Add nuclear norm regularizer to an optimization problem.

    Parameters
    ----------
    problem : picos.Problem
        Optimization problem.
    U : picos.RealVariable
        Koopman matrix variable.
    alpha_other : float
        Regularization coefficient (already divided by ``q`` if applicable).
    picos_eps : float
        Tolerance used for strict LMIs.

    Returns
    -------
    picos.Problem
        Optimization problem with regularizer added.
    """
    # Extract information from problem
    direction = problem.objective.direction
    objective = problem.objective.function
    # Get needed sizes
    p_theta, p = U.shape
    # Add new constraint
    gamma = picos.RealVariable('gamma', 1)
    W_1 = picos.SymmetricVariable('W_1', (p_theta, p_theta))
    W_2 = picos.SymmetricVariable('W_2', (p, p))
    problem.add_constraint(picos.trace(W_1) + picos.trace(W_2) <= 2 * gamma)
    problem.add_constraint(picos.block([[W_1, U], [U.T, W_2]]) >> 0)
    # Add term to cost function
    alpha_scaled = picos.Constant('alpha_scaled_*', alpha_other)
    objective += alpha_scaled * gamma
    problem.set_objective(direction, objective)
    return problem


def _validate_tsvd_method(tsvd_method: Union[str, tuple[str, ...]],
                          manual_len: int) -> None:
    """Validate ``tsvd_method``.

    Parameters
    ----------
    tsvd_method : Union[str, tuple[str, ...]]
        Singular value truncation method if ``inv_method='svd'``.
    manual_len : int
        Number of ranks that need to be specified in the ``'manual'`` tuple.
    """
    # Value is required tuple length
    valid_tsvd_methods = {
        'economy': 1,
        'unknown_noise': 1,
        'known_noise': 2,
        'manual': manual_len,
    }
    # String representation of options
    valid_tsvd_methods_str = [
        "'economy'",
        "'unknown_noise'",
        "('economy', )",
        "('unknown_noise', )",
        "('known_noise', `sigma`)",
    ]
    # Add last option depending on ``manual_len``
    if manual_len == 2:
        valid_tsvd_methods_str.append("('manual', `rank`)")
    elif manual_len == 3:
        valid_tsvd_methods_str.append(
            "('manual', `rank_unshifted`, `rank_shifted`)")
    else:
        raise NotImplementedError('Not a valid ``manual_len``. Add its '
                                  'string representation here?')
    if type(tsvd_method) is str:
        if tsvd_method not in ['economy', 'unknown_noise']:
            raise ValueError('`tsvd_method` must be one of '
                             f'{valid_tsvd_methods_str}.')
    else:
        if tsvd_method[0] not in valid_tsvd_methods:
            raise ValueError('`tsvd_method` must be one of '
                             f'{valid_tsvd_methods_str}.')
        if (len(tsvd_method) != valid_tsvd_methods[tsvd_method[0]]):
            raise ValueError('Wrong number of tuple elements in '
                             '`tsvd_method`. Must be one of'
                             f'{valid_tsvd_methods_str}.')


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
        Tikhonov regularization coefficient (divided by ``q``).

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
    # ``alpha`` is already divided by ``q`` to be consistent with ``G`` and
    # ``H``
    H_reg = H_unreg + (alpha * np.eye(p))
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
def _calc_QSig(X: np.ndarray, alpha: float,
               tsvd_method: Union[str, tuple[str, ...]]) -> np.ndarray:
    """Split ``H`` using the truncated SVD of ``X``.

    ``H`` is defined as:

        H = 1/q * X_unshifted @ X_unshifted.T

    Consider the SVD::

        X_unshifted = Q @ Sig @ V.T

    Without regularization, ``H`` is then::

        H = 1/q * Q @ Sig**2 @ Q.T
          = Q @ (Sig**2 / q) @ Q.T
          = (Q @ sqrt(Sig**2 / q)) @ (sqrt(Sig**2 / q) @ Q.T)

    With regularization::

        H = (Q @ sqrt(Sig**2 / q + alpha)) @ (sqrt(Sig**2 / q + alpha) @ Q.T)

    Parameters
    ----------
    X : np.ndarray
        ``X``, where ``H = 1/q * X @ X.T``.
    alpha : float
        Tikhonov regularization coefficient (divided by ``q``).
     tsvd_method : Union[str, tuple[str, ...]]
        Singular value truncation method if ``inv_method='svd'``.

    Returns
    -------
    np.ndarray
        Split ``H`` matrix.
    """
    # SVD
    Q, s, _ = linalg.svd(X.T, full_matrices=False)
    # Compute truncation rank
    if ((tsvd_method == 'economy') or (tsvd_method[0] == 'economy')):
        r = s.shape[0]
    elif ((tsvd_method == 'unknown_noise')
          or (tsvd_method[0] == 'unknown_noise')):
        r = optht.optht(X.T, s)
    elif tsvd_method[0] == 'known_noise':
        variance = tsvd_method[1]
        r = optht.optht(X.T, s, variance)
    elif tsvd_method[0] == 'manual':
        r = tsvd_method[1]
    else:
        # Already checked
        assert False
    # Truncate
    Qr = Q[:, :r]
    sr = s[:r]
    # Regularize
    q = X.shape[0]
    # ``alpha`` is already divided by ``q`` to be consistent with ``G`` and
    # ``H``.
    sr_reg = np.sqrt((sr**2 / q) + alpha)
    Sr_reg = np.diag(sr_reg)
    # Multiply with Q and return
    QSig = Qr @ Sr_reg
    log.info(f'_calc_QSig() stats: r={r}, alpha={alpha}, len(s)={len(s)}, '
             f's[0]={s[0]}, s[-1]={s[-1]}')
    return QSig
