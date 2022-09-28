"""Experimental LMI-based Koopman regressors from [DF21]_ and [DF22]_.

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
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import picos
import scipy.signal
import sklearn.base
from scipy import linalg

from . import koopman_pipeline, regressors, tsvd

# Create logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# Create temporary cache directory for memoized computations
_cachedir = tempfile.TemporaryDirectory(prefix='pykoop_')
log.info(f'Temporary directory created at `{_cachedir.name}`')
memory = joblib.Memory(_cachedir.name, verbose=0)

# Create signal handler to politely stop computations
polite_stop = False


def _sigint_handler(sig, frame):
    """Signal handler for ^C."""
    global polite_stop
    if not polite_stop:
        print('Stop requested. Regression will stop safely at next iteration. '
              'Press ^C again to force quit.')
        polite_stop = True
    else:
        print('Force quitting now.')
        exit()


signal.signal(signal.SIGINT, _sigint_handler)


class LmiRegressor(koopman_pipeline.KoopmanRegressor):
    """Base class for LMI regressors.

    For derivations of LMIs, see [DF21]_ and [DF22]_.

    This base class is mostly used to share common ``scikit-learn`` parameters.

    Attributes
    ----------
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    coef_ : np.ndarray
        Fit coefficient matrix.
    """

    # Default solver parameters
    _default_solver_params: Dict[str, Any] = {
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
    _check_X_y_params: Dict[str, Any] = {
        'multi_output': True,
        'y_numeric': True,
        'dtype': 'float64',
    }

    def _more_tags(self):
        reason = ('Hard to guarantee exact idempotence when calling external '
                  'solver.')
        return {
            'multioutput': True,
            'multioutput_only': True,
            '_xfail_checks': {
                'check_fit_idempotent': reason,
            }
        }


class LmiEdmd(LmiRegressor):
    """LMI-based EDMD with regularization.

    Supports Tikhonov regularization, optionally mixed with matrix two-norm
    regularization or nuclear norm regularization.

    Attributes
    ----------
    alpha_tikhonov_ : float
        Tikhonov regularization coefficient used.
    alpha_other_ : float
        Matrix two norm or nuclear norm regularization coefficient used.
    tsvd_ : pykoop.Tsvd
        Fit truncated SVD object.
    solver_params_ : Dict[str, Any]
        Solver parameters used (defaults merged with constructor input).
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    coef_ : np.ndarray
        Fit coefficient matrix.

    Examples
    --------
    LMI EDMD without regularization

    >>> kp = pykoop.KoopmanPipeline(regressor=pykoop.lmi_regressors.LmiEdmd())
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
    KoopmanPipeline(regressor=LmiEdmd())

    LMI EDMD with Tikhonov regularization

    >>> kp = pykoop.KoopmanPipeline(
    ...     regressor=pykoop.lmi_regressors.LmiEdmd(
    ...         alpha=1,
    ...         reg_method='tikhonov',
    ...     )
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
    KoopmanPipeline(regressor=LmiEdmd(alpha=1))

    LMI EDMD with matrix two-norm regularization

    >>> kp = pykoop.KoopmanPipeline(
    ...     regressor=pykoop.lmi_regressors.LmiEdmd(
    ...         alpha=1,
    ...         reg_method='twonorm',
    ...     )
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
    KoopmanPipeline(regressor=LmiEdmd(alpha=1, reg_method='twonorm'))

    LMI EDMD with mixed Tikhonov and squared-nuclear-norm regularization

    >>> kp = pykoop.KoopmanPipeline(
    ...     regressor=pykoop.lmi_regressors.LmiEdmd(
    ...         alpha=1,
    ...         ratio=0.5,
    ...         reg_method='nuclear',
    ...         square_norm=True,
    ...     )
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
    KoopmanPipeline(regressor=LmiEdmd(alpha=1, ratio=0.5, reg_method='nuclear',
    square_norm=True))
    """

    def __init__(self,
                 alpha: float = 0,
                 ratio: float = 1,
                 reg_method: str = 'tikhonov',
                 inv_method: str = 'svd',
                 tsvd: tsvd.Tsvd = None,
                 square_norm: bool = False,
                 picos_eps: float = 0,
                 solver_params: Dict[str, Any] = None) -> None:
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

        tsvd : pykoop.Tsvd()
            Singular value truncation method if ``inv_method='svd'``. If
            ``None``, economy SVD is used.

        square_norm : bool
            Square norm in matrix two-norm or nuclear norm regularizer.
            Enabling may increase computation time. Frobenius norm used in
            Tikhonov regularizer is always squared.

        picos_eps : float
            Tolerance used for strict LMIs. If nonzero, should be larger than
            solver tolerance.

        solver_params : Dict[str, Any]
            Parameters passed to PICOS :func:`picos.Problem.solve()`. By
            default, allows chosen solver to select its own tolerances.
        """
        self.alpha = alpha
        self.ratio = ratio
        self.reg_method = reg_method
        self.inv_method = inv_method
        self.tsvd = tsvd
        self.square_norm = square_norm
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
        # Clone TSVD
        self.tsvd_ = (sklearn.base.clone(self.tsvd)
                      if self.tsvd is not None else tsvd.Tsvd())
        # Form optimization problem. Regularization coefficients must be scaled
        # because of how G and H are defined.
        q = X_unshifted.shape[0]
        problem = self._create_base_problem(X_unshifted, X_shifted,
                                            self.alpha_tikhonov_ / q,
                                            self.inv_method, self.tsvd_,
                                            self.picos_eps)
        if self.reg_method == 'twonorm':
            problem = _add_twonorm(problem, problem.variables['U'],
                                   self.alpha_other_ / q, self.square_norm,
                                   self.picos_eps)
        elif self.reg_method == 'nuclear':
            problem = _add_nuclear(problem, problem.variables['U'],
                                   self.alpha_other_ / q, self.square_norm,
                                   self.picos_eps)
        # Solve optimization problem
        problem.solve(**self.solver_params_)
        # Save solution status
        self.solution_status_ = problem.last_solution.claimedStatus
        # Extract solution from ``Problem`` object
        coef = self._extract_solution(problem)
        return coef

    def _validate_parameters(self) -> None:
        # Check regularization methods
        valid_reg_methods = ['tikhonov', 'twonorm', 'nuclear']
        if self.reg_method not in valid_reg_methods:
            raise ValueError('`reg_method` must be one of '
                             f'{valid_reg_methods}.')
        # Check ratio
        if (self.ratio <= 0) or (self.ratio > 1):
            raise ValueError('`ratio` must be positive and less than one.')

    @staticmethod
    def _create_base_problem(
        X_unshifted: np.ndarray,
        X_shifted: np.ndarray,
        alpha_tikhonov: float,
        inv_method: str,
        tsvd: tsvd.Tsvd,
        picos_eps: float,
    ) -> picos.Problem:
        """Create optimization problem."""
        # Validate ``alpha``
        if alpha_tikhonov < 0:
            raise ValueError('Parameter `alpha` must be positive or zero.')
        # Validate ``inv_method``
        valid_inv_methods = [
            'inv', 'pinv', 'eig', 'ldl', 'chol', 'sqrt', 'svd'
        ]
        if inv_method not in valid_inv_methods:
            raise ValueError('`inv_method` must be one of '
                             f'{valid_inv_methods}.')
        # Validate ``picos_eps``
        if picos_eps < 0:
            raise ValueError('Parameter `picos_eps` must be positive or zero.')
        # Compute ``G`` and ``H``. ``alpha_tikhonov`` must already be scaled
        # by ``q`` if applicable.
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
            problem.add_constraint(
                picos.block([
                    [Z, U],
                    [U.T, H_inv],
                ]) >> picos_eps)
        elif inv_method == 'pinv':
            H_inv = picos.Constant('H^+', _calc_Hpinv(H))
            problem.add_constraint(
                picos.block([
                    [Z, U],
                    [U.T, H_inv],
                ]) >> picos_eps)
        elif inv_method == 'eig':
            VsqrtLmb = picos.Constant('(V Lambda^(1/2))', _calc_VsqrtLmb(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * VsqrtLmb],
                    [VsqrtLmb.T * U.T, 'I'],
                ]) >> picos_eps)
        elif inv_method == 'ldl':
            LsqrtD = picos.Constant('(L D^(1/2))', _calc_LsqrtD(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * LsqrtD],
                    [LsqrtD.T * U.T, 'I'],
                ]) >> picos_eps)
        elif inv_method == 'chol':
            L = picos.Constant('L', _calc_L(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * L],
                    [L.T * U.T, 'I'],
                ]) >> picos_eps)
        elif inv_method == 'sqrt':
            sqrtH = picos.Constant('sqrt(H)', _calc_sqrtH(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * sqrtH],
                    [sqrtH.T * U.T, 'I'],
                ]) >> picos_eps)
        elif inv_method == 'svd':
            QSig = picos.Constant(
                'Q Sigma', _calc_QSig(X_unshifted, alpha_tikhonov, tsvd))
            problem.add_constraint(
                picos.block([
                    [Z, U * QSig],
                    [QSig.T * U.T, 'I'],
                ]) >> picos_eps)
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


class LmiDmdc(LmiRegressor):
    """LMI-based DMDc with regularization.

    Supports Tikhonov regularization, optionally mixed with matrix two-norm
    regularization or nuclear norm regularization.

    Attributes
    ----------
    alpha_tikhonov_ : float
        Tikhonov regularization coefficient used.
    alpha_other_ : float
        Matrix two norm or nuclear norm regularization coefficient used.
    tsvd_unshifted_ : pykoop.Tsvd
        Fit truncated SVD object for unshifted data matrix.
    tsvd_shifted_ : pykoop.Tsvd
        Fit truncated SVD object for shifted data matrix.
    U_hat_ : np.ndarray
        Reduced Koopman matrix for debugging.
    solver_params_ : Dict[str, Any]
        Solver parameters used (defaults merged with constructor input).
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    coef_ : np.ndarray
        Fit coefficient matrix.

    Examples
    --------
    LMI DMDc without regularization

    >>> kp = pykoop.KoopmanPipeline(regressor=pykoop.lmi_regressors.LmiDmdc())
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
    KoopmanPipeline(regressor=LmiDmdc())

    LMI DMDc with Tikhonov regularization

    >>> kp = pykoop.KoopmanPipeline(
    ...     regressor=pykoop.lmi_regressors.LmiDmdc(
    ...         alpha=1,
    ...         reg_method='tikhonov',
    ...     )
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
    KoopmanPipeline(regressor=LmiDmdc(alpha=1))

    LMI DMDc with matrix two-norm regularization

    >>> kp = pykoop.KoopmanPipeline(
    ...     regressor=pykoop.lmi_regressors.LmiDmdc(
    ...         alpha=1,
    ...         reg_method='twonorm',
    ...     )
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
    KoopmanPipeline(regressor=LmiDmdc(alpha=1, reg_method='twonorm'))

    LMI DMDc with nuclear norm regularization and SVD truncation

    >>> kp = pykoop.KoopmanPipeline(
    ...     regressor=pykoop.lmi_regressors.LmiDmdc(
    ...         alpha=1,
    ...         reg_method='nuclear',
    ...         tsvd_unshifted=pykoop.Tsvd('known_noise', 0.1),
    ...         tsvd_shifted=pykoop.Tsvd('known_noise', 0.1),
    ...     )
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
    KoopmanPipeline(regressor=LmiDmdc(alpha=1, reg_method='nuclear',
    tsvd_shifted=Tsvd(truncation='known_noise', truncation_param=0.1),
    tsvd_unshifted=Tsvd(truncation='known_noise', truncation_param=0.1)))
    """

    def __init__(self,
                 alpha: float = 0,
                 ratio: float = 1,
                 tsvd_unshifted: tsvd.Tsvd = None,
                 tsvd_shifted: tsvd.Tsvd = None,
                 reg_method: str = 'tikhonov',
                 square_norm: bool = False,
                 picos_eps: float = 0,
                 solver_params: Dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiDmdc`.

        Parameters
        ----------
        alpha : float
            Regularization coefficient. Can only be zero if
            ``reg_method='tikhonov'``.

        ratio : float
            Ratio of matrix two-norm or nuclear norm to use in mixed
            regularization. If ``ratio=1``, no Tikhonov regularization is
            used. Cannot be zero. Ignored if ``reg_method='tikhonov'``.

        tsvd_unshifted : pykoop.Tsvd
            Singular value truncation method used to change basis of unshifted
            data matrix. If ``None``, economy SVD is used.

        tsvd_shifted : pykoop.Tsvd
            Singular value truncation method used to change basis of shifted
            data matrix. If ``None``, economy SVD is used.

        reg_method : str
            Regularization method to use. Possible values are

            - ``'tikhonov'`` -- pure Tikhonov regularization (``ratio``
              is ignored),
            - ``'twonorm'`` -- matrix two-norm regularization mixed with
              Tikhonov regularization, or
            - ``'nuclear'`` -- nuclear norm regularization mixed with Tikhonov
              regularization.

        square_norm : bool
            Square norm in matrix two-norm or nuclear norm regularizer.
            Enabling may increase computation time. Frobenius norm used in
            Tikhonov regularizer is always squared.

        picos_eps : float
            Tolerance used for strict LMIs. If nonzero, should be larger than
            solver tolerance.

        solver_params : Dict[str, Any]
            Parameters passed to PICOS :func:`picos.Problem.solve()`. By
            default, allows chosen solver to select its own tolerances.
        """
        self.alpha = alpha
        self.ratio = ratio
        self.tsvd_unshifted = tsvd_unshifted
        self.tsvd_shifted = tsvd_shifted
        self.reg_method = reg_method
        self.square_norm = square_norm
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
        # Clone TSVDs
        self.tsvd_unshifted_ = (sklearn.base.clone(self.tsvd_unshifted)
                                if self.tsvd_unshifted is not None else
                                tsvd.Tsvd())
        self.tsvd_shifted_ = (sklearn.base.clone(self.tsvd_shifted) if
                              self.tsvd_shifted is not None else tsvd.Tsvd())
        # Compute SVDs
        self.tsvd_unshifted_.fit(X_unshifted.T)
        Q_tld = self.tsvd_unshifted_.left_singular_vectors_
        sig_tld = self.tsvd_unshifted_.singular_values_
        Z_tld = self.tsvd_unshifted_.right_singular_vectors_
        self.tsvd_shifted_.fit(X_shifted.T)
        Q_hat = self.tsvd_shifted_.left_singular_vectors_
        sig_hat = self.tsvd_shifted_.singular_values_
        Z_hat = self.tsvd_shifted_.right_singular_vectors_
        # Form optimization problem
        problem = self._create_base_problem(Q_tld, sig_tld, Z_tld, Q_hat,
                                            sig_hat, Z_hat,
                                            self.alpha_tikhonov_ / q,
                                            self.picos_eps)
        if self.reg_method == 'twonorm':
            problem = _add_twonorm(problem, problem.variables['U_hat'],
                                   self.alpha_other_ / q, self.square_norm,
                                   self.picos_eps)
        elif self.reg_method == 'nuclear':
            problem = _add_nuclear(problem, problem.variables['U_hat'],
                                   self.alpha_other_ / q, self.square_norm,
                                   self.picos_eps)
        # Solve optimization problem
        problem.solve(**self.solver_params_)
        # Save solution status
        self.solution_status_ = problem.last_solution.claimedStatus
        # Extract solution from ``Problem`` object
        U_hat = self._extract_solution(problem).T
        # Save SVDs and reduced U for debugging
        self.U_hat_ = U_hat
        # Reconstruct Koopman operator
        p_upsilon = p - p_theta
        U = Q_hat @ U_hat @ linalg.block_diag(Q_hat, np.eye(p_upsilon)).T
        coef = U.T
        return coef

    def _validate_parameters(self) -> None:
        # Check regularization methods
        valid_reg_methods = ['tikhonov', 'twonorm', 'nuclear']
        if self.reg_method not in valid_reg_methods:
            raise ValueError('`reg_method` must be one of '
                             f'{valid_reg_methods}.')
        # Check ratio
        if (self.ratio <= 0) or (self.ratio > 1):
            raise ValueError('`ratio` must be positive and less than one.')

    @staticmethod
    def _create_base_problem(
        Q_tld: np.ndarray,
        sig_tld: np.ndarray,
        Z_tld: np.ndarray,
        Q_hat: np.ndarray,
        sig_hat: np.ndarray,
        Z_hat: np.ndarray,
        alpha_tikhonov: float,
        picos_eps: float,
    ) -> picos.Problem:
        """Create optimization problem."""
        # Validate ``alpha``
        if alpha_tikhonov < 0:
            raise ValueError('Parameter `alpha` must be positive or zero.')
        # Validate ``picos_eps``
        if picos_eps < 0:
            raise ValueError('Parameter `picos_eps` must be positive or zero.')
        # Compute needed sizes
        q = Z_hat.shape[0]
        p, r_tld = Q_tld.shape
        p_theta, r_hat = Q_hat.shape
        # Compute Q_hat
        p_upsilon = p - p_theta
        Q_bar = linalg.block_diag(Q_hat, np.eye(p_upsilon)).T @ Q_tld
        # Create optimization problem
        problem = picos.Problem()
        # Constants.
        # Sigmas are scaled by ``1/sqrt(q)`` to scale cost function, like EDMD.
        Sigma_hat_sq = picos.Constant('Sigma_hat^2', np.diag(sig_hat**2 / q))
        # Add regularizer to ``Sigma_tld``
        Sigma_hat = np.diag(sig_hat / np.sqrt(q))
        Sigma_tld = np.diag(np.sqrt(sig_tld**2 / q + alpha_tikhonov))
        big_constant = picos.Constant(
            'Q_bar Sigma_tld Z_tld.T Z_hat Sigma_Hat',
            Q_bar @ Sigma_tld @ Z_tld.T @ Z_hat @ Sigma_hat,
        )
        Q_bar_Sigma_tld = picos.Constant(
            'Q_bar Sigma_tld',
            Q_bar @ Sigma_tld,
        )
        m1 = picos.Constant('-1', -1 * np.eye(r_tld))
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


class LmiEdmdSpectralRadiusConstr(LmiRegressor):
    """LMI-based EDMD with spectral radius constraint.

    Optionally supports Tikhonov regularization.

    Attributes
    ----------
    tsvd_ : pykoop.Tsvd
        Fit truncated SVD object.
    P_ : np.ndarray
        ``P`` matrix for debugging.
    objective_log_ : List[float]
        Objective function history.
    stop_reason_ : str
        Reason iteration stopped.
    n_iter_ : int
        Number of iterations
    solver_params_ : Dict[str, Any]
        Solver parameters used (defaults merged with constructor input).
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    coef_ : np.ndarray
        Fit coefficient matrix.

    Examples
    --------
    Apply EDMD spectral radius constraint to mass-spring-damper data

    >>> kp = pykoop.KoopmanPipeline(
    ...     regressor=pykoop.lmi_regressors.LmiEdmdSpectralRadiusConstr(
    ...         spectral_radius=0.9,
    ...     )
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
    KoopmanPipeline(regressor=LmiEdmdSpectralRadiusConstr(spectral_radius=0.9))
    """

    def __init__(self,
                 spectral_radius: float = 1.0,
                 max_iter: int = 100,
                 iter_atol: float = 1e-6,
                 iter_rtol: float = 0,
                 alpha: float = 0,
                 inv_method: str = 'svd',
                 tsvd: tsvd.Tsvd = None,
                 picos_eps: float = 0,
                 solver_params: Dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiEdmdSpectralRadiusConstr`.

        To disable regularization, use ``alpha=0``.

        Parameters
        ----------
        spectral_radius : float
            Maximum spectral radius.

        max_iter : int
            Maximum number of solver iterations.

        iter_atol : float
            Absolute tolerance for change in objective function value.

        iter_rtol : float
            Relative tolerance for change in objective function value.

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

         tsvd : pykoop.Tsvd
            Singular value truncation method if ``inv_method='svd'``. If
            ``None``, economy SVD is used.

        picos_eps : float
            Tolerance used for strict LMIs. If nonzero, should be larger than
            solver tolerance.

        solver_params : Dict[str, Any]
            Parameters passed to PICOS :func:`picos.Problem.solve()`. By
            default, allows chosen solver to select its own tolerances.
        """
        self.spectral_radius = spectral_radius
        self.max_iter = max_iter
        self.iter_atol = iter_atol
        self.iter_rtol = iter_rtol
        self.alpha = alpha
        self.inv_method = inv_method
        self.tsvd = tsvd
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Clone TSVD
        self.tsvd_ = (sklearn.base.clone(self.tsvd)
                      if self.tsvd is not None else tsvd.Tsvd())
        # Get needed sizes
        p = X_unshifted.shape[1]
        p_theta = X_shifted.shape[1]
        # Make initial guesses and iterate
        P = np.eye(p_theta)
        # Set scope of other variables
        U = np.zeros((p_theta, p))
        self.objective_log_ = []
        for k in range(self.max_iter):
            # Formulate Problem A
            problem_a = self._create_problem_a(X_unshifted, X_shifted, P)
            # Solve Problem A
            if polite_stop:
                self.stop_reason_ = 'User requested stop.'
                log.warning(self.stop_reason_)
                break
            log.info(f'Solving problem A{k}')
            problem_a.solve(**self.solver_params_)
            solution_status_a = problem_a.last_solution.claimedStatus
            if solution_status_a != 'optimal':
                self.stop_reason_ = (
                    'Unable to solve `problem_a`. Used last valid `U`. '
                    f'Solution status: `{solution_status_a}`.')
                log.warning(self.stop_reason_)
                break
            U = np.array(problem_a.get_valued_variable('U'), ndmin=2)
            # Check stopping condition
            self.objective_log_.append(problem_a.value)
            if len(self.objective_log_) > 1:
                curr_obj = self.objective_log_[-1]
                prev_obj = self.objective_log_[-2]
                diff_obj = prev_obj - curr_obj
                log.info(f'Objective: {curr_obj}; Change: {diff_obj}.')
                if np.allclose(curr_obj,
                               prev_obj,
                               atol=self.iter_atol,
                               rtol=self.iter_rtol):
                    self.stop_reason_ = f'Reached tolerance {diff_obj}'
                    break
            # Formulate Problem B
            problem_b = self._create_problem_b(U)
            # Solve Problem B
            if polite_stop:
                self.stop_reason_ = 'User requested stop.'
                log.warning(self.stop_reason_)
                break
            log.info(f'Solving problem B{k}')
            problem_b.solve(**self.solver_params_)
            solution_status_b = problem_b.last_solution.claimedStatus
            if solution_status_b != 'optimal':
                self.stop_reason_ = (
                    'Unable to solve `problem_b`. Used last valid `U`. '
                    f'Solution status: `{solution_status_b}`.')
                log.warning(self.stop_reason_)
                break
            P = np.array(problem_b.get_valued_variable('P'), ndmin=2)
        else:
            self.stop_reason_ = f'Reached maximum iterations {self.max_iter}'
            log.warning(self.stop_reason_)
        self.n_iter_ = k + 1
        coef = U.T
        # Only useful for debugging
        self.P_ = P
        return coef

    def _validate_parameters(self) -> None:
        # Check spectral radius
        if self.spectral_radius <= 0:
            raise ValueError('`spectral_radius` must be positive.')
        if self.max_iter <= 0:
            raise ValueError('`max_iter` must be positive.')
        if self.iter_atol < 0:
            raise ValueError('`iter_atol` must be positive or zero.')
        if self.iter_rtol < 0:
            raise ValueError('`iter_rtol` must be positive or zero.')

    def _create_problem_a(self, X_unshifted: np.ndarray, X_shifted: np.ndarray,
                          P: np.ndarray) -> picos.Problem:
        """Create first problem in iteration scheme."""
        q = X_unshifted.shape[0]
        problem_a = LmiEdmd._create_base_problem(X_unshifted, X_shifted,
                                                 self.alpha / q,
                                                 self.inv_method, self.tsvd_,
                                                 self.picos_eps)
        # Extract information from problem
        U = problem_a.variables['U']
        # Get needed sizes
        p_theta = U.shape[0]
        # Add new constraints
        rho_bar = picos.Constant('rho_bar', self.spectral_radius)
        P = picos.Constant('P', P)
        problem_a.add_constraint(
            picos.block([
                # Use ``(P + P.T) / 2`` so PICOS understands it's symmetric.
                [rho_bar * (P + P.T) / 2, U[:, :p_theta].T * P],
                [P.T * U[:, :p_theta], rho_bar * (P + P.T) / 2],
            ]) >> self.picos_eps)
        return problem_a

    def _create_problem_b(self, U: np.ndarray) -> picos.Problem:
        """Create second problem in iteration scheme."""
        # Create optimization problem
        problem_b = picos.Problem()
        # Get needed sizes
        p_theta = U.shape[0]
        # Create constants
        rho_bar = picos.Constant('rho_bar', self.spectral_radius)
        U = picos.Constant('U', U)
        # Create variables
        P = picos.SymmetricVariable('P', p_theta)
        # Add constraints
        problem_b.add_constraint(P >> self.picos_eps)
        problem_b.add_constraint(
            picos.block([
                [rho_bar * P, U[:, :p_theta].T * P],
                [P.T * U[:, :p_theta], rho_bar * P],
            ]) >> self.picos_eps)
        # Set objective
        problem_b.set_objective('find')
        return problem_b


class LmiDmdcSpectralRadiusConstr(LmiRegressor):
    """LMI-based Dmdc with spectral radius constraint.

    Optionally supports Tikhonov regularization.

    Attributes
    ----------
    tsvd_unshifted_ : pykoop.Tsvd
        Fit truncated SVD object for unshifted data matrix.
    tsvd_shifted_ : pykoop.Tsvd
        Fit truncated SVD object for shifted data matrix.
    U_hat_ : np.ndarray
        Reduced Koopman matrix for debugging.
    P_ : np.ndarray
        ``P`` matrix for debugging.
    objective_log_ : List[float]
        Objective function history.
    stop_reason_ : str
        Reason iteration stopped.
    n_iter_ : int
        Number of iterations
    solver_params_ : Dict[str, Any]
        Solver parameters used (defaults merged with constructor input).
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    coef_ : np.ndarray
        Fit coefficient matrix.

    Examples
    --------
    Apply DMDc spectral radius constraint to mass-spring-damper data

    >>> kp = pykoop.KoopmanPipeline(
    ...     regressor=pykoop.lmi_regressors.LmiDmdcSpectralRadiusConstr(
    ...         spectral_radius=0.9,
    ...         tsvd_unshifted=pykoop.Tsvd('cutoff', 1e-6),
    ...         tsvd_shifted=pykoop.Tsvd('cutoff', 1e-6),
    ...     )
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
    KoopmanPipeline(regressor=LmiDmdcSpectralRadiusConstr(spectral_radius=0.9,
    tsvd_shifted=Tsvd(truncation='cutoff', truncation_param=1e-06),
    tsvd_unshifted=Tsvd(truncation='cutoff', truncation_param=1e-06)))
    """

    def __init__(self,
                 spectral_radius: float = 1.0,
                 max_iter: int = 100,
                 iter_atol: float = 1e-6,
                 iter_rtol: float = 0,
                 alpha: float = 0,
                 tsvd_unshifted: tsvd.Tsvd = None,
                 tsvd_shifted: tsvd.Tsvd = None,
                 picos_eps: float = 0,
                 solver_params: Dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiDmdcSpectralRadiusConstr`.

        To disable regularization, use ``alpha=0``.

        Parameters
        ----------
        spectral_radius : float
            Maximum spectral radius.
        max_iter : int
            Maximum number of solver iterations.
        iter_atol : float
            Absolute tolerance for change in objective function value.
        iter_rtol : float
            Relative tolerance for change in objective function value.
        alpha : float
            Tikhonov regularization coefficient.
        tsvd_unshifted : pykoop.Tsvd
            Singular value truncation method used to change basis of unshifted
            data matrix. If ``None``, economy SVD is used.
        tsvd_shifted : pykoop.Tsvd
            Singular value truncation method used to change basis of shifted
            data matrix. If ``None``, economy SVD is used.
        picos_eps : float
            Tolerance used for strict LMIs. If nonzero, should be larger than
            solver tolerance.
        solver_params : Dict[str, Any]
            Parameters passed to PICOS :func:`picos.Problem.solve()`. By
            default, allows chosen solver to select its own tolerances.
        """
        self.spectral_radius = spectral_radius
        self.max_iter = max_iter
        self.iter_atol = iter_atol
        self.iter_rtol = iter_rtol
        self.alpha = alpha
        self.tsvd_unshifted = tsvd_unshifted
        self.tsvd_shifted = tsvd_shifted
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
        # Clone TSVDs
        self.tsvd_unshifted_ = (sklearn.base.clone(self.tsvd_unshifted)
                                if self.tsvd_unshifted is not None else
                                tsvd.Tsvd())
        self.tsvd_shifted_ = (sklearn.base.clone(self.tsvd_shifted) if
                              self.tsvd_shifted is not None else tsvd.Tsvd())
        # Compute SVDs
        self.tsvd_unshifted_.fit(X_unshifted.T)
        Q_tld = self.tsvd_unshifted_.left_singular_vectors_
        sig_tld = self.tsvd_unshifted_.singular_values_
        Z_tld = self.tsvd_unshifted_.right_singular_vectors_
        self.tsvd_shifted_.fit(X_shifted.T)
        Q_hat = self.tsvd_shifted_.left_singular_vectors_
        sig_hat = self.tsvd_shifted_.singular_values_
        Z_hat = self.tsvd_shifted_.right_singular_vectors_
        # Get truncation values
        r_tld = Q_tld.shape[1]
        r_hat = Q_hat.shape[1]
        # Make initial guesses and iterate
        P = np.eye(r_hat)
        # Set scope of other variables
        U_hat = np.zeros((r_hat, r_hat + p - p_theta))
        self.objective_log_ = []
        for k in range(self.max_iter):
            # Formulate Problem A
            problem_a = self._create_problem_a(Q_tld, sig_tld, Z_tld, Q_hat,
                                               sig_hat, Z_hat, P)
            # Solve Problem A
            if polite_stop:
                self.stop_reason_ = 'User requested stop.'
                log.warning(self.stop_reason_)
                break
            log.info(f'Solving problem A{k}')
            problem_a.solve(**self.solver_params_)
            solution_status_a = problem_a.last_solution.claimedStatus
            if solution_status_a != 'optimal':
                self.stop_reason_ = (
                    'Unable to solve `problem_a`. Used last valid `U_hat`. '
                    f'Solution status: `{solution_status_a}`.')
                log.warning(self.stop_reason_)
                break
            U_hat = np.array(problem_a.get_valued_variable('U_hat'), ndmin=2)
            # Check stopping condition
            self.objective_log_.append(problem_a.value)
            if len(self.objective_log_) > 1:
                curr_obj = self.objective_log_[-1]
                prev_obj = self.objective_log_[-2]
                diff_obj = prev_obj - curr_obj
                log.info(f'Objective: {curr_obj}; Change: {diff_obj}.')
                if np.allclose(curr_obj,
                               prev_obj,
                               atol=self.iter_atol,
                               rtol=self.iter_rtol):
                    self.stop_reason_ = f'Reached tolerance {diff_obj}'
                    break
            # Formulate Problem B
            problem_b = self._create_problem_b(U_hat)
            # Solve Problem B
            if polite_stop:
                self.stop_reason_ = 'User requested stop.'
                log.warning(self.stop_reason_)
                break
            log.info(f'Solving problem B{k}')
            problem_b.solve(**self.solver_params_)
            solution_status_b = problem_b.last_solution.claimedStatus
            if solution_status_b != 'optimal':
                self.stop_reason_ = (
                    'Unable to solve `problem_b`. Used last valid `U_hat`. '
                    f'Solution status: `{solution_status_b}`.')
                log.warning(self.stop_reason_)
                break
            P = np.array(problem_b.get_valued_variable('P'), ndmin=2)
        else:
            self.stop_reason_ = f'Reached maximum iterations {self.max_iter}'
            log.warning(self.stop_reason_)
        self.n_iter_ = k + 1
        p_upsilon = p - p_theta
        U = Q_hat @ U_hat @ linalg.block_diag(Q_hat, np.eye(p_upsilon)).T
        coef = U.T
        # Only useful for debugging
        self.U_hat_ = U_hat
        self.P_ = P
        return coef

    def _validate_parameters(self) -> None:
        # Check spectral radius
        if self.spectral_radius <= 0:
            raise ValueError('`spectral_radius` must be positive.')
        if self.max_iter <= 0:
            raise ValueError('`max_iter` must be positive.')
        if self.iter_atol < 0:
            raise ValueError('`iter_atol` must be positive or zero.')
        if self.iter_rtol < 0:
            raise ValueError('`iter_rtol` must be positive or zero.')

    def _create_problem_a(
        self,
        Q_tld: np.ndarray,
        sig_tld: np.ndarray,
        Z_tld: np.ndarray,
        Q_hat: np.ndarray,
        sig_hat: np.ndarray,
        Z_hat: np.ndarray,
        P: np.ndarray,
    ) -> picos.Problem:
        """Create first problem in iteration scheme."""
        q = Z_hat.shape[0]
        problem_a = LmiDmdc._create_base_problem(Q_tld, sig_tld, Z_tld, Q_hat,
                                                 sig_hat, Z_hat,
                                                 self.alpha / q,
                                                 self.picos_eps)
        # Extract information from problem
        U_hat = problem_a.variables['U_hat']
        # Get needed sizes
        p_theta = U_hat.shape[0]
        # Add new constraints
        rho_bar = picos.Constant('rho_bar', self.spectral_radius)
        P = picos.Constant('P', P)
        problem_a.add_constraint(
            picos.block([
                # Use ``(P + P.T) / 2`` so PICOS understands it's symmetric.
                [rho_bar * (P + P.T) / 2, U_hat[:, :p_theta].T * P],
                [P.T * U_hat[:, :p_theta], rho_bar * (P + P.T) / 2],
            ]) >> self.picos_eps)
        return problem_a

    def _create_problem_b(self, U_hat: np.ndarray) -> picos.Problem:
        """Create second problem in iteration scheme."""
        # Create optimization problem
        problem_b = picos.Problem()
        # Get needed sizes
        p_theta = U_hat.shape[0]
        # Create constants
        rho_bar = picos.Constant('rho_bar', self.spectral_radius)
        U = picos.Constant('U', U_hat)
        # Create variables
        P = picos.SymmetricVariable('P', p_theta)
        # Add constraints
        problem_b.add_constraint(P >> self.picos_eps)
        problem_b.add_constraint(
            picos.block([
                [rho_bar * P, U_hat[:, :p_theta].T * P],
                [P.T * U_hat[:, :p_theta], rho_bar * P],
            ]) >> self.picos_eps)
        # Set objective
        problem_b.set_objective('find')
        return problem_b


class LmiEdmdHinfReg(LmiRegressor):
    """LMI-based EDMD with H-infinity norm regularization.

    Optionally supports additional Tikhonov regularization.

    Attributes
    ----------
    tsvd_ : pykoop.Tsvd
        Fit truncated SVD object.
    P_ : np.ndarray
        ``P`` matirx for debugging.
    gamma_ : np.ndarray
        H-infinity norm for debugging.
    objective_log_ : List[float]
        Objective function history.
    stop_reason_ : str
        Reason iteration stopped.
    n_iter_ : int
        Number of iterations
    solver_params_ : Dict[str, Any]
        Solver parameters used (defaults merged with constructor input).
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    coef_ : np.ndarray
        Fit coefficient matrix.

    Examples
    --------
    Apply EDMD with H-infinity regularization to mass-spring-damper data

    >>> kp = pykoop.KoopmanPipeline(
    ...     regressor=pykoop.lmi_regressors.LmiEdmdHinfReg(
    ...         alpha=1e-3,
    ...     )
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
    KoopmanPipeline(regressor=LmiEdmdHinfReg(alpha=0.001))

    Apply EDMD with weighted H-infinity regularization to mass-spring-damper
    data

    >>> from scipy import signal
    >>> ss_ct = signal.ZerosPolesGain([0], [-4], [1]).to_ss()
    >>> ss_dt = ss_ct.to_discrete(dt=0.1, method='bilinear')
    >>> kp = pykoop.KoopmanPipeline(
    ...     regressor=pykoop.lmi_regressors.LmiEdmdHinfReg(
    ...         alpha=1e-3,
    ...         weight=('pre', ss_dt.A, ss_dt.B, ss_dt.C, ss_dt.D),
    ...     )
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
    KoopmanPipeline(regressor=LmiEdmdHinfReg(alpha=0.001,
    weight=('pre', array([[...]]), array([[...]]), array([[...]]),
    array([[...]]))))
    """

    def __init__(
        self,
        alpha: float = 1,
        ratio: float = 1,
        weight: Tuple[str, np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray] = None,
        max_iter: int = 100,
        iter_atol: float = 1e-6,
        iter_rtol: float = 0,
        inv_method: str = 'svd',
        tsvd: tsvd.Tsvd = None,
        square_norm: bool = False,
        picos_eps: float = 0,
        solver_params: Dict[str, Any] = None,
    ) -> None:
        """Instantiate :class:`LmiEdmdHinfReg`.

        Supports cascading the plant with an LTI weighting function.

        Parameters
        ----------
        alpha : float
            Regularization coefficient. Cannot be zero.

        ratio : float
            Ratio of H-infinity norm to use in mixed regularization. If
            ``ratio=1``, no Tikhonov regularization is used. Cannot be zero.

        weight : Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing weight type (``'pre'`` or ``'post'``), and the
            weight state space matrices (``A``, ``B``, ``C``, and ``D``). If
            ``None``, no weighting is used.

        max_iter : int
            Maximum number of solver iterations.

        iter_atol : float
            Absolute tolerance for change in objective function value.

        iter_rtol : float
            Relative tolerance for change in objective function value.

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

         tsvd : pykoop.Tsvd
            Singular value truncation method if ``inv_method='svd'``. If
            ``None``, economy SVD is used.

        square_norm : bool
            Square norm H-infinity norm in regularizer. Enabling may increase
            computation time. Frobenius norm used in Tikhonov regularizer is
            always squared.

        picos_eps : float
            Tolerance used for strict LMIs. If nonzero, should be larger than
            solver tolerance.

        solver_params : Dict[str, Any]
            Parameters passed to PICOS :func:`picos.Problem.solve()`. By
            default, allows chosen solver to select its own tolerances.
        """
        self.alpha = alpha
        self.ratio = ratio
        self.weight = weight
        self.max_iter = max_iter
        self.iter_atol = iter_atol
        self.iter_rtol = iter_rtol
        self.inv_method = inv_method
        self.tsvd = tsvd
        self.square_norm = square_norm
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Clone TSVD
        self.tsvd_ = (sklearn.base.clone(self.tsvd)
                      if self.tsvd is not None else tsvd.Tsvd())
        # Set regularization coefficients
        self.alpha_tikhonov_ = self.alpha * (1 - self.ratio)
        self.alpha_other_ = self.alpha * self.ratio
        # Get needed sizes
        p = X_unshifted.shape[1]
        p_theta = X_shifted.shape[1]
        # Check that at least one input is present
        if p_theta == p:
            # If you remove the ``{p} features(s)`` part of this message,
            # the scikit-learn estimator_checks will fail!
            raise ValueError('`LmiEdmdHinfReg()` requires an input to '
                             'function. `X` and `y` must therefore have '
                             'different numbers of features. `X` and `y` both '
                             f'have {p} feature(s).')
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
                log.warning(self.stop_reason_)
                break
            log.info(f'Solving problem A{k}')
            problem_a.solve(**self.solver_params_)
            solution_status_a = problem_a.last_solution.claimedStatus
            if solution_status_a != 'optimal':
                self.stop_reason_ = (
                    'Unable to solve `problem_a`. Used last valid `U`. '
                    f'Solution status: `{solution_status_a}`.')
                log.warning(self.stop_reason_)
                break
            U = np.array(problem_a.get_valued_variable('U'), ndmin=2)
            gamma = np.array(problem_a.get_valued_variable('gamma'))
            self.objective_log_.append(problem_a.value)
            if len(self.objective_log_) > 1:
                curr_obj = self.objective_log_[-1]
                prev_obj = self.objective_log_[-2]
                diff_obj = prev_obj - curr_obj
                log.info(f'Objective: {curr_obj}; Change: {diff_obj}.')
                if np.allclose(curr_obj,
                               prev_obj,
                               atol=self.iter_atol,
                               rtol=self.iter_rtol):
                    self.stop_reason_ = f'Reached tolerance {diff_obj}'
                    break
            # Formulate Problem B
            problem_b = self._create_problem_b(U, gamma)
            # Solve Problem B
            if polite_stop:
                self.stop_reason_ = 'User requested stop.'
                log.warning(self.stop_reason_)
                break
            log.info(f'Solving problem B{k}')
            problem_b.solve(**self.solver_params_)
            solution_status_b = problem_b.last_solution.claimedStatus
            if solution_status_b != 'optimal':
                self.stop_reason_ = (
                    'Unable to solve `problem_b`. Used last valid `U`. '
                    'Solution status: f`{solution_status_b}`.')
                log.warning(self.stop_reason_)
                break
            P = np.array(problem_b.get_valued_variable('P'), ndmin=2)
        else:
            self.stop_reason_ = f'Reached maximum iterations {self.max_iter}'
            log.warning(self.stop_reason_)
        self.n_iter_ = k + 1
        coef = U.T
        # Only useful for debugging
        self.P_ = P
        self.gamma_ = gamma
        return coef

    def _validate_parameters(self) -> None:
        if (self.ratio <= 0) or (self.ratio > 1):
            raise ValueError('`ratio` must be positive and less than one.')
        valid_weight_types = ['pre', 'post']
        if self.weight is not None:
            if self.weight[0] not in valid_weight_types:
                raise ValueError('First element of the `weight` must be one '
                                 f'of {valid_weight_types}.')
        if self.max_iter <= 0:
            raise ValueError('`max_iter` must be positive.')
        if self.iter_atol < 0:
            raise ValueError('`iter_atol` must be positive or zero.')
        if self.iter_rtol < 0:
            raise ValueError('`iter_rtol` must be positive or zero.')

    def _create_problem_a(self, X_unshifted: np.ndarray, X_shifted: np.ndarray,
                          P: np.ndarray) -> picos.Problem:
        """Create first problem in iteration scheme."""
        q = X_unshifted.shape[0]
        problem_a = LmiEdmd._create_base_problem(X_unshifted, X_shifted,
                                                 self.alpha_tikhonov_ / q,
                                                 self.inv_method, self.tsvd_,
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
        A, B, C, D = _create_ss(U, self.weight)
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
        if self.alpha_other_ <= 0:
            raise ValueError('`alpha_other_` must be positive.')
        alpha_scaled = picos.Constant('alpha_scaled_inf',
                                      self.alpha_other_ / q)
        if self.square_norm:
            objective += alpha_scaled * gamma**2
        else:
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
        A, B, C, D = _create_ss(U, self.weight)
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


class LmiDmdcHinfReg(LmiRegressor):
    """LMI-based DMDc with H-infinity norm regularization.

    Optionally supports additional Tikhonov regularization.

    Attributes
    ----------
    tsvd_unshifted_ : pykoop.Tsvd
        Fit truncated SVD object for unshifted data matrix.
    tsvd_shifted_ : pykoop.Tsvd
        Fit truncated SVD object for shifted data matrix.
    U_hat_ : np.ndarray
        Reduced Koopman matrix for debugging.
    P_ : np.ndarray
        ``P`` matirx for debugging.
    gamma_ : np.ndarray
        H-infinity norm for debugging.
    objective_log_ : List[float]
        Objective function history.
    stop_reason_ : str
        Reason iteration stopped.
    n_iter_ : int
        Number of iterations
    solver_params_ : Dict[str, Any]
        Solver parameters used (defaults merged with constructor input).
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    coef_ : np.ndarray
        Fit coefficient matrix

    Examples
    --------
    Apply DMDc with H-infinity regularization to mass-spring-damper data

    >>> kp = pykoop.KoopmanPipeline(
    ...     regressor=pykoop.lmi_regressors.LmiDmdcHinfReg(
    ...         alpha=1e-3,
    ...     )
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
    KoopmanPipeline(regressor=LmiDmdcHinfReg(alpha=0.001))

    Apply reduced-order DMDc with weighted H-infinity regularization to
    mass-spring-damper data

    >>> from scipy import signal
    >>> ss_ct = signal.ZerosPolesGain([0], [-4], [1]).to_ss()
    >>> ss_dt = ss_ct.to_discrete(dt=0.1, method='bilinear')
    >>> kp = pykoop.KoopmanPipeline(
    ...     regressor=pykoop.lmi_regressors.LmiDmdcHinfReg(
    ...         alpha=1e-3,
    ...         weight=('pre', ss_dt.A, ss_dt.B, ss_dt.C, ss_dt.D),
    ...         tsvd_unshifted=pykoop.Tsvd('cutoff', 1e-3),
    ...         tsvd_shifted=pykoop.Tsvd('cutoff', 1e-3),
    ...     )
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
    KoopmanPipeline(regressor=LmiDmdcHinfReg(alpha=0.001,
    tsvd_shifted=Tsvd(truncation='cutoff', truncation_param=0.001),
    tsvd_unshifted=Tsvd(truncation='cutoff', truncation_param=0.001),
    weight=('pre', array([[...]]), array([[...]]), array([[...]]),
    array([[...]]))))

    """

    def __init__(
        self,
        alpha: float = 1,
        ratio: float = 1,
        weight: Tuple[str, np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray] = None,
        max_iter: int = 100,
        iter_atol: float = 1e-6,
        iter_rtol: float = 0,
        tsvd_unshifted: tsvd.Tsvd = None,
        tsvd_shifted: tsvd.Tsvd = None,
        square_norm: bool = False,
        picos_eps: float = 0,
        solver_params: Dict[str, Any] = None,
    ) -> None:
        """Instantiate :class:`LmiDmdcHinfReg`.

        Supports cascading the plant with an LTI weighting function.

        Parameters
        ----------
        alpha : float
            Regularization coefficient. Cannot be zero.
        ratio : float
            Ratio of H-infinity norm to use in mixed regularization. If
            ``ratio=1``, no Tikhonov regularization is used. Cannot be zero.
        weight : Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing weight type (``'pre'`` or ``'post'``), and the
            weight state space matrices (``A``, ``B``, ``C``, and ``D``). If
            ``None``, no weighting is used.
        max_iter : int
            Maximum number of solver iterations.
        iter_atol : float
            Absolute tolerance for change in objective function value.
        iter_rtol : float
            Relative tolerance for change in objective function value.
        tsvd_unshifted : pykoop.Tsvd
            Singular value truncation method used to change basis of unshifted
            data matrix. If ``None``, economy SVD is used.
        tsvd_shifted : pykoop.Tsvd
            Singular value truncation method used to change basis of shifted
            data matrix. If ``None``, economy SVD is used.
        square_norm : bool
            Square norm H-infinity norm in regularizer. Enabling may increase
            computation time. Frobenius norm used in Tikhonov regularizer is
            always squared.
        picos_eps : float
            Tolerance used for strict LMIs. If nonzero, should be larger than
            solver tolerance.
        solver_params : Dict[str, Any]
            Parameters passed to PICOS :func:`picos.Problem.solve()`. By
            default, allows chosen solver to select its own tolerances.
        """
        self.alpha = alpha
        self.ratio = ratio
        self.weight = weight
        self.max_iter = max_iter
        self.iter_atol = iter_atol
        self.iter_rtol = iter_rtol
        self.tsvd_unshifted = tsvd_unshifted
        self.tsvd_shifted = tsvd_shifted
        self.square_norm = square_norm
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Clone TSVDs
        self.tsvd_unshifted_ = (sklearn.base.clone(self.tsvd_unshifted)
                                if self.tsvd_unshifted is not None else
                                tsvd.Tsvd())
        self.tsvd_shifted_ = (sklearn.base.clone(self.tsvd_shifted) if
                              self.tsvd_shifted is not None else tsvd.Tsvd())
        # Set regularization coefficients
        self.alpha_tikhonov_ = self.alpha * (1 - self.ratio)
        self.alpha_other_ = self.alpha * self.ratio
        # Get needed sizes
        p = X_unshifted.shape[1]
        p_theta = X_shifted.shape[1]
        # Check that at least one input is present
        if p_theta == p:
            # If you remove the ``{p} features(s)`` part of this message,
            # the scikit-learn estimator_checks will fail!
            raise ValueError('`LmiDmdcHinfReg()` requires an input to '
                             'function. `X` and `y` must therefore have '
                             'different numbers of features. `X` and `y` both '
                             f'have {p} feature(s).')
        # Compute SVDs
        self.tsvd_unshifted_.fit(X_unshifted.T)
        Q_tld = self.tsvd_unshifted_.left_singular_vectors_
        sig_tld = self.tsvd_unshifted_.singular_values_
        Z_tld = self.tsvd_unshifted_.right_singular_vectors_
        self.tsvd_shifted_.fit(X_shifted.T)
        Q_hat = self.tsvd_shifted_.left_singular_vectors_
        sig_hat = self.tsvd_shifted_.singular_values_
        Z_hat = self.tsvd_shifted_.right_singular_vectors_
        r_tld = Q_tld.shape[1]
        r_hat = Q_hat.shape[1]
        # Set up weights
        if self.weight is None:
            P = np.eye(r_hat)
        elif self.weight[0] == 'pre':
            p_upsilon = p - p_theta
            P = np.eye(r_hat + p_upsilon * self.weight[1].shape[0])
        elif self.weight[0] == 'post':
            P = np.eye(r_hat + p_theta * self.weight[1].shape[0])
        else:
            # Already checked. Should never get here.
            assert False
        # Solve optimization problem iteratively
        U_hat = np.zeros((r_hat, r_hat + p - p_theta))
        gamma = np.zeros((1, ))
        self.objective_log_ = []
        for k in range(self.max_iter):
            # Formulate Problem A
            problem_a = self._create_problem_a(Q_tld, sig_tld, Z_tld, Q_hat,
                                               sig_hat, Z_hat, P)
            # Solve Problem A
            if polite_stop:
                self.stop_reason_ = 'User requested stop.'
                log.warning(self.stop_reason_)
                break
            log.info(f'Solving problem A{k}')
            problem_a.solve(**self.solver_params_)
            solution_status_a = problem_a.last_solution.claimedStatus
            if solution_status_a != 'optimal':
                self.stop_reason_ = (
                    'Unable to solve `problem_a`. Used last valid `U_hat`. '
                    f'Solution status: `{solution_status_a}`.')
                log.warning(self.stop_reason_)
                break
            U_hat = np.array(problem_a.get_valued_variable('U_hat'), ndmin=2)
            gamma = np.array(problem_a.get_valued_variable('gamma'))
            self.objective_log_.append(problem_a.value)
            if len(self.objective_log_) > 1:
                curr_obj = self.objective_log_[-1]
                prev_obj = self.objective_log_[-2]
                diff_obj = prev_obj - curr_obj
                log.info(f'Objective: {curr_obj}; Change: {diff_obj}.')
                if np.allclose(curr_obj,
                               prev_obj,
                               atol=self.iter_atol,
                               rtol=self.iter_rtol):
                    self.stop_reason_ = f'Reached tolerance {diff_obj}'
                    break
            # Formulate Problem B
            problem_b = self._create_problem_b(U_hat, gamma)
            # Solve Problem B
            if polite_stop:
                self.stop_reason_ = 'User requested stop.'
                log.warning(self.stop_reason_)
                break
            log.info(f'Solving problem B{k}')
            problem_b.solve(**self.solver_params_)
            solution_status_b = problem_b.last_solution.claimedStatus
            if solution_status_b != 'optimal':
                self.stop_reason_ = (
                    'Unable to solve `problem_b`. Used last valid `U_hat`. '
                    'Solution status: f`{solution_status_b}`.')
                log.warning(self.stop_reason_)
                break
            P = np.array(problem_b.get_valued_variable('P'), ndmin=2)
        else:
            self.stop_reason_ = f'Reached maximum iterations {self.max_iter}'
            log.warning(self.stop_reason_)
        self.n_iter_ = k + 1
        p_upsilon = p - p_theta
        U = Q_hat @ U_hat @ linalg.block_diag(Q_hat, np.eye(p_upsilon)).T
        coef = U.T
        # Only useful for debugging
        self.U_hat_ = U_hat
        self.P_ = P
        self.gamma_ = gamma
        return coef

    def _validate_parameters(self) -> None:
        if (self.ratio <= 0) or (self.ratio > 1):
            raise ValueError('`ratio` must be positive and less than one.')
        valid_weight_types = ['pre', 'post']
        if self.weight is not None:
            if self.weight[0] not in valid_weight_types:
                raise ValueError('First element of the `weight` must be one'
                                 f'of {valid_weight_types}.')
        if self.max_iter <= 0:
            raise ValueError('`max_iter` must be positive.')
        if self.iter_atol < 0:
            raise ValueError('`iter_atol` must be positive or zero.')
        if self.iter_rtol < 0:
            raise ValueError('`iter_rtol` must be positive or zero.')

    def _create_problem_a(
        self,
        Q_tld: np.ndarray,
        sig_tld: np.ndarray,
        Z_tld: np.ndarray,
        Q_hat: np.ndarray,
        sig_hat: np.ndarray,
        Z_hat: np.ndarray,
        P: np.ndarray,
    ) -> picos.Problem:
        """Create first problem in iteration scheme."""
        q = Z_hat.shape[0]
        problem_a = LmiDmdc._create_base_problem(Q_tld, sig_tld, Z_tld, Q_hat,
                                                 sig_hat, Z_hat,
                                                 self.alpha_tikhonov_ / q,
                                                 self.picos_eps)
        # Extract information from problem
        U_hat = problem_a.variables['U_hat']
        direction = problem_a.objective.direction
        objective = problem_a.objective.function
        # Get needed sizes
        p_theta = U_hat.shape[0]
        # Add new constraint
        P = picos.Constant('P', P)
        gamma = picos.RealVariable('gamma', 1)
        # Get weighted state space matrices
        A, B, C, D = _create_ss(
            U_hat,
            self.weight,
            Q_hat=self.tsvd_shifted_.left_singular_vectors_,
        )
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
        if self.alpha_other_ <= 0:
            raise ValueError('`alpha_other_` must be positive.')
        alpha_scaled = picos.Constant('alpha_scaled_inf',
                                      self.alpha_other_ / q)
        if self.square_norm:
            objective += alpha_scaled * gamma**2
        else:
            objective += alpha_scaled * gamma
        problem_a.set_objective(direction, objective)
        return problem_a

    def _create_problem_b(self, U_hat: np.ndarray,
                          gamma: np.ndarray) -> picos.Problem:
        """Create second problem in iteration scheme."""
        # Create optimization problem
        problem_b = picos.Problem()
        # Get needed sizes
        p_theta = U_hat.shape[0]
        # Create constants
        U_hat = picos.Constant('U_hat', U_hat)
        gamma = picos.Constant('gamma', gamma)
        # Get weighted state space matrices
        A, B, C, D = _create_ss(
            U_hat,
            self.weight,
            Q_hat=self.tsvd_shifted_.left_singular_vectors_,
        )
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


class LmiEdmdDissipativityConstr(LmiRegressor):
    """LMI-based EDMD with dissipativity constraint.

    Optionally supports additional Tikhonov regularization.

    Originally proposed in [HIS19]_.

    Attributes
    ----------
    tsvd_ : pykoop.Tsvd
        Fit truncated SVD object.
    objective_log_ : List[float]
        Objective function history.
    stop_reason_ : str
        Reason iteration stopped.
    n_iter_ : int
        Number of iterations
    solver_params_ : Dict[str, Any]
        Solver parameters used (defaults merged with constructor input).
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    coef_ : np.ndarray
        Fit coefficient matrix.

    Examples
    --------
    Apply dissipativity-constrainted EDMD to mass-spring-damper data

    >>> kp = pykoop.KoopmanPipeline(
    ...     regressor=pykoop.lmi_regressors.LmiEdmdDissipativityConstr()
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
    KoopmanPipeline(regressor=LmiEdmdDissipativityConstr())
    """

    def __init__(
        self,
        alpha: float = 1,
        supply_rate: np.ndarray = None,
        max_iter: int = 100,
        iter_atol: float = 1e-6,
        iter_rtol: float = 0,
        inv_method: str = 'svd',
        tsvd: tsvd.Tsvd = None,
        picos_eps: float = 0,
        solver_params: Dict[str, Any] = None,
    ) -> None:
        """Instantiate :class:`LmiEdmdDissipativityConstr`.

        The supply rate ``s(u, y)`` is specified by ``Xi``::

            s(u, y) = -[y, u] Xi [y; u]

        Some example supply rate matrices ``Xi`` are::

            Xi = [0, -1; -1, 0] -> passivity,
            Xi = [1/gamma, 0; 0, -gamma] -> bounded L2 gain of gamma.

        Parameters
        ----------
        alpha : float
            Regularization coefficient. Cannot be zero.

        supply_rate : np.ndarray
            Supply rate matrix ``Xi``, where ``s(u, y) = -[y, u] Xi [y; u]``.
            If ``None``, the an L2 gain of ``gamma=1`` is imposed.

        max_iter : int
            Maximum number of solver iterations.

        iter_atol : float
            Absolute tolerance for change in objective function value.

        iter_rtol : float
            Relative tolerance for change in objective function value.

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

         tsvd : pykoop.Tsvd
            Singular value truncation method if ``inv_method='svd'``. If
            ``None``, economy SVD is used.

        picos_eps : float
            Tolerance used for strict LMIs. If nonzero, should be larger than
            solver tolerance.

        solver_params : Dict[str, Any]
            Parameters passed to PICOS :func:`picos.Problem.solve()`. By
            default, allows chosen solver to select its own tolerances.
        """
        self.alpha = alpha
        self.supply_rate = supply_rate
        self.max_iter = max_iter
        self.iter_atol = iter_atol
        self.iter_rtol = iter_rtol
        self.inv_method = inv_method
        self.tsvd = tsvd
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Clone TSVD
        self.tsvd_ = (sklearn.base.clone(self.tsvd)
                      if self.tsvd is not None else tsvd.Tsvd())
        # Set regularization coefficients
        # Get needed sizes
        p = X_unshifted.shape[1]
        p_theta = X_shifted.shape[1]
        # Check that at least one input is present
        if p_theta == p:
            # If you remove the ``{p} features(s)`` part of this message,
            # the scikit-learn estimator_checks will fail!
            raise ValueError('`LmiEdmdDissipativityConstr()` requires an '
                             'input to function. `X` and `y` must therefore '
                             'have different numbers of features. `X` and `y` '
                             f'both have {p} feature(s).')
        # Initialize ``P``
        P = np.eye(p_theta)
        # Solve optimization problem iteratively
        U = np.zeros((p_theta, p))
        self.objective_log_ = []
        for k in range(self.max_iter):
            # Formulate Problem A
            problem_a = self._create_problem_a(X_unshifted, X_shifted, P)
            # Solve Problem A
            if polite_stop:
                self.stop_reason_ = 'User requested stop.'
                log.warning(self.stop_reason_)
                break
            log.info(f'Solving problem A{k}')
            problem_a.solve(**self.solver_params_)
            solution_status_a = problem_a.last_solution.claimedStatus
            if solution_status_a != 'optimal':
                self.stop_reason_ = (
                    'Unable to solve `problem_a`. Used last valid `U`. '
                    f'Solution status: `{solution_status_a}`.')
                log.warning(self.stop_reason_)
                break
            U = np.array(problem_a.get_valued_variable('U'), ndmin=2)
            self.objective_log_.append(problem_a.value)
            if len(self.objective_log_) > 1:
                curr_obj = self.objective_log_[-1]
                prev_obj = self.objective_log_[-2]
                diff_obj = prev_obj - curr_obj
                log.info(f'Objective: {curr_obj}; Change: {diff_obj}.')
                if np.allclose(curr_obj,
                               prev_obj,
                               atol=self.iter_atol,
                               rtol=self.iter_rtol):
                    self.stop_reason_ = f'Reached tolerance {diff_obj}'
                    break
            # Formulate Problem B
            problem_b = self._create_problem_b(U)
            # Solve Problem B
            if polite_stop:
                self.stop_reason_ = 'User requested stop.'
                log.warning(self.stop_reason_)
                break
            log.info(f'Solving problem B{k}')
            problem_b.solve(**self.solver_params_)
            solution_status_b = problem_b.last_solution.claimedStatus
            if solution_status_b != 'optimal':
                self.stop_reason_ = (
                    'Unable to solve `problem_b`. Used last valid `U`. '
                    'Solution status: f`{solution_status_b}`.')
                log.warning(self.stop_reason_)
                break
            P = np.array(problem_b.get_valued_variable('P'), ndmin=2)
        else:
            self.stop_reason_ = f'Reached maximum iterations {self.max_iter}'
            log.warning(self.stop_reason_)
        self.n_iter_ = k + 1
        coef = U.T
        # Only useful for debugging
        self.P_ = P
        return coef

    def _validate_parameters(self) -> None:
        # Check other parameters
        if self.max_iter <= 0:
            raise ValueError('`max_iter` must be positive.')
        if self.iter_atol < 0:
            raise ValueError('`iter_atol` must be positive or zero.')
        if self.iter_rtol < 0:
            raise ValueError('`iter_rtol` must be positive or zero.')

    def _create_problem_a(self, X_unshifted: np.ndarray, X_shifted: np.ndarray,
                          P: np.ndarray) -> picos.Problem:
        """Create first problem in iteration scheme."""
        q = X_unshifted.shape[0]
        problem_a = LmiEdmd._create_base_problem(X_unshifted, X_shifted,
                                                 self.alpha / q,
                                                 self.inv_method, self.tsvd_,
                                                 self.picos_eps)
        # Extract information from problem
        U = problem_a.variables['U']
        # Get needed sizes
        p_theta, p = U.shape
        # Add new constraint
        P = picos.Constant('P', P)
        # Get weighted state space matrices
        A, B, C, D = _create_ss(U, None)
        # Add dissipativity constraint
        if self.supply_rate is None:
            n_u = p - p_theta
            Xi = np.block([
                [np.eye(p_theta), np.zeros((p_theta, n_u))],
                [np.zeros((n_u, p_theta)), -np.eye(n_u)],
            ])
        else:
            Xi = self.supply_rate
        Xi11 = picos.Constant('Xi_11', Xi[:p_theta, :p_theta])
        Xi12 = picos.Constant('Xi_12', Xi[:p_theta, p_theta:])
        Xi22 = picos.Constant('Xi_22', Xi[p_theta:, p_theta:])
        problem_a.add_constraint(
            picos.block([
                [P - C.T * Xi11 * C, -C.T * Xi12, A.T * P],
                [-Xi12.T * C, -Xi22, B.T * P],
                [P * A, P * B, P],
            ]) >> self.picos_eps)
        return problem_a

    def _create_problem_b(self, U: np.ndarray) -> picos.Problem:
        """Create second problem in iteration scheme."""
        # Create optimization problem
        problem_b = picos.Problem()
        # Get needed sizes
        p_theta, p = U.shape
        # Create constants
        U = picos.Constant('U', U)
        # Get weighted state space matrices
        A, B, C, D = _create_ss(U, None)
        # Create variables
        P = picos.SymmetricVariable('P', A.shape[0])
        # Add constraints
        problem_b.add_constraint(P >> self.picos_eps)
        # Add dissipativity constraint
        if self.supply_rate is None:
            n_u = p - p_theta
            Xi = np.block([
                [np.eye(p_theta), np.zeros((p_theta, n_u))],
                [np.zeros((n_u, p_theta)), -np.eye(n_u)],
            ])
        else:
            Xi = self.supply_rate
        Xi11 = picos.Constant('Xi_11', Xi[:p_theta, :p_theta])
        Xi12 = picos.Constant('Xi_12', Xi[:p_theta, p_theta:])
        Xi22 = picos.Constant('Xi_22', Xi[p_theta:, p_theta:])
        problem_b.add_constraint(P >> self.picos_eps)
        problem_b.add_constraint(
            picos.block([
                [P - C.T * Xi11 * C, -C.T * Xi12, A.T * P],
                [-Xi12.T * C, -Xi22, B.T * P],
                [P * A, P * B, P],
            ]) >> self.picos_eps)
        # Set objective
        problem_b.set_objective('find')
        return problem_b


class LmiHinfZpkMeta(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """Meta-estimator where H-infinity weight is specified in ZPK format.

    H-infinity regularization weights must normally be specified in
    discrete-time state space format. This can make cross-validating pole or
    zero positions annoying.  This meta-estimator wraps :class:`LmiEdmdHinfReg`
    or :class:`LmiDmdcHinfReg` and allows the weighting filter to be specified
    using zeros, poles, and a gain (i.e., ZPK format) in continuous-time.

    All attributes with a trailing underscore are set by :func:`fit`.

    Attributes
    ----------
    hinf_regressor_ : koopman_pipeline.KoopmanRegressor
        Fit internal regressor.
    ss_ct_ : scipy.signal.lti
        Continuous-times state space weight.
    ss_dt_ : scipy.signal.dlti
        Discrete-times state space weight.
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    coef_ : np.ndarray
        Fit coefficient matrix.
    """

    def __init__(
        self,
        hinf_regressor: koopman_pipeline.KoopmanRegressor = None,
        type: str = 'post',
        zeros: Union[float, np.ndarray] = None,
        poles: Union[float, np.ndarray] = None,
        gain: float = 1,
        discretization: str = 'bilinear',
        t_step: float = 1,
        units: str = 'rad/s',
    ) -> None:
        """Instantiate :class:`LmiHinfZpkMeta`.

        Parameters
        ----------
        hinf_regressor : koopman_pipeline.KoopmanRegressor
            Instance of :class:`LmiEdmdHinfReg` or :class:`LmiDmdcHinfReg`.

        type : str
            Type of weight (``'pre'`` or ``'post'``).

        zeros : Union[float, np.ndarray]
            Filter zeros. If ``None``, no zeros are used. Accepts scalar input
            if only one zero is required.

        poles : Union[float, np.ndarray]
            Filter poles. If ``None``, no poles are used. Accepts scalar input
            if only one pole is required.

        gain : float
            Filter gain.

        discretization : str
            Discretization method supported by
            :func:`scipy.signal.cont2discrete` (except ``'gbt'``).
            Specifically, possible valued are

            - ``'bilinear'`` -- Tustin's approximation (recommended),
            - ``'euler'`` -- Euler (or forward differencing) method,
            - ``'backward_diff'`` -- backwards differencing method,
            - ``'zoh'`` -- zero-order hold method,
            - ``'foh'`` -- first-order hold method, or
            - ``'impulse'`` -- equivalent impulse response method.

        t_step : float
            Timestep beween samples. Used for discretization.

        units : str
            Units of poles and zeros. Possible values are

            - ``'rad/s'`` -- radians per second,
            - ``'hz'`` -- Hertz, or
            - ``'normalized'`` -- normalized, where 1 is the Nyquist frequency.

        Notes
        -----
        The zeros and poles in the weight should usually have negative real
        parts! If you want a pole at ``10 rad/s``, then ``poles`` must be
        ``-10``.

        Examples
        --------
        >>> est = pykoop.lmi_regressors.LmiHinfZpkMeta(
        ...     hinf_regressor=pykoop.lmi_regressors.LmiEdmdHinfReg(),
        ...     type='post',
        ...     zeros=-0,
        ...     poles=-5,
        ...     gain=1,
        ...     discretization='bilinear',
        ...     t_step=0.1,
        ... )
        >>> est.fit(X_msd, n_inputs=1, episode_feature=True)  # doctest: +SKIP
        LmiHinfZpkMeta(hinf_regressor=LmiEdmdHinfReg(), poles=-5, t_step=0.1,
        zeros=0)
        """
        self.hinf_regressor = hinf_regressor
        self.type = type
        self.zeros = zeros
        self.poles = poles
        self.gain = gain
        self.discretization = discretization
        self.t_step = t_step
        self.units = units

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = False) -> 'LmiHinfZpkMeta':
        """Fit the regressor.

        If only ``X`` is specified, the regressor will compute its unshifted
        and shifted versions. If ``X`` and ``y`` are specified, ``X`` is
        treated as the unshifted data matrix, while ``y`` is treated as the
        shifted data matrix.

        Parameters
        ----------
        X : np.ndarray
            Full data matrix if ``y=None``. Unshifted data matrix if ``y`` is
            specified.
        y : np.ndarray
            Optional shifted data matrix. If ``None``, shifted data matrix is
            computed using ``X``.
        n_inputs : int
            Number of input features at the end of ``X``.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.

        Returns
        -------
        LmiHinfZpkMeta
            Instance of itself.

        Raises
        -----
        ValueError
            If constructor or fit parameters are incorrect.
        """
        z_in = np.atleast_1d(self.zeros if self.zeros is not None else [])
        p_in = np.atleast_1d(self.poles if self.poles is not None else [])
        if self.units == 'rad/s':
            z = z_in
            p = p_in
        elif self.units == 'hz':
            z = 2 * np.pi * z_in
            p = 2 * np.pi * p_in
        elif self.units == 'normalized':
            sampling_freq = 1 / self.t_step
            nyquist_freq_hz = sampling_freq / 2
            nyquist_freq_rads = 2 * np.pi * nyquist_freq_hz
            z = nyquist_freq_rads * z_in
            p = nyquist_freq_rads * p_in
        else:
            valid_units = ['rad/s', 'hz', 'normalized']
            raise ValueError(f'`units` must be one of {valid_units}.')
        self.ss_ct_ = scipy.signal.ZerosPolesGain(z, p, self.gain).to_ss()
        self.ss_dt_ = self.ss_ct_.to_discrete(
            self.t_step,
            self.discretization,
        )
        weight = (
            self.type,
            self.ss_dt_.A,
            self.ss_dt_.B,
            self.ss_dt_.C,
            self.ss_dt_.D,
        )
        self.hinf_regressor_ = sklearn.base.clone(self.hinf_regressor)
        self.hinf_regressor_.set_params(weight=weight)
        self.hinf_regressor_.fit(
            X,
            y,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        self.n_features_in_ = self.hinf_regressor_.n_features_in_
        self.n_states_in_ = self.hinf_regressor_.n_states_in_
        self.n_inputs_in_ = self.hinf_regressor_.n_inputs_in_
        self.episode_feature_ = self.hinf_regressor_.episode_feature_
        self.feature_names_in_ = self.hinf_regressor_.feature_names_in_
        self.coef_ = self.hinf_regressor_.coef_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform a single-step prediction for each state in each episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Predicted data matrix.
        """
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        return self.hinf_regressor_.predict(X)

    def _more_tags(self):
        return {
            'multioutput': True,
            'multioutput_only': True,
        }


def _create_ss(
    U: np.ndarray,
    weight: Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray,
                           np.ndarray]],
    Q_hat: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Augment Koopman system with weight if present.

    Parameters
    ----------
    U : np.ndarray
        Koopman matrix containing ``A`` and ``B`` concatenated
        horizontally.
    weight : Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray]]
        Tuple containing weight type (``'pre'`` or ``'post'``), and the
        weight state space matrices (``A``, ``B``, ``C``, and ``D``). If
        ``None``, no weighting is used.
    Q_hat : np.ndarray
        Left singular vectors of shifted data matrix. Used to construct ``C``
        matrix. Should only be used with DMDc methods.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Weighted state space matrices (``A``, ``B``, ``C``, ``D``).
    """
    p_theta = U.shape[0]
    if weight is None:
        A = U[:, :p_theta]
        B = U[:, p_theta:]
        C = picos.Constant('C',
                           Q_hat if Q_hat is not None else np.eye(p_theta))
        D = picos.Constant('D', np.zeros((C.shape[0], B.shape[1])))
    else:
        Am = U[:, :p_theta]
        Bm = U[:, p_theta:]
        Cm = picos.Constant('Cm',
                            Q_hat if Q_hat is not None else np.eye(p_theta))
        Dm = picos.Constant('Dm', np.zeros((Cm.shape[0], Bm.shape[1])))
        if weight[0] == 'pre':
            n_u = Bm.shape[1]
            Aw_blk = linalg.block_diag(*([weight[1]] * n_u))
            Bw_blk = linalg.block_diag(*([weight[2]] * n_u))
            Cw_blk = linalg.block_diag(*([weight[3]] * n_u))
            Dw_blk = linalg.block_diag(*([weight[4]] * n_u))
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
        elif weight[0] == 'post':
            n_x = Cm.shape[0]
            Aw_blk = linalg.block_diag(*([weight[1]] * n_x))
            Bw_blk = linalg.block_diag(*([weight[2]] * n_x))
            Cw_blk = linalg.block_diag(*([weight[3]] * n_x))
            Dw_blk = linalg.block_diag(*([weight[4]] * n_x))
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
                 alpha_other: float, square_norm: bool,
                 picos_eps: float) -> picos.Problem:
    """Add matrix two norm regularizer to an optimization problem.

    Parameters
    ----------
    problem : picos.Problem
        Optimization problem.
    U : picos.RealVariable
        Koopman matrix variable.
    alpha_other : float
        Regularization coefficient (already divided by ``q`` if applicable).
    square_norm : bool
        Square matrix two-norm.
    picos_eps : float
        Tolerance used for strict LMIs.

    Returns
    -------
    picos.Problem
        Optimization problem with regularizer added.
    """
    # Validate ``alpha``
    if alpha_other <= 0:
        raise ValueError('Parameter `alpha` must be positive.')
    # Extract information from problem
    direction = problem.objective.direction
    objective = problem.objective.function
    # Get needed sizes
    p_theta, p = U.shape
    # Add new constraint
    gamma = picos.RealVariable('gamma', 1)
    problem.add_constraint(
        picos.block([[picos.diag(gamma, p), U.T],
                     [U, picos.diag(gamma, p_theta)]]) >> picos_eps)
    # Add term to cost function
    alpha_scaled = picos.Constant('alpha_scaled_2', alpha_other)
    if square_norm:
        objective += alpha_scaled * gamma**2
    else:
        objective += alpha_scaled * gamma
    problem.set_objective(direction, objective)
    return problem


def _add_nuclear(problem: picos.Problem, U: picos.RealVariable,
                 alpha_other: float, square_norm: bool,
                 picos_eps: float) -> picos.Problem:
    """Add nuclear norm regularizer to an optimization problem.

    Parameters
    ----------
    problem : picos.Problem
        Optimization problem.
    U : picos.RealVariable
        Koopman matrix variable.
    alpha_other : float
        Regularization coefficient (already divided by ``q`` if applicable).
    square_norm : bool
        Square nuclear norm.
    picos_eps : float
        Tolerance used for strict LMIs.

    Returns
    -------
    picos.Problem
        Optimization problem with regularizer added.
    """
    # Validate ``alpha``
    if alpha_other <= 0:
        raise ValueError('Parameter `alpha` must be positive.')
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
    if square_norm:
        objective += alpha_scaled * gamma**2
    else:
        objective += alpha_scaled * gamma
    problem.set_objective(direction, objective)
    return problem


@memory.cache
def _calc_c_G_H(
    X_unshifted: np.ndarray,
    X_shifted: np.ndarray,
    alpha: float,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
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
    Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]
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
    log.info(f'`_calc_c_G_H()` stats: {stats_str}')
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
def _calc_QSig(X: np.ndarray, alpha: float, tsvd: tsvd.Tsvd) -> np.ndarray:
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
    tsvd : pykoop.Tsvd
        Truncated singular value object.

    Returns
    -------
    np.ndarray
        Split ``H`` matrix.
    """
    # SVD
    tsvd.fit(X.T)
    Qr = tsvd.left_singular_vectors_
    sr = tsvd.singular_values_
    # Regularize
    q = X.shape[0]
    # ``alpha`` is already divided by ``q`` to be consistent with ``G`` and
    # ``H``.
    sr_reg = np.sqrt((sr**2 / q) + alpha)
    Sr_reg = np.diag(sr_reg)
    # Multiply with Q and return
    QSig = Qr @ Sr_reg
    return QSig
    sr_reg = np.sqrt((sr**2 / q) + alpha)
    Sr_reg = np.diag(sr_reg)
    # Multiply with Q and return
    QSig = Qr @ Sr_reg
    return QSig
