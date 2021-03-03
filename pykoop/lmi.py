import sklearn.base
import sklearn.utils.validation
import picos
import numpy as np
from scipy import linalg
import logging

# TODO Inheritance for regularizers to make some stuff more uniform
# TODO Warn that ratio=0 is bad!


class LmiEdmdTikhonovReg(sklearn.base.BaseEstimator,
                         sklearn.base.RegressorMixin):

    # The real number of minimum samples is based on the data. But this value
    # shuts up pytest for now. Picos also requires float64 so everything is
    # promoted.
    _check_X_y_params = {
        'multi_output': True,
        'y_numeric': True,
        'dtype': 'float64',
        'ensure_min_samples': 2,
    }

    # If H has a condition number higher than `_warn_cond`, a warning will be
    # generated.
    _warn_cond = 1e6

    def __init__(self, alpha=0.0, inv_method='eig', solver='mosek',
                 picos_eps=1e-9):
        self.alpha = alpha
        self.inv_method = inv_method
        self.solver = solver
        self.picos_eps = picos_eps

    def fit(self, X, y):
        self._validate_parameters()
        X, y = self._validate_data(X, y, reset=True, **self._check_X_y_params)
        self.alpha_tikhonov_reg_ = self.alpha
        problem = self._get_base_problem(X, y)
        problem.solve(solver=self.solver)
        self.coef_ = self._extract_solution(problem)
        return self

    def predict(self, X):
        X = self._validate_data(X, reset=False)
        sklearn.utils.validation.check_is_fitted(self)
        Psi = X.T
        Theta_p = self.coef_.T @ Psi
        return Theta_p.T

    def _validate_parameters(self):
        # Validate alpha
        if self.alpha < 0:
            raise ValueError('`alpha` must be greater than zero.')
        # Validate inverse method
        valid_inv_methods = ['inv', 'pinv', 'eig', 'ldl', 'chol', 'sqrt']
        if self.inv_method not in valid_inv_methods:
            raise ValueError('`inv_method` must be one of: '
                             f'{" ".join(valid_inv_methods)}.')
        # Validate solver
        valid_solvers = [None, 'mosek']
        if self.solver not in valid_solvers:
            raise ValueError('`solver` must be one of: '
                             f'{" ".join(valid_solvers)}.')

    def _validate_data(self, X, y=None, reset=True, **check_array_params):
        if y is None:
            X = sklearn.utils.validation.check_array(X, **check_array_params)
        else:
            X, y = sklearn.utils.validation.check_X_y(X, y,
                                                      **check_array_params)
        if reset:
            self.n_features_in_ = X.shape[1]
        return X if y is None else (X, y)

    def _get_base_problem(self, X, y):
        # Compute G and H
        Psi = X.T
        Theta_p = y.T
        p = Psi.shape[0]
        q = Psi.shape[1]
        # Compute G and Tikhonov-regularized H
        G = (Theta_p @ Psi.T) / q
        H = (Psi @ Psi.T + self.alpha_tikhonov_reg_ * np.eye(p)) / q
        # Check rank of H
        cond = np.linalg.cond(H)
        rk = np.linalg.matrix_rank(H)
        if cond > self._warn_cond:
            logging.warning('H has a high condition number. '
                            f'H is {H.shape[0]}x{H.shape[1]}. rk(H)={rk}. '
                            f'cond(H) = {cond}.')
        if rk < H.shape[0]:
            logging.warning(
                'H must be full rank. '
                f'H is {H.shape[0]}x{H.shape[1]}. rk(H)={rk}. '
                f'cond(H) = {cond}.'
            )
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
            H_inv = picos.Constant('H^-1', linalg.inv(H))
            problem.add_constraint((
                (  Z &     U) //  # noqa: E2
                (U.T & H_inv)
            ) >> 0)
        elif self.inv_method == 'pinv':
            H_inv = picos.Constant('H^+', linalg.pinv(H))
            problem.add_constraint((
                (  Z &     U) //  # noqa: E2
                (U.T & H_inv)
            ) >> 0)
        elif self.inv_method == 'eig':
            lmb, V = linalg.eigh(H)
            VsqrtLmb = picos.Constant('(V Lambda^(1/2))',
                                      V @ np.diag(np.sqrt(lmb)))
            problem.add_constraint((
                (               Z &       U * VsqrtLmb) //  # noqa: E2
                (VsqrtLmb.T * U.T & np.eye(U.shape[1]))     # noqa: E2
            ) >> 0)
        elif self.inv_method == 'ldl':
            L, D, _ = linalg.ldl(H)
            LsqrtD = picos.Constant('(L D^(1/2))', L @ np.sqrt(D))
            problem.add_constraint((
                (             Z &         U * LsqrtD) //  # noqa: E2
                (LsqrtD.T * U.T & np.eye(U.shape[1]))     # noqa: E2
            ) >> 0)
        elif self.inv_method == 'chol':
            L = picos.Constant('L', linalg.cholesky(H, lower=True))
            problem.add_constraint((
                (        Z &              U * L) //  # noqa: E2
                (L.T * U.T & np.eye(U.shape[1]))
            ) >> 0)
        elif self.inv_method == 'sqrt':
            # Since H is symmetric, its square root is symmetric.
            # Otherwise, this would not work!
            sqrtH = picos.Constant('sqrt(H)', linalg.sqrtm(H))
            problem.add_constraint((
                (            Z &          U * sqrtH) //  # noqa: E2
                (sqrtH.T * U.T & np.eye(U.shape[1]))
            ) >> 0)
        else:
            # Should never get here since input validation is done in `fit()`.
            raise ValueError('Invalid value for `inv_method`.')
        # Set objective
        problem.set_objective('min',
                              (-2 * picos.trace(U * G_T) + picos.trace(Z)))
        return problem

    def _extract_solution(self, problem):
        return np.array(problem.get_valued_variable('U'), ndmin=2).T

    def _more_tags(self):
        return {
            'multioutput': True,
            'multioutput_only': True,
        }


class LmiEdmdTwoNormReg(LmiEdmdTikhonovReg):

    def __init__(self, alpha=1.0, ratio=1.0, inv_method='eig', solver='mosek',
                 picos_eps=1e-9):
        self.alpha = alpha
        self.ratio = ratio
        self.inv_method = inv_method
        self.solver = solver
        self.picos_eps = picos_eps

    def fit(self, X, y):
        # TODO Warn if alpha is zero?
        self._validate_parameters()
        X, y = self._validate_data(X, y, reset=True, **self._check_X_y_params)
        self.alpha_tikhonov_reg_ = self.alpha * (1 - self.ratio)
        self.alpha_other_reg_ = self.alpha * self.ratio
        problem = self._get_base_problem(X, y)
        self._add_twonorm(X, y, problem)
        problem.solve(solver=self.solver)
        self.coef_ = self._extract_solution(problem)
        return self

    def _add_twonorm(self, X, y, problem):
        # Extract information from problem
        U = problem.variables['U']
        direction = problem.objective.direction
        objective = problem.objective.function
        # Get needed sizes
        p_theta = U.shape[0]
        p = U.shape[1]
        q = X.shape[0]
        # Add new constraint
        gamma = picos.RealVariable('gamma', 1)
        problem.add_constraint((
            (picos.diag(gamma, p) & U.T) //
            (U & picos.diag(gamma, p_theta))
        ) >> 0)
        # Add term to cost function
        alpha_scaled = picos.Constant('alpha_2/q', self.alpha_other_reg_/q)
        objective += alpha_scaled * gamma**2
        problem.set_objective(direction, objective)


class LmiEdmdNuclearNormReg(LmiEdmdTikhonovReg):

    def __init__(self, alpha=1.0, ratio=1.0, inv_method='eig', solver='mosek',
                 picos_eps=1e-9):
        self.alpha = alpha
        self.ratio = ratio
        self.inv_method = inv_method
        self.solver = solver
        self.picos_eps = picos_eps

    def fit(self, X, y):
        self._validate_parameters()
        X, y = self._validate_data(X, y, reset=True, **self._check_X_y_params)
        self.alpha_tikhonov_reg_ = self.alpha * (1 - self.ratio)
        self.alpha_other_reg_ = self.alpha * self.ratio
        problem = self._get_base_problem(X, y)
        self._add_nuclear(X, y, problem)
        problem.solve(solver=self.solver)
        self.coef_ = self._extract_solution(problem)
        return self

    def _add_nuclear(self, X, y, problem):
        # Extract information from problem
        U = problem.variables['U']
        direction = problem.objective.direction
        objective = problem.objective.function
        # Get needed sizes
        p_theta = U.shape[0]
        p = U.shape[1]
        q = X.shape[0]
        # Add new constraint
        gamma = picos.RealVariable('gamma', 1)
        W_1 = picos.SymmetricVariable('W_1', (p_theta, p_theta))
        W_2 = picos.SymmetricVariable('W_2', (p, p))
        problem.add_constraint(picos.trace(W_1) + picos.trace(W_2)
                               <= 2 * gamma)
        problem.add_constraint((
            (W_1 & U) //
            (U.T & W_2)
        ) >> 0)
        # Add term to cost function
        alpha_scaled = picos.Constant('alpha_*/q', self.alpha_other_reg_/q)
        objective += alpha_scaled * gamma**2
        problem.set_objective(direction, objective)


class LmiEdmdSpectralRadiusConstr(LmiEdmdTikhonovReg):

    def __init__(self, rho_bar=1.0, alpha=0, max_iter=100, tol=1e-6,
                 inv_method='eig', solver='mosek', picos_eps=1e-9):
        self.rho_bar = rho_bar
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.inv_method = inv_method
        self.solver = solver
        self.picos_eps = picos_eps

    def fit(self, X, y):
        self._validate_parameters()
        X, y = self._validate_data(X, y, reset=True, **self._check_X_y_params)
        self.alpha_tikhonov_reg_ = self.alpha
        # Get needed sizes
        p_theta = y.shape[1]
        p = X.shape[1]
        # Make initial guesses and iterate
        Gamma = np.eye(p_theta)
        U_prev = np.zeros((p_theta, p))
        for k in range(self.max_iter):
            # Formulate Problem A
            problem_a = self._get_problem_a(X, y, Gamma)
            # Solve Problem A
            problem_a.solve(solver=self.solver)
            U = np.array(problem_a.get_valued_variable('U'), ndmin=2)
            P = np.array(problem_a.get_valued_variable('P'), ndmin=2)
            # Formulate Problem B
            problem_b = self._get_problem_b(X, y, U, P)
            # Solve Problem B
            problem_b.solve(solver=self.solver)
            Gamma = np.array(problem_b.get_valued_variable('Gamma'), ndmin=2)
            # Check stopping condition
            difference = _fast_frob_norm(U_prev - U)
            if (difference < self.tol):
                self.stop_reason_ = 'Reached tolerance {self.tol}'
                break
            U_prev = U
        else:
            self.stop_reason_ = 'Reached maximum iterations {self.max_iter}'
        self.tol_reached_ = difference
        self.n_iter_ = k + 1
        self.coef_ = U.T
        # Only useful for debugging
        self.Gamma_ = Gamma
        self.P_ = P
        return self

    def _get_problem_a(self, X, y, Gamma):
        problem_a = self._get_base_problem(X, y)
        # Extract information from problem
        U = problem_a.variables['U']
        # Get needed sizes
        p_theta = U.shape[0]
        # Add new constraints
        rho_bar_sq = picos.Constant('rho_bar^2', self.rho_bar**2)
        Gamma = picos.Constant('Gamma', Gamma)
        P = picos.SymmetricVariable('P', p_theta)
        problem_a.add_constraint(P >> self.picos_eps)
        problem_a.add_constraint((
            (rho_bar_sq * P & U[:, :p_theta].T * Gamma) //
            (Gamma.T * U[:, :p_theta] & Gamma + Gamma.T - P)
        ) >> self.picos_eps)
        return problem_a

    def _get_problem_b(self, X, y, U, P):
        # Create optimization problem
        problem_b = picos.Problem()
        # Get needed sizes
        p_theta = U.shape[0]
        # Create constants
        rho_bar_sq = picos.Constant('rho_bar^2', self.rho_bar**2)
        U = picos.Constant('U', U)
        P = picos.Constant('P', P)
        # Create variables
        Gamma = picos.RealVariable('Gamma', P.shape)
        # Add constraints
        problem_b.add_constraint((
            (rho_bar_sq * P & U[:, :p_theta].T * Gamma) //
            (Gamma.T * U[:, :p_theta] & Gamma + Gamma.T - P)
        ) >> self.picos_eps)
        # Set objective
        problem_b.set_objective('find')
        return problem_b


class LmiEdmdHinfReg(LmiEdmdTikhonovReg):

    def __init__(self, alpha=1.0, ratio=1.0, max_iter=100, tol=1e-6,
                 inv_method='eig', solver='mosek', picos_eps=1e-9):
        self.alpha = alpha
        self.ratio = ratio
        self.max_iter = max_iter
        self.tol = tol
        self.inv_method = inv_method
        self.solver = solver
        self.picos_eps = picos_eps

    def fit(self, X, y):
        self._validate_parameters()
        X, y = self._validate_data(X, y, reset=True, **self._check_X_y_params)
        self.alpha_tikhonov_reg_ = self.alpha * (1 - self.ratio)
        self.alpha_other_reg_ = self.alpha * self.ratio
        # Get needed sizes
        p_theta = y.shape[1]
        p = X.shape[1]
        # Check that at least one input is present
        if p_theta == p:
            # If you remove the `{p} features(s)` part of this message,
            # the scikit-learn estimator_checks will fail!
            raise ValueError('LmiEdmdHinfReg() requires an input to function.'
                             '`X` and `y` must therefore have different '
                             'numbers of features. `X and y` both have '
                             f'{p} feature(s).')
        # Make initial guesses and iterate
        P = np.eye(p_theta)
        U_prev = np.zeros((p_theta, p))
        for k in range(self.max_iter):
            # Formulate Problem A
            problem_a = self._get_problem_a(X, y, P)
            # Solve Problem A
            problem_a.solve(solver=self.solver)
            U = np.array(problem_a.get_valued_variable('U'), ndmin=2)
            gamma = np.array(problem_a.get_valued_variable('gamma'))
            # Formulate Problem B
            problem_b = self._get_problem_b(X, y, U, gamma)
            # Solve Problem B
            problem_b.solve(solver=self.solver)
            P = np.array(problem_b.get_valued_variable('P'), ndmin=2)
            # Check stopping condition
            difference = _fast_frob_norm(U_prev - U)
            if (difference < self.tol):
                self.stop_reason_ = 'Reached tolerance {self.tol}'
                break
            U_prev = U
        else:
            self.stop_reason_ = 'Reached maximum iterations {self.max_iter}'
        self.tol_reached_ = difference
        self.n_iter_ = k + 1
        self.coef_ = U.T
        # Only useful for debugging
        self.P_ = P
        self.gamma_ = gamma
        return self

    def _get_problem_a(self, X, y, P):
        problem_a = self._get_base_problem(X, y)
        # Extract information from problem
        U = problem_a.variables['U']
        direction = problem_a.objective.direction
        objective = problem_a.objective.function
        # Get needed sizes
        p_theta = U.shape[0]
        q = X.shape[0]
        # Add new constraint
        P = picos.Constant('P', P)
        gamma = picos.RealVariable('gamma', 1)
        A = U[:p_theta, :p_theta]
        B = U[:p_theta, p_theta:]
        C = np.eye(p_theta)
        D = np.zeros((C.shape[0], B.shape[1]))
        zero14 = np.zeros(C.T.shape)
        zero23 = np.zeros(B.shape)
        geye1 = gamma * np.eye(D.shape[1])
        geye2 = gamma * np.eye(D.shape[0])
        problem_a.add_constraint((
            (       P &      A*P &      B & zero14) //  # noqa
            ( P.T*A.T &        P & zero23 &  P*C.T) //  # noqa
            (     B.T & zero23.T &  geye1 &    D.T) //  # noqa
            (zero14.T &    C*P.T &      D &  geye2)     # noqa
        ) >> self.picos_eps)
        # Add term to cost function
        alpha_scaled = picos.Constant('alpha_inf/q', self.alpha_other_reg_/q)
        objective += alpha_scaled * gamma**2
        problem_a.set_objective(direction, objective)
        return problem_a

    def _get_problem_b(self, X, y, U, gamma):
        # Create optimization problem
        problem_b = picos.Problem()
        # Get needed sizes
        p_theta = U.shape[0]
        # Create constants
        U = picos.Constant('U', U)
        gamma = picos.Constant('gamma', gamma)
        # Create variables
        P = picos.SymmetricVariable('P', p_theta)
        # Add constraints
        problem_b.add_constraint(P >> self.picos_eps)
        A = U[:p_theta, :p_theta]
        B = U[:p_theta, p_theta:]
        C = np.eye(p_theta)
        D = np.zeros((C.shape[0], B.shape[1]))
        zero14 = np.zeros(C.T.shape)
        zero23 = np.zeros(B.shape)
        geye1 = gamma * np.eye(D.shape[1])
        geye2 = gamma * np.eye(D.shape[0])
        problem_b.add_constraint((
            (       P &      A*P &      B & zero14) //  # noqa
            ( P.T*A.T &        P & zero23 &  P*C.T) //  # noqa
            (     B.T & zero23.T &  geye1 &    D.T) //  # noqa
            (zero14.T &    C*P.T &      D &  geye2)     # noqa
        ) >> self.picos_eps)
        # Set objective
        problem_b.set_objective('find')
        return problem_b


class LmiEdmdDissipativityConstr(LmiEdmdTikhonovReg):
    """See hara_2019_learning

    Supply rate:
        s(u, y) = -[y, u] Xi [y; u],
    where
        Xi = [0, -1; -1, 0] -> passivity,
        Xi = [1, 0; 0, gamma] -> bounded L2 gain of gamma.

    (Using MATLAB array notation here. To clean up and LaTeX)


    @article{hara_2019_learning,
        title={Learning Koopman Operator under Dissipativity Constraints},
        author={Keita Hara and Masaki Inoue and Noboru Sebe},
        year={2019},
        journaltitle={{\tt arXiv:1911.03884v1 [eess.SY]}}
    }

    Currently not fully tested!
    """

    def __init__(self, max_iter=100, tol=1e-6,
                 inv_method='eig', solver='mosek', picos_eps=1e-9):
        self.max_iter = max_iter
        self.tol = tol
        self.inv_method = inv_method
        self.solver = solver
        self.picos_eps = picos_eps

    def fit(self, X, y, supply_rate_xi=None):
        self._validate_parameters()
        X, y = self._validate_data(X, y, reset=True, **self._check_X_y_params)
        self.supply_rate_xi_ = supply_rate_xi
        # Get needed sizes
        p_theta = y.shape[1]
        p = X.shape[1]
        # Make initial guess and iterate
        P = np.eye(p_theta)
        U_prev = np.zeros((p_theta, p))
        for k in range(self.max_iter):
            # Formulate Problem A
            problem_a = self._get_problem_a(X, y, P)
            # Solve Problem A
            problem_a.solve(solver=self.solver)
            U = np.array(problem_a.get_valued_variable('U'), ndmin=2)
            # Formulate Problem B
            problem_b = self._get_problem_b(X, y, U)
            # Solve Problem B
            problem_b.solve(solver=self.solver)
            P = np.array(problem_b.get_valued_variable('P'), ndmin=2)
            # Check stopping condition
            difference = _fast_frob_norm(U_prev - U)
            if (difference < self.tol):
                self.stop_reason_ = 'Reached tolerance {self.tol}'
                break
            U_prev = U
        else:
            self.stop_reason_ = 'Reached maximum iterations {self.max_iter}'
        self.tol_reached_ = difference
        self.n_iter_ = k + 1
        self.coef_ = U.T
        # Only useful for debugging
        self.P_ = P
        return self

    def _get_problem_a(self, X, y, P):
        problem_a = self._get_base_problem(X, y)
        # Extract information from problem
        U = problem_a.variables['U']
        # Get needed sizes
        p_theta = U.shape[0]
        # Add new constraints
        P = picos.Constant('P', P)
        A = U[:, :p_theta]
        B = U[:, p_theta:]
        C = picos.Constant('C', np.eye(p_theta))
        Xi11 = picos.Constant('Xi_11',
                              self.supply_rate_xi_[:p_theta, :p_theta])
        Xi12 = picos.Constant('Xi_12',
                              self.supply_rate_xi_[:p_theta, p_theta:])
        Xi22 = picos.Constant('Xi_22',
                              self.supply_rate_xi_[p_theta:, p_theta:])
        problem_a.add_constraint((
            (P - C.T*Xi11*C & -C.T*Xi12 & A.T*P) //
            (     -Xi12.T*C &     -Xi22 & B.T*P) //  # noqa: E201 E221 E222
            (           P*A &       P*B &     P)     # noqa: E201 E222
        ) >> self.picos_eps)
        return problem_a

    def _get_problem_b(self, X, y, U):
        # Create optimization problem
        problem_b = picos.Problem()
        # Get needed sizes
        p_theta = U.shape[0]
        # Create constants
        U = picos.Constant('U', U)
        # Create variables
        P = picos.SymmetricVariable('P', p_theta)
        # Add constraints
        A = U[:, :p_theta]
        B = U[:, p_theta:]
        C = picos.Constant('C', np.eye(p_theta))
        Xi11 = picos.Constant('Xi_11',
                              self.supply_rate_xi_[:p_theta, :p_theta])
        Xi12 = picos.Constant('Xi_12',
                              self.supply_rate_xi_[:p_theta, p_theta:])
        Xi22 = picos.Constant('Xi_22',
                              self.supply_rate_xi_[p_theta:, p_theta:])
        problem_b.add_constraint(P >> self.picos_eps)
        problem_b.add_constraint((
            (P - C.T*Xi11*C & -C.T*Xi12 & A.T*P) //
            (     -Xi12.T*C &     -Xi22 & B.T*P) //  # noqa: E201 E221 E222
            (           P*A &       P*B &     P)     # noqa: E201 E222
        ) >> self.picos_eps)
        # Set objective
        problem_b.set_objective('find')
        return problem_b


def _fast_frob_norm(A):
    # Maybe this is premature optimization but scipy.linalg.norm() is
    # notoriously slow.
    return np.sqrt(np.trace(A @ A.T))
