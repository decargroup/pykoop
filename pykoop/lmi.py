import sklearn.base
import sklearn.utils.validation
import picos
import numpy as np
from scipy import linalg
import logging
import tempfile
import joblib

# TODO Decide how to handle solve(primals=..., duals=...).
# Should it crash when it can't find a solution?
# Should I just catch the exception?

cachedir = tempfile.TemporaryDirectory(prefix='pykoop_')
logging.info(f'Temporary directory created at `{cachedir.name}`')
memory = joblib.Memory(cachedir.name, verbose=0)


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

    def __init__(self, alpha=0.0, inv_method='chol', picos_eps=0,
                 solver_params=None):
        self.alpha = alpha
        self.inv_method = inv_method
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def fit(self, X, y):
        self._validate_parameters()
        X, y = self._validate_data(X, y, reset=True, **self._check_X_y_params)
        self.alpha_tikhonov_reg_ = self.alpha
        problem = self._get_base_problem(X, y)
        problem.solve(**self.solver_params_)
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
                             f'{", ".join(valid_inv_methods)}.')
        # Set solver params
        if self.solver_params is None:
            self.solver_params_ = {}

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
        G, H, stats = _calc_G_H(X, y, self.alpha_tikhonov_reg_)
        logging.info("_calc_G_H() stats: " + str(stats))
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
                [  Z,     U],  # noqa: E201
                [U.T, H_inv]
            ]) >> 0)
        elif self.inv_method == 'pinv':
            H_inv = picos.Constant('H^+', _calc_Hpinv(H))
            problem.add_constraint(picos.block([
                [  Z,     U],  # noqa: E201
                [U.T, H_inv]
            ]) >> 0)
        elif self.inv_method == 'eig':
            VsqrtLmb = picos.Constant('(V Lambda^(1/2))', _calc_VsqrtLmb(H))
            problem.add_constraint(picos.block([
                [               Z, U * VsqrtLmb],  # noqa: E201
                [VsqrtLmb.T * U.T,          'I']
            ]) >> 0)
        elif self.inv_method == 'ldl':
            LsqrtD = picos.Constant('(L D^(1/2))', _calc_LsqrtD(H))
            problem.add_constraint(picos.block([
                [             Z, U * LsqrtD],  # noqa: E201
                [LsqrtD.T * U.T,        'I']
            ]) >> 0)
        elif self.inv_method == 'chol':
            L = picos.Constant('L', _calc_L(H))
            problem.add_constraint(picos.block([
                [        Z, U * L],  # noqa: E201
                [L.T * U.T,   'I']
            ]) >> 0)
        elif self.inv_method == 'sqrt':
            sqrtH = picos.Constant('sqrt(H)', _calc_sqrtH(H))
            problem.add_constraint(picos.block([
                [            Z, U * sqrtH],  # noqa: E201
                [sqrtH.T * U.T,       'I']
            ]) >> 0)
        else:
            # Should never get here since input validation is done in `fit()`.
            raise ValueError('Invalid value for `inv_method`.')
        # Set objective
        obj = -2 * picos.trace(U * G_T) + picos.trace(Z)
        problem.set_objective('min', obj)
        return problem

    def _extract_solution(self, problem):
        return np.array(problem.get_valued_variable('U'), ndmin=2).T

    def _more_tags(self):
        return {
            'multioutput': True,
            'multioutput_only': True,
        }


class LmiEdmdTwoNormReg(LmiEdmdTikhonovReg):

    def __init__(self, alpha=1.0, ratio=1.0, inv_method='chol', picos_eps=0,
                 solver_params=None):
        self.alpha = alpha
        self.ratio = ratio
        self.inv_method = inv_method
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def fit(self, X, y):
        self._validate_parameters()
        X, y = self._validate_data(X, y, reset=True, **self._check_X_y_params)
        self.alpha_tikhonov_reg_ = self.alpha * (1 - self.ratio)
        self.alpha_other_reg_ = self.alpha * self.ratio
        problem = self._get_base_problem(X, y)
        self._add_twonorm(X, y, problem)
        problem.solve(**self.solver_params_)
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
        problem.add_constraint(picos.block([
            [picos.diag(gamma, p),                        U.T],
            [                   U, picos.diag(gamma, p_theta)]  # noqa: E201
        ]) >> 0)
        # Add term to cost function
        alpha_scaled = picos.Constant('alpha_2/q', self.alpha_other_reg_/q)
        objective += alpha_scaled * gamma
        problem.set_objective(direction, objective)

    def _validate_parameters(self):
        if self.alpha == 0:
            raise ValueError('Value of `alpha` should not be zero. Use '
                             '`LmiEdmdTikhonovReg()` if you want to disable '
                             'regularization.')
        if self.ratio == 0:
            raise ValueError('Value of `ratio` should not be zero. Use '
                             '`LmiEdmdTikhonovReg()` if you want to disable '
                             'regularization.')
        super()._validate_parameters()


class LmiEdmdNuclearNormReg(LmiEdmdTikhonovReg):

    def __init__(self, alpha=1.0, ratio=1.0, inv_method='chol', picos_eps=0,
                 solver_params=None):
        self.alpha = alpha
        self.ratio = ratio
        self.inv_method = inv_method
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def fit(self, X, y):
        self._validate_parameters()
        X, y = self._validate_data(X, y, reset=True, **self._check_X_y_params)
        self.alpha_tikhonov_reg_ = self.alpha * (1 - self.ratio)
        self.alpha_other_reg_ = self.alpha * self.ratio
        problem = self._get_base_problem(X, y)
        self._add_nuclear(X, y, problem)
        problem.solve(**self.solver_params_)
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
        problem.add_constraint(picos.block([
            [W_1,   U],
            [U.T, W_2]
        ]) >> 0)
        # Add term to cost function
        alpha_scaled = picos.Constant('alpha_*/q', self.alpha_other_reg_/q)
        objective += alpha_scaled * gamma
        problem.set_objective(direction, objective)

    def _validate_parameters(self):
        if self.alpha == 0:
            raise ValueError('Value of `alpha` should not be zero. Use '
                             '`LmiEdmdTikhonovReg()` if you want to disable '
                             'regularization.')
        if self.ratio == 0:
            raise ValueError('Value of `ratio` should not be zero. Use '
                             '`LmiEdmdTikhonovReg()` if you want to disable '
                             'regularization.')
        super()._validate_parameters()


class LmiEdmdSpectralRadiusConstr(LmiEdmdTikhonovReg):

    def __init__(self, rho_bar=1.0, alpha=0, max_iter=100, tol=1e-6,
                 inv_method='chol', picos_eps=0, solver_params=None):
        self.rho_bar = rho_bar
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.inv_method = inv_method
        self.picos_eps = picos_eps
        self.solver_params = solver_params

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
            problem_a.solve(**self.solver_params_)
            U = np.array(problem_a.get_valued_variable('U'), ndmin=2)
            P = np.array(problem_a.get_valued_variable('P'), ndmin=2)
            # Formulate Problem B
            problem_b = self._get_problem_b(X, y, U, P)
            # Solve Problem B
            problem_b.solve(**self.solver_params_)
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
        problem_a.add_constraint(picos.block([
            [          rho_bar_sq * P, U[:, :p_theta].T * Gamma],  # noqa: E201
            [Gamma.T * U[:, :p_theta],      Gamma + Gamma.T - P]
        ]) >> self.picos_eps)
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
        problem_b.add_constraint(picos.block([
            [          rho_bar_sq * P, U[:, :p_theta].T * Gamma],  # noqa: E201
            [Gamma.T * U[:, :p_theta],      Gamma + Gamma.T - P]
        ]) >> self.picos_eps)
        # Set objective
        problem_b.set_objective('find')
        return problem_b


class LmiEdmdHinfReg(LmiEdmdTikhonovReg):

    def __init__(self, alpha=1.0, ratio=1.0, max_iter=100, tol=1e-6,
                 inv_method='chol', picos_eps=0, solver_params=None):
        self.alpha = alpha
        self.ratio = ratio
        self.max_iter = max_iter
        self.tol = tol
        self.inv_method = inv_method
        self.picos_eps = picos_eps
        self.solver_params = solver_params

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
            problem_a.solve(**self.solver_params_)
            U = np.array(problem_a.get_valued_variable('U'), ndmin=2)
            gamma = np.array(problem_a.get_valued_variable('gamma'))
            # Formulate Problem B
            problem_b = self._get_problem_b(X, y, U, gamma)
            # Solve Problem B
            problem_b.solve(**self.solver_params_)
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
        gamma_33 = picos.diag(gamma, D.shape[1])
        gamma_44 = picos.diag(gamma, D.shape[0])
        problem_a.add_constraint(picos.block([
            [      P,   A*P,        B,        0],  # noqa: E201
            [P.T*A.T,     P,        0,    P*C.T],
            [    B.T,     0, gamma_33,      D.T],  # noqa: E201
            [      0, C*P.T,        D, gamma_44]   # noqa: E201
        ]) >> self.picos_eps)
        # Add term to cost function
        alpha_scaled = picos.Constant('alpha_inf/q', self.alpha_other_reg_/q)
        objective += alpha_scaled * gamma
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
        gamma_33 = picos.diag(gamma, D.shape[1])
        gamma_44 = picos.diag(gamma, D.shape[0])
        problem_b.add_constraint(picos.block([
            [      P,   A*P,        B,        0],  # noqa: E201
            [P.T*A.T,     P,        0,    P*C.T],
            [    B.T,     0, gamma_33,      D.T],  # noqa: E201
            [      0, C*P.T,        D, gamma_44]   # noqa: E201
        ]) >> self.picos_eps)
        # Set objective
        problem_b.set_objective('find')
        return problem_b

    def _validate_parameters(self):
        if self.alpha == 0:
            raise ValueError('Value of `alpha` should not be zero. Use '
                             '`LmiEdmdTikhonovReg()` if you want to disable '
                             'regularization.')
        if self.ratio == 0:
            raise ValueError('Value of `ratio` should not be zero. Use '
                             '`LmiEdmdTikhonovReg()` if you want to disable '
                             'regularization.')
        super()._validate_parameters()


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

    def __init__(self, alpha=0.0, max_iter=100, tol=1e-6, inv_method='chol',
                 picos_eps=0, solver_params=None):
        self.alpha = 0
        self.max_iter = max_iter
        self.tol = tol
        self.inv_method = inv_method
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def fit(self, X, y, supply_rate_xi=None):
        self._validate_parameters()
        X, y = self._validate_data(X, y, reset=True, **self._check_X_y_params)
        self.alpha_tikhonov_reg_ = self.alpha
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
            problem_a.solve(**self.solver_params_)
            U = np.array(problem_a.get_valued_variable('U'), ndmin=2)
            # Formulate Problem B
            problem_b = self._get_problem_b(X, y, U)
            # Solve Problem B
            problem_b.solve(**self.solver_params_)
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
        problem_a.add_constraint(picos.block([
            [P - C.T*Xi11*C, -C.T*Xi12, A.T*P],
            [     -Xi12.T*C,     -Xi22, B.T*P],  # noqa: E201 E221
            [           P*A,       P*B,     P]   # noqa: E201
        ]) >> self.picos_eps)
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
        problem_b.add_constraint(picos.block([
            [P - C.T*Xi11*C, -C.T*Xi12, A.T*P],
            [     -Xi12.T*C,     -Xi22, B.T*P],  # noqa: E201 E221
            [           P*A,       P*B,     P],  # noqa: E201 E222
        ]) >> self.picos_eps)
        # Set objective
        problem_b.set_objective('find')
        return problem_b


def _calc_G_H(X, y, alpha):
    """Memoized computation of ``G`` and ``H``. If this function is called
    a second time with the same parameters, cached versions of ``G`` and
    ``H`` are returned.
    """
    # Compute G and H
    Psi = X.T
    Theta_p = y.T
    p = Psi.shape[0]
    q = Psi.shape[1]
    # Compute G and Tikhonov-regularized H
    G = (Theta_p @ Psi.T) / q
    H_unreg = (Psi @ Psi.T) / q
    H_reg = H_unreg + (alpha * np.eye(p)) / q
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
    return G, H_reg, stats


@memory.cache
def _calc_Hinv(H):
    return linalg.inv(H)


@memory.cache
def _calc_Hpinv(H):
    return linalg.pinv(H)


@memory.cache
def _calc_VsqrtLmb(H):
    lmb, V = linalg.eigh(H)
    return V @ np.diag(np.sqrt(lmb))


@memory.cache
def _calc_LsqrtD(H):
    L, D, _ = linalg.ldl(H)
    return L @ np.sqrt(D)


@memory.cache
def _calc_L(H):
    return linalg.cholesky(H, lower=True)


@memory.cache
def _calc_sqrtH(H):
    # Since H is symmetric, its square root is symmetric.
    # Otherwise, this would not work!
    return linalg.sqrtm(H)


def _fast_frob_norm(A):
    # Maybe this is premature optimization but scipy.linalg.norm() is
    # notoriously slow.
    return np.sqrt(np.trace(A @ A.T))
