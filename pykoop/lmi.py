import sklearn.base
import sklearn.utils.validation
import picos
import numpy as np
from scipy import linalg


class LmiEdmd(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):

    # TODO: The real number of minimum samples is based on the data. But this
    # value shuts up pytest for now. Picos also requires float64 so everything
    # is promoted.
    _check_X_y_params = {
        'multi_output': True,
        'y_numeric': True,
        'dtype': 'float64',
        'ensure_min_samples': 2,
    }

    def __init__(self, inv_method='eig', solver='mosek', picos_eps=1e-9):
        self.inv_method = inv_method
        self.solver = solver
        self.picos_eps = picos_eps

    def fit(self, X, y):
        self._validate_parameters()
        X, y = self._validate_data(X, y, reset=True, **self._check_X_y_params)
        problem = self._base_problem(X, y)
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
        # Validate inverse method
        valid_inv_methods = ['inv', 'eig', 'ldl', 'chol', 'sqrt']
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

    def _compute_G_H(self, Psi, Theta_p, q):
        G = (Theta_p @ Psi.T) / q
        H = (Psi @ Psi.T) / q
        return G, H

    def _base_problem(self, X, y):
        # Compute G and H
        Psi = X.T
        Theta_p = y.T
        q = Psi.shape[1]
        G, H = self._compute_G_H(Psi, Theta_p, q)
        # Check rank of H
        rk = np.linalg.matrix_rank(H)
        if rk < H.shape[0]:
            # TODO Is it possible to use a pseudo-inverse here?
            # That would mean H could be rank deficient.
            # TODO Condition number warning? Log it?
            raise ValueError(
                'H must be full rank. '
                f'H is {H.shape[0]}x{X.shape[1]}. rk(H)={rk}.'
                'TODO: Loosen this requirement with a '
                'psuedo-inverse?'
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


class LmiEdmdTikhonovReg(LmiEdmd):

    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _compute_G_H(self, Psi, Theta_p, q):
        G = (Theta_p @ Psi.T) / q
        # Add in regularizer to H
        H = (Psi @ Psi.T + self.alpha * np.eye(Psi.shape[0])) / q
        return G, H


class LmiEdmdTwoNormReg(LmiEdmd):

    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def fit(self, X, y):
        # TODO Warn if alpha is zero?
        self._validate_parameters()
        X, y = self._validate_data(X, y, reset=True, **self._check_X_y_params)
        problem = self._base_problem(X, y)
        self._add_tikhonov(X, y, problem)
        problem.solve(solver=self.solver)
        self.coef_ = self._extract_solution(problem)
        return self

    def _add_tikhonov(self, X, y, problem):
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
        alpha_scaled = picos.Constant('alpha/q', self.alpha/q)
        objective += alpha_scaled * gamma**2
        problem.set_objective(direction, objective)


class LmiEdmdNuclearNormReg(LmiEdmd):

    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def fit(self, X, y):
        # TODO Warn if alpha is zero?
        self._validate_parameters()
        X, y = self._validate_data(X, y, reset=True, **self._check_X_y_params)
        problem = self._base_problem(X, y)
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
        alpha_scaled = picos.Constant('alpha/q', self.alpha/q)
        objective += alpha_scaled * gamma**2
        problem.set_objective(direction, objective)


class LmiEdmdSpectralRadiusConstr(LmiEdmd):

    def __init__(self, rho_bar=1.0, max_iter=100, tol=1e-6, init_guess=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.rho_bar = rho_bar
        self.max_iter = max_iter
        self.tol = tol
        self.init_guess = init_guess

    def fit(self, X, y):
        # TODO Warn if alpha is zero?
        self._validate_parameters()
        X, y = self._validate_data(X, y, reset=True, **self._check_X_y_params)
        # Get needed sizes
        p_theta = y.shape[1]
        p = X.shape[1]
        # Make initial guesses and iterate
        Gamma = np.eye(p_theta) if self.init_guess is None else self.init_guess
        U_prev = np.zeros((p_theta, p))
        for k in range(self.max_iter):
            # Formulate Problem A
            problem_a = self._base_problem(X, y)
            self._add_constraint_a(X, y, Gamma, problem_a)
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

    def _add_constraint_a(self, X, y, Gamma, problem_a):
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
        problem_b.add_constraint((
            (rho_bar_sq * P & U[:, :p_theta].T * Gamma) //
            (Gamma.T * U[:, :p_theta] & Gamma + Gamma.T - P)
        ) >> self.picos_eps)
        problem_b.set_objective('find')
        return problem_b


def _fast_frob_norm(A):
    # scipy.linalg.norm() is notoriously slow.
    # Maybe this is premature optimization... sue me.
    return np.sqrt(np.trace(A @ A.T))
