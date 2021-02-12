import sklearn.base
import sklearn.utils.validation
import picos
import numpy as np
from scipy import linalg


class LmiKoopBaseRegressor(sklearn.base.BaseEstimator,
                           sklearn.base.RegressorMixin):

    def __init__(self, solver='mosek', inv_method='eig', picos_eps=1e-9):
        self.solver = solver
        self.inv_method = inv_method
        self.picos_eps = picos_eps

    def fit(self, X, y):
        # Validate solver
        valid_solvers = [None, 'mosek']
        if self.solver not in valid_solvers:
            raise ValueError('`solver` must be one of: '
                             f'{" ".join(valid_solvers)}.')
        # Validate inverse method
        valid_inv_methods = ['inv', 'eig', 'ldl']
        if self.inv_method not in valid_inv_methods:
            raise ValueError('`inv_method` must be one of: '
                             f'{" ".join(valid_inv_methods)}.')
        # Validate data
        # TODO ensure_min_samples should really be X.shape[1], but that's not
        # super straightforward right now. Hard-coding 2 shuts up the pytest
        # failure. But H needs to be full rank for the algorithm to work!!
        # Check the rank before starting!
        X, y = sklearn.utils.validation.check_X_y(X, y, multi_output=True,
                                                  y_numeric=True,
                                                  dtype='float64',
                                                  ensure_min_samples=2)
        Psi = X.T
        Theta_p = y.T
        q = Psi.shape[1]
        G = (Theta_p @ Psi.T) / q
        H = (Psi @ Psi.T) / q
        problem = self._base_problem(G, H)
        problem.solve(solver=self.solver)
        self.U_ = np.array(problem.get_valued_variable('U'))
        self.n_features_in_ = Psi.shape[0]
        return self

    def predict(self, X):
        X = sklearn.utils.validation.check_array(X)
        sklearn.utils.validation.check_is_fitted(self)
        Psi = X.T
        Theta_p = self.U_ @ Psi
        return Theta_p.T

    def _base_problem(self, G, H):
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
        else:
            # Should never get here since input validation is done in `fit()`.
            raise ValueError('Invalid value for `inv_method`.')
        # Set objective
        problem.set_objective('min',
                              (-2 * picos.trace(U * G_T) + picos.trace(Z)))
        return problem

    def _more_tags(self):
        return {
            'multioutput': True,
            'multioutput_only': True,
        }
