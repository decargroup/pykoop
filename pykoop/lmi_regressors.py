"""Collection of LMI-based Koopman regressors."""

from . import koopman_pipeline


class LmiEdmdTikhonovReg(koopman_pipeline.KoopmanRegressor):
    """LMI-based EDMD with Tikhonov regularization."""

    # Default solver parameters
    _default_solver_params = {
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
    _check_X_y_params = {
        'multi_output': True,
        'y_numeric': True,
        'dtype': 'float64',
    }

    def __init__(self,
                 alpha: int = 0,
                 inv_method: str = 'svd',
                 picos_eps: float = 0,
                 solver_params: dict = None) -> None:
        """Instantiate :class:`LmiEdmdTikhonovReg`.

        Parameters
        ----------
        alpha : int
            Tikhonov regularization coefficient. Can be zero without
            introducing numerical problems.

        inv_method : str
            Method to handle or avoid inversion of the ``H`` matrix when
            forming the LMI problem. Possible values are

            - ``'inv'`` -- invert ``H`` directly,
            - ``'pinv'`` -- apply the Moore-Penrose pseudoinverse to ``H``,
            - ``'eig'`` --
            - ``'ldl'`` --
            - ``'chol'`` --
            - ``'sqrt'`` --
            - ``'svd'`` --

        picos_eps : float
            Tolerance used for strict LMIs. If nonzero, should be larger than
            solver tolerance.

        solver_params : dict
            Parameters passed to PICOS
            :func:`picos.modeling.problem.Problem.solve()`.
        """
        self.alpha = alpha
        self.inv_method = inv_method
        self.picos_eps = picos_eps
        self.solver_params = solver_params
