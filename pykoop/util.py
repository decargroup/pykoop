"""Utilities for data generation and preprocessing."""

from typing import Any, Dict, Tuple

import numpy as np
from scipy import interpolate, signal

from . import dynamic_models, koopman_pipeline


class AnglePreprocessor(koopman_pipeline.EpisodeIndependentLiftingFn):
    """Preprocessor used to replace angles with their cosines and sines.

    Even though it inherits from :class:`EpisodeIndependengLiftingFn`, this
    class is intended as a preprocessor to be applied once to the input, rather
    than as a lifting function.

    Attributes
    ----------
    angles_in_ : np.ndarray
        Boolean array that indicates which input features are angles.
    lin_out_ : np.ndarray
        Boolean array that indicates which output features are linear.
    cos_out_ : np.ndarray
        Boolean array that indicates which output features are cosines.
    sin_out_ : np.ndarray
        Boolean array that indicates which output features are sines.
    n_features_in_ : int
        Number of features before transformation, including episode feature.
    n_states_in_ : int
        Number of states before transformation.
    n_inputs_in_ : int
        Number of inputs before transformation.
    n_features_out_ : int
        Number of features after transformation, including episode feature.
    n_states_out_ : int
        Number of states after transformation.
    n_inputs_out_ : int
        Number of inputs after transformation.
    min_samples_ : int
        Minimum number of samples needed to use the transformer.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.

    Warnings
    --------
    The inverse of this preprocessor is not the true inverse unless the data is
    inside ``[-pi, pi]``. Any offsets of ``2pi`` are lost when ``cos`` and
    ``sin`` are applied to the angles.

    Examples
    --------
    Preprocess first feature in pendulum data.

    >>> angle_pp = pykoop.AnglePreprocessor(angle_features=np.array([0]))
    >>> angle_pp.fit(X_pend, n_inputs=1, episode_feature=True)
    AnglePreprocessor(angle_features=array([0]))
    >>> X_pend_pp = angle_pp.transform(X_pend)
    """

    def __init__(self,
                 angle_features: np.ndarray = None,
                 unwrap_inverse: bool = False) -> None:
        """Instantiate :class:`AnglePreprocessor`.

        Parameters
        ----------
        angle_features : np.ndarray
            Indices of features that are angles.
        unwrap_inverse : bool
            Unwrap inverse by replacing absolute jumps greater than ``pi`` by
            their ``2pi`` complement.
        """
        self.angle_features = angle_features
        self.unwrap_inverse = unwrap_inverse

    def _fit_one_ep(self, X: np.ndarray) -> Tuple[int, int]:
        # Compute boolean array with one entry per feature. A ``True`` value
        # indicates that the feature is an angle.
        n_states_inputs_in = self.n_states_in_ + self.n_inputs_in_
        if ((self.angle_features is None)
                or (self.angle_features.shape[0] == 0)):
            self.angles_in_ = np.zeros((n_states_inputs_in, ), dtype=bool)
        else:
            # Create an array with all ```False``.
            angles_bool = np.zeros((n_states_inputs_in, ), dtype=bool)
            # Set indicated entries to ``True``.
            angles_bool[self.angle_features] = True
            self.angles_in_ = angles_bool
        # Figure out how many linear and angular features there are.
        n_lin_states = np.sum(~self.angles_in_[:self.n_states_in_])
        n_lin_inputs = np.sum(~self.angles_in_[self.n_states_in_:])
        n_ang_states = 2 * np.sum(self.angles_in_[:self.n_states_in_])
        n_ang_inputs = 2 * np.sum(self.angles_in_[self.n_states_in_:])
        n_states_out = n_lin_states + n_ang_states
        n_inputs_out = n_lin_inputs + n_ang_inputs
        # Create array for linear, cosine, and sine feature indices.
        self.lin_out_ = np.zeros((n_states_out + n_inputs_out, ), dtype=bool)
        self.cos_out_ = np.zeros((n_states_out + n_inputs_out, ), dtype=bool)
        self.sin_out_ = np.zeros((n_states_out + n_inputs_out, ), dtype=bool)
        # Figure out which features are cosines, sines, or linear.
        i = 0
        for j in range(n_states_inputs_in):
            if self.angles_in_[j]:
                self.cos_out_[i] = True
                self.sin_out_[i + 1] = True
                i += 2
            else:
                self.lin_out_[i] = True
                i += 1
        return (n_states_out, n_inputs_out)

    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Create blank array
        n_states_inputs_out = self.n_states_out_ + self.n_inputs_out_
        Xt = np.zeros((X.shape[0], n_states_inputs_out))
        # Apply cos and sin to appropriate features.
        Xt[:, self.lin_out_] = X[:, ~self.angles_in_]
        Xt[:, self.cos_out_] = np.cos(X[:, self.angles_in_])
        Xt[:, self.sin_out_] = np.sin(X[:, self.angles_in_])
        return Xt

    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Create blank array
        n_states_inputs_in = self.n_states_in_ + self.n_inputs_in_
        Xt = np.zeros((X.shape[0], n_states_inputs_in))
        # Put linear features back where they belong
        Xt[:, ~self.angles_in_] = X[:, self.lin_out_]
        # Invert transform for angles. Unwrap if necessary.
        angle_values = np.arctan2(X[:, self.sin_out_], X[:, self.cos_out_])
        if self.unwrap_inverse:
            Xt[:, self.angles_in_] = np.unwrap(angle_values, axis=0)
        else:
            Xt[:, self.angles_in_] = angle_values
        return Xt

    def _validate_parameters(self) -> None:
        pass  # No constructor parameters need validation.

    def _transform_feature_names(
        self,
        feature_names: np.ndarray,
        format: str = None,
    ) -> np.ndarray:
        # noqa: D102
        if format == 'latex':
            cos = r'\cos'
            sin = r'\sin'
            pre = '{'
            post = '}'
        else:
            cos = 'cos'
            sin = 'sin'
            pre = ''
            post = ''
        names_out = []
        if self.episode_feature_:
            names_out.append(feature_names[0])
        for k in range(self.n_states_in_ + self.n_inputs_in_):
            name_idx = k + 1 if self.episode_feature_ else k
            if self.angles_in_[k]:
                names_out.append(
                    f'{cos}{pre}({feature_names[name_idx]}){post}')
                names_out.append(
                    f'{sin}{pre}({feature_names[name_idx]}){post}')
            else:
                names_out.append(feature_names[name_idx])
        feature_names_out = np.array(names_out, dtype=object)
        return feature_names_out


def random_state(low, high, rng=None):
    """Generate a random initial state.

    Generates uniform random data between specified bounds.

    Very simply wrapper. Really only exists to keep a common interface with
    `random_input`, which is much more complex.

    Parameters
    ----------
    low : float or (n, 1) np.ndarray
        Lower bound for uniform random distribution.
    high : float or (n, 1) np.ndarray
        Upper bound for uniform random distribution.
    rng : Generator
        Random number generator, `numpy.random.default_rng(seed)`.

    Returns
    -------
    np.ndarray:
        Random initial state.

    Examples
    --------
    Simulate a mass-spring-damper with random initial condition

    >>> x_max = np.array([1, 1])
    >>> x0 = pykoop.random_state(-x_max, x_max)
    >>> t, x = msd.simulate((0, 1), 1e-3, x0, lambda t: 0)
    """
    if rng is None:
        rng = np.random.default_rng()
    x_rand = rng.uniform(low, high, low.shape)
    return x_rand


def random_input(t_range,
                 t_step,
                 low,
                 high,
                 cutoff,
                 order=2,
                 rng=None,
                 output='function'):
    """Generate a smooth random input.

    Generates uniform random data between specified bounds, lowpass filters the
    data, then optionally linearly interpolates to return a function of time.

    Uses a Butterworth filter of specified order.

    Parameters
    ----------
    t_range : (2,) tuple
        Start and end times in a tuple (s).
    t_step : float
        Time step at which to generate random data (s).
    low : float or (n, 1) np.ndarray
        Lower bound for uniform random distribution.
    high : float or (n, 1) np.ndarray
        Upper bound for uniform random distribution.
    cutoff : float
        Cutoff frequency for Butterworth lowpass filter (Hz).
    order : int
        Order of Butterworth lowpass filter.
    rng : Generator
        Random number generator, `numpy.random.default_rng(seed)`.
    output : str
        Output format to use. Value 'array' causes the function to return an
        array of smoothed data. Value 'function' causes the function to return
        a function generated by linearly interpolating that same array.

    Returns
    -------
    function or np.ndarray :
        If `output` is 'function', returns a function representing
        linearly-interpolated lowpass-filtered uniformly-random data. If
        `output` is 'array', returns an array containing lowpass-filtered
        uniformly-random data. Units are same as `low` and `high`.

    Examples
    --------
    Simulate a mass-spring-damper with random input

    >>> t_range = (0, 1)
    >>> t_step = 1e-3
    >>> x0 = np.array([0, 0])
    >>> u_max = np.array([1])
    >>> u = pykoop.random_input(t_range, t_step, -u_max, u_max, cutoff=0.01)
    >>> t, x = msd.simulate(t_range, t_step, x0, u)
    """
    t = np.arange(*t_range, t_step)
    size = np.shape(low) + (t.shape[-1], )  # Concatenate tuples
    if rng is None:
        rng = np.random.default_rng()
    u_rough = rng.uniform(np.reshape(low, size[:-1] + (1, )),
                          np.reshape(high, size[:-1] + (1, )), size)
    sos = signal.butter(order, cutoff, output='sos', fs=1 / t_step)
    u_smooth = signal.sosfilt(sos, u_rough)
    if output == 'array':
        return u_smooth
    elif output == 'function':
        f_smooth = interpolate.interp1d(t, u_smooth, fill_value='extrapolate')
        return f_smooth
    else:
        raise ValueError(f'{output} is not a valid output form.')


def example_data_msd() -> Dict[str, Any]:
    """Get example mass-spring-damper data.

    Returns
    -------
    Dict[str, Any]
        Sample mass-spring damper data.
    """
    # Create mass-spring-damper object
    msd = dynamic_models.MassSpringDamper(
        mass=0.5,
        stiffness=0.7,
        damping=0.6,
    )
    # Set timestep
    t_range = (0, 10)
    t_step = 0.1
    t_sim = np.arange(*t_range, t_step)
    # Initial conditions and inputs
    conditions = [
        (0, np.array([0, 0]), lambda t: 0.1 * np.sin(t)),
        (1, np.array([0, 0]), lambda t: 0.2 * np.cos(t)),
        (2, np.array([0, 0]), lambda t: -0.2 * np.sin(t)),
        (3, np.array([0, 0]), lambda t: -0.1 * np.cos(t)),
    ]
    X_msd_lst = []
    # Loop over initial conditions and inputs
    for ep, x0, u in conditions:
        # Simulate ODE
        t, x = msd.simulate(
            t_range=t_range,
            t_step=t_step,
            x0=x0,
            u=u,
            rtol=1e-8,
            atol=1e-8,
        )
        # Format the data
        X_msd_lst.append(
            np.hstack((
                ep * np.ones((t.shape[0], 1)),
                x,
                np.reshape(u(t), (-1, 1)),
            )))
    # Stack data and return
    X_msd = np.vstack(X_msd_lst)
    X_train = X_msd[X_msd[:, 0] < 3, :]
    X_valid = X_msd[X_msd[:, 0] == 3, :]
    n_inputs = 1
    episode_feature = True
    x0_valid = koopman_pipeline.extract_initial_conditions(
        X_valid,
        n_inputs=n_inputs,
        episode_feature=episode_feature,
    )
    u_valid = koopman_pipeline.extract_input(
        X_valid,
        n_inputs=n_inputs,
        episode_feature=episode_feature,
    )
    return {
        'X_train': X_train,
        'X_valid': X_valid,
        'x0_valid': x0_valid,
        'u_valid': u_valid,
        'n_inputs': n_inputs,
        'episode_feature': episode_feature,
        't': t_sim,
        'dynamic_model': msd,
    }


def example_data_vdp() -> Dict[str, Any]:
    """Get example Van der Pol oscillator data.

    Returns
    -------
    Dict[str, Any]
        Sample Van der Pol oscillator data.
    """
    # Create Van der Pol object
    t_step = 0.01
    vdp = dynamic_models.DiscreteVanDerPol(t_step, mu=2)
    # Initial conditions and inputs
    t_range = (0, 10)
    t_sim = np.arange(*t_range, t_step)
    conditions = [
        (0, np.array([1, 0]), 0.1 * np.sin(t_sim)),
        (1, np.array([0, 1]), 0.2 * np.cos(t_sim)),
        (2, np.array([-1, 0]), -0.2 * np.sin(t_sim)),
        (3, np.array([0, -1]), -0.1 * np.cos(t_sim)),
        (4, np.array([0.5, 0]), 0.1 * np.cos(t_sim)),
    ]
    X_vdp_lst = []
    # Loop over initial conditions and inputs
    for ep, x0, u in conditions:
        # Simulate ODE
        t, x = vdp.simulate(
            t_range=t_range,
            t_step=t_step,
            x0=x0,
            u=u,
        )
        # Format the data
        X_vdp_lst.append(
            np.hstack((
                ep * np.ones((t.shape[0], 1)),
                x,
                np.reshape(u, (-1, 1)),
            )))
    # Stack data and return
    X_vdp = np.vstack(X_vdp_lst)
    X_train = X_vdp[X_vdp[:, 0] < 4, :]
    X_valid = X_vdp[X_vdp[:, 0] == 4, :]
    n_inputs = 1
    episode_feature = True
    x0_valid = koopman_pipeline.extract_initial_conditions(
        X_valid,
        n_inputs=n_inputs,
        episode_feature=episode_feature,
    )
    u_valid = koopman_pipeline.extract_input(
        X_valid,
        n_inputs=n_inputs,
        episode_feature=episode_feature,
    )
    return {
        'X_train': X_train,
        'X_valid': X_valid,
        'x0_valid': x0_valid,
        'u_valid': u_valid,
        'n_inputs': n_inputs,
        'episode_feature': episode_feature,
        't': t_sim,
        'dynamic_model': vdp,
    }


def example_data_pendulum() -> Dict[str, Any]:
    """Get example pendulum data.

    Always starts with zero initial velocity.

    Returns
    -------
    Dict[str, Any]
        Sample pendulum data.
    """
    # Create pendulum object
    t_step = 0.01
    pend = dynamic_models.Pendulum(mass=1, length=1, damping=1e-1)
    # Initial conditions
    t_range = (0, 20)
    t_sim = np.arange(*t_range, t_step)
    n_ep = 50
    conditions = [(i, np.array([np.pi * i * 0.1, 0])) for i in range(n_ep)]
    X_pend_lst = []
    # Loop over initial conditions and inputs
    for ep, x0 in conditions:
        # Simulate ODE
        t, x = pend.simulate(
            t_range=t_range,
            t_step=t_step,
            x0=x0,
            u=lambda t: 0,
        )
        # Format the data
        X_pend_lst.append(np.hstack((
            ep * np.ones((t.shape[0], 1)),
            x,
        )))
    # Stack data and return
    X_pend = np.vstack(X_pend_lst)
    valid_ep = [5, 25, 45]
    train_ep = list(set(range(n_ep)) - set(valid_ep))
    valid_idx = np.where(np.in1d(X_pend[:, 0], valid_ep))[0]
    train_idx = np.where(np.in1d(X_pend[:, 0], train_ep))[0]
    X_train = X_pend[train_idx, :]
    X_valid = X_pend[valid_idx, :]
    n_inputs = 0
    episode_feature = True
    x0_valid = koopman_pipeline.extract_initial_conditions(
        X_valid,
        n_inputs=n_inputs,
        episode_feature=episode_feature,
    )
    u_valid = koopman_pipeline.extract_input(
        X_valid,
        n_inputs=n_inputs,
        episode_feature=episode_feature,
    )
    return {
        'X_train': X_train,
        'X_valid': X_valid,
        'x0_valid': x0_valid,
        'u_valid': u_valid,
        'n_inputs': n_inputs,
        'episode_feature': episode_feature,
        't': t_sim,
        'dynamic_model': pend,
    }


def example_data_duffing() -> Dict[str, Any]:
    """Get example Duffing oscillator data.

    Parameters from [WHDKR16]_.

    Returns
    -------
    Dict[str, Any]
        Sample Duffing oscillator data.
    """
    # Create Duffing oscillator object
    t_step = 0.1
    do = dynamic_models.DuffingOscillator(alpha=1, beta=-1, delta=0.2)
    # Initial conditions
    t_range = (0, 20)
    t_sim = np.arange(*t_range, t_step)
    n_ep = 50
    X_do_lst = []
    # Loop over initial conditions and inputs
    rng = np.random.default_rng(1234)
    for ep in range(n_ep):
        # Simulate ODE
        x0 = random_state(
            np.array([-1, -1]),
            np.array([1, 1]),
            rng=rng,
        )
        a = rng.uniform(0, 0.3)

        def u(t):
            """Input."""
            return a * np.cos(t)

        t, x = do.simulate(
            t_range=t_range,
            t_step=t_step,
            x0=x0,
            u=u,
        )
        # Format the data
        X_do_lst.append(
            np.hstack((
                ep * np.ones((t.shape[0], 1)),
                x,
                u(t).reshape(-1, 1),
            )))
    # Stack data and return
    X_do = np.vstack(X_do_lst)
    valid_ep = [n_ep - 3, n_ep - 2, n_ep - 1]
    train_ep = list(set(range(n_ep)) - set(valid_ep))
    valid_idx = np.where(np.in1d(X_do[:, 0], valid_ep))[0]
    train_idx = np.where(np.in1d(X_do[:, 0], train_ep))[0]
    X_train = X_do[train_idx, :]
    X_valid = X_do[valid_idx, :]
    n_inputs = 1
    episode_feature = True
    x0_valid = koopman_pipeline.extract_initial_conditions(
        X_valid,
        n_inputs=n_inputs,
        episode_feature=episode_feature,
    )
    u_valid = koopman_pipeline.extract_input(
        X_valid,
        n_inputs=n_inputs,
        episode_feature=episode_feature,
    )
    return {
        'X_train': X_train,
        'X_valid': X_valid,
        'x0_valid': x0_valid,
        'u_valid': u_valid,
        'n_inputs': n_inputs,
        'episode_feature': episode_feature,
        't': t_sim,
        'dynamic_model': do,
    }
