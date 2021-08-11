"""Collection of Koopman lifting functions.

All of the lifting functions included in this module adhere to the interface
defined in :class:`KoopmanLiftingFn`.
"""

from typing import Tuple

import numpy as np
import sklearn.base
import sklearn.preprocessing
import sklearn.utils.validation

from . import koopman_pipeline


class SkLearnLiftingFn(koopman_pipeline.EpisodeIndependentLiftingFn):
    """Lifting function that wraps a ``scikit-learn`` transformer.

    Examples of appropriate transformers are

    - ``sklearn.preprocessing.StandardScaler``,
    - ``sklearn.preprocessing.MinMaxScaler``, or
    - ``sklearn.preprocessing.MaxAbsScaler``.

    See [transformers]_ for more appropriate transformers (though not all have
    been tested).

    Attributes
    ----------
    transformer_ : sklearn.base.BaseEstimator
        Fit transformer.
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

    References
    ----------
    .. [transformers] https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing  # noqa: E501

    Examples
    --------
    Preprocess mass-spring-damper data to have zero mean and unit variance

    >>> std_scaler = pykoop.SkLearnLiftingFn(
    ...     sklearn.preprocessing.StandardScaler())
    >>> std_scaler.fit(X_msd, n_inputs=1, episode_feature=True)
    SkLearnLiftingFn(transformer=StandardScaler())
    >>> X_msd_pp = std_scaler.transform(X_msd)
    >>> np.mean(X_msd_pp[:, 1:], axis=0)
    array([ 1.99840144e-17, -3.10862447e-17, -1.55431223e-17])
    >>> np.std(X_msd_pp[:, 1:], axis=0)
    array([1., 1., 1.])
    """

    def __init__(
        self,
        transformer: sklearn.base.BaseEstimator = None,
    ) -> None:
        """Instantiate :class:`SkLearnLiftingFn`.

        Parameters
        ----------
        transformer : sklearn.base.BaseEstimator
            Transformer to wrap.
        """
        self.transformer = transformer

    def _fit_one_ep(self, X: np.ndarray) -> Tuple[int, int]:
        self.transformer_ = sklearn.base.clone(self.transformer)
        self.transformer_.fit(X)
        return (self.n_states_in_, self.n_inputs_in_)

    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        Xt = self.transformer_.transform(X)
        return Xt

    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        Xt = self.transformer_.inverse_transform(X)
        return Xt

    def _validate_parameters(self) -> None:
        # Let transformer do its own validation
        pass


class PolynomialLiftingFn(koopman_pipeline.EpisodeIndependentLiftingFn):
    """Lifting function to generate all monomials of the input features.

    Attributes
    ----------
    transformer_ : sklearn.preprocessing.PolynomialFeatures
        Internal transformer generating the polynomial features.
    transform_order_ : np.ndarray
        Indices indicating reordering needed after applying ``transformer_``.
    inverse_transform_order_ : np.ndarray
        Indices indicating reverse ordering needed to reverse
        ``transform_order_``.
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

    Examples
    --------
    Apply polynomial features to mass-spring-damper data

    >>> poly = pykoop.PolynomialLiftingFn(order=2)
    >>> poly.fit(X_msd, n_inputs=1, episode_feature=True)
    PolynomialLiftingFn(order=2)
    >>> Xt_msd = poly.transform(X_msd[:2, :])
    """

    def __init__(self, order: int = 1, interaction_only: bool = False) -> None:
        """Instantiate :class:`PolynomialLiftingFn`.

        Parameters
        ----------
        order : int
            Order of monomials to generate.
        interaction_only : bool
            If ``True``, skip powers of the same feature.
        """
        self.order = order
        self.interaction_only = interaction_only

    def _fit_one_ep(self, X: np.ndarray) -> Tuple[int, int]:
        self.transformer_ = sklearn.preprocessing.PolynomialFeatures(
            degree=self.order,
            interaction_only=self.interaction_only,
            include_bias=False)
        self.transformer_.fit(X)
        # Figure out which lifted states correspond to the original states and
        # original inputs
        orig_states = []
        orig_inputs = []
        eye = np.eye(self.n_states_in_ + self.n_inputs_in_)
        for i in range(self.n_states_in_ + self.n_inputs_in_):
            # Original states and inputs will have a ``power_`` that's a
            # row of the identity matrix, since they are mononomials of the
            # form ``x_1^1 * x_2^0 * x_3^0``. In that case, the row would be
            # ``[1, 0, 0]``.
            index = np.nonzero(
                np.all(self.transformer_.powers_ == eye[i, :], axis=1))
            # Split up the "original states" from the "original inputs"
            if i < self.n_states_in_:
                orig_states.append(index)
            else:
                orig_inputs.append(index)
        original_states = np.ravel(orig_states).astype(int)
        original_input = np.ravel(orig_inputs).astype(int)
        # Figure out which other lifted states contain inputs. These ones will
        # go at the bottom of the output.
        # Find all input-dependent features:
        all_inputs = np.nonzero(
            np.any(self.transformer_.powers_[:, self.n_states_in_:] != 0,
                   axis=1))[0].astype(int)
        # Do a set difference to remove the unlifted input features:
        other_inputs = np.setdiff1d(all_inputs, original_input)
        # Figure out which other lifted states contain states (but are not the
        # original states themselves. Accomplish this by subtracting off
        # all the other types of features we found. That is, if it's not in
        # ``original_state_features``, ``original_input_features``, or
        # ``other_input_features``, it must be in ``other_state_features``.
        other_states = np.setdiff1d(
            np.arange(self.transformer_.powers_.shape[0]),
            np.union1d(np.union1d(original_states, original_input),
                       other_inputs),
        ).astype(int)
        # Form new ordering of states. Input-dependent states go at the end.
        self.transform_order_ = np.concatenate(
            (original_states, other_states, original_input, other_inputs))
        # Compute how many input-independent lifted states and input-dependent
        # lifted states there are
        n_states_out = (original_states.shape[0] + other_states.shape[0])
        n_inputs_out = (original_input.shape[0] + other_inputs.shape[0])
        # Figure out original order of features
        self.inverse_transform_order_ = np.concatenate((
            np.arange(0, self.n_states_in_),
            np.arange(n_states_out, (n_states_out + self.n_inputs_in_)),
        ))
        return (n_states_out, n_inputs_out)

    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Transform states
        Xt = self.transformer_.transform(X)
        # Reorder states
        return Xt[:, self.transform_order_]

    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Extract the original features from the lifted features
        return X[:, self.inverse_transform_order_]

    def _validate_parameters(self) -> None:
        if self.order <= 0:
            raise ValueError('`order` must be greater than or equal to 1.')


class BilinearInputLiftingFn(koopman_pipeline.EpisodeIndependentLiftingFn):
    """Lifting function to generate bilinear products of the state and input.

    As proposed in [bilinear]_. Given a state ``x`` and input::

        u = np.array([
            [u1],
            [u2],
            [u3],
        ])

    the bilinear lifted state has the form::

        psi = np.array([
            [x],
            [x * u1],
            [x * u2],
            [x * u3],
            [u].
        ])

    where the products are element-wise.

    Attributes
    ----------
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

    Examples
    --------
    Apply bilinear input features to mass-spring-damper data

    >>> bilin = pykoop.BilinearInputLiftingFn()
    >>> bilin.fit(X_msd, n_inputs=1, episode_feature=True)
    BilinearInputLiftingFn()
    >>> Xt_msd = bilin.transform(X_msd[:2, :])
    """

    def __init__(self) -> None:
        """Instantiate :class:`BilinearInputLiftingFn`."""
        # Nothing to do but write the docstring.
        pass

    def _fit_one_ep(self, X: np.ndarray) -> Tuple[int, int]:
        n_states_out = self.n_states_in_
        n_inputs_out = (self.n_states_in_ + 1) * self.n_inputs_in_
        return (n_states_out, n_inputs_out)

    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        states = X[:, :self.n_states_in_]
        inputs = X[:, self.n_states_in_:]
        features = [states]
        for k in range(self.n_inputs_in_):
            features.append(states * inputs[:, [k]])
        features.append(inputs)
        Xt = np.hstack(features)
        return Xt

    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        Xt = np.hstack((
            X[:, :self.n_states_in_],
            X[:, self.n_states_out_ + self.n_inputs_out_ - self.n_inputs_in_:],
        ))
        return Xt

    def _validate_parameters(self) -> None:
        # No parameters to validate
        pass


class DelayLiftingFn(koopman_pipeline.EpisodeDependentLiftingFn):
    """Lifting function to generate delay coordinates for state and input.

    Attributes
    ----------
    n_features_in_ : int
        Number of features before transformation, including episode feature if
        present.
    n_states_in_ : int
        Number of states before transformation.
    n_inputs_in_ : int
        Number of inputs before transformation.
    n_features_out_ : int
        Number of features after transformation, including episode feature if
        present.
    n_states_out_ : int
        Number of states after transformation.
    n_inputs_out_ : int
        Number of inputs after transformation.
    min_samples_ : int
        Minimum number of samples needed to use the transformer.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.

    Warnings
    --------
    :func:`transform` and :func:`inverse_transform` are not exact inverses
    unless ``n_delays_x`` and ``n_delays_u`` are the same. Only the last
    samples will be the same, since the ``abs(n_delays_x - n_delays_u)``
    earliest samples will need to be dropped to ensure the output array is
    rectangular.

    Examples
    --------
    Apply delay lifting function to mass-spring-damper data

    >>> delay = pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=1)
    >>> delay.fit(X_msd, n_inputs=1, episode_feature=True)
    DelayLiftingFn(n_delays_input=1, n_delays_state=1)
    >>> Xt_msd = delay.transform(X_msd[:3, :])
    """

    def __init__(self,
                 n_delays_state: int = 0,
                 n_delays_input: int = 0) -> None:
        """Instantiate :class:`DelayLiftingFn`.

        Parameters
        ----------
        n_delays_state : int
            Number of delays to apply to the state.
        n_delays_input : int
            Number of delays to apply to the input.
        """
        self.n_delays_state = n_delays_state
        self.n_delays_input = n_delays_input

    def n_samples_in(self, n_samples_out: int = 1) -> int:
        # noqa: D102
        return n_samples_out + max(self.n_delays_state, self.n_delays_input)

    def _fit_one_ep(self, X: np.ndarray) -> Tuple[int, int, int]:
        # Compute number of states and inputs that will be output.
        n_states_out = self.n_states_in_ * (self.n_delays_state + 1)
        n_inputs_out = self.n_inputs_in_ * (self.n_delays_input + 1)
        # Compute the minimum number of samples needed to use this transformer.
        min_samples = max(self.n_delays_state, self.n_delays_input) + 1
        return (n_states_out, n_inputs_out, min_samples)

    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Separate states and inputs
        X_x = X[:, :self.n_states_in_]
        X_u = X[:, self.n_states_in_:]
        # Delay states and inputs separately
        Xd_x = DelayLiftingFn._delay(X_x, self.n_delays_state)
        Xd_u = DelayLiftingFn._delay(X_u, self.n_delays_input)
        # ``Xd_x`` and ``Xd_u`` have different numbers of samples at this
        # point. Truncate the larger one to the same shape.
        n_samples = min(Xd_x.shape[0], Xd_u.shape[0])
        Xd = np.hstack((Xd_x[-n_samples:, :], Xd_u[-n_samples:, :]))
        return Xd

    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Separate states and inputs.
        X_x = X[:, :self.n_states_out_]
        X_u = X[:, self.n_states_out_:]
        # Undelay states and inputs separately.
        Xu_x = DelayLiftingFn._undelay(X_x, self.n_delays_state)
        Xu_u = DelayLiftingFn._undelay(X_u, self.n_delays_input)
        # ``Xu_x`` and ``Xu_u`` have different numbers of samples at this
        # point. Truncate the larger one to the same shape.
        n_samples = min(Xu_x.shape[0], Xu_u.shape[0])
        Xu = np.hstack((Xu_x[-n_samples:, :], Xu_u[-n_samples:, :]))
        return Xu

    def _validate_parameters(self):
        if self.n_delays_state < 0:
            raise ValueError(
                '`n_delays_x` must be greater than or equal to 0.')
        if self.n_delays_input < 0:
            raise ValueError(
                '`n_delays_u` must be greater than or equal to 0.')

    @staticmethod
    def _delay(X: np.ndarray, n_delays: int) -> np.ndarray:
        """Delay an array by a specific number of samples.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        n_delays : int
            Number of delays.

        Returns
        -------
        np.ndarray
            Delayed matrix.
        """
        n_samples_out = X.shape[0] - n_delays
        # Go through matrix with moving window and treat each window as a new
        # set of features.
        delays = []
        for i in range(n_delays, -1, -1):
            delays.append(X[i:(n_samples_out + i), :])
        Xd = np.concatenate(delays, axis=1)
        return Xd

    @staticmethod
    def _undelay(X: np.ndarray, n_delays: int) -> np.ndarray:
        """Undelay an array by a specific number of samples.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        n_delays : int
            Number of delays.

        Returns
        -------
        np.ndarray
            Undelayed matrix.

        Notes
        -----
        To understand how this implementation works, consider a matrix::

            [1 -1]
            [2 -2]
            [3 -3]
            [4 -4]

        and its delayed-by-one version::

            [2 -2 1 -1]
            [3 -3 2 -2]
            [4 -4 3 -3]

        The last two features contain the first three timesteps of the original
        matrix::

            [1 -1]
            [2 -2]
            [3 -3]

        The last timestep contains the last two timesteps of the original
        matrix::

            [4 -4 3 -3]

        By merging these two blocks of the matrix (omitting the last timestep
        of the last two features to avoid duplication), we can reconstruct the
        original matrix.
        """
        # Compute number of features
        n_features = X.shape[1] // (n_delays + 1)
        # Get the first part of the sequence from the last features (omitting
        # the last sample to avoid duplication).
        Xu_1 = [X[:-1, -n_features:]]
        # Get the rest of the sequence from the last sample
        Xu_2 = np.split(X[[-1], :], n_delays + 1, axis=1)[::-1]
        # Put the sequence together
        Xu = np.vstack(Xu_1 + Xu_2)
        return Xu
