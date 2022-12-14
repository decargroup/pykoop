"""Collection of Koopman lifting functions.

All of the lifting functions included in this module adhere to the interface
defined in :class:`KoopmanLiftingFn`.
"""

from typing import Callable, Tuple, Union

import numpy as np
import sklearn.base
import sklearn.kernel_approximation
import sklearn.preprocessing
import sklearn.utils.validation
from scipy import linalg

from . import centers, kernel_approximation, koopman_pipeline


class SkLearnLiftingFn(koopman_pipeline.EpisodeIndependentLiftingFn):
    """Lifting function that wraps a ``scikit-learn`` transformer.

    Examples of appropriate transformers are

    - ``sklearn.preprocessing.StandardScaler``,
    - ``sklearn.preprocessing.MinMaxScaler``, or
    - ``sklearn.preprocessing.MaxAbsScaler``.

    Any ``scikit-learn`` transformer [#tr]_ should be compatible, though not
    all have been tested.

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
    feature_names_in_ : np.ndarray
        Array of input feature name strings.

    References
    ----------
    .. [#tr] https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

    Examples
    --------
    Preprocess mass-spring-damper data to have zero mean and unit variance

    >>> std_scaler = pykoop.SkLearnLiftingFn(
    ...     sklearn.preprocessing.StandardScaler())
    >>> std_scaler.fit(X_msd, n_inputs=1, episode_feature=True)
    SkLearnLiftingFn(transformer=StandardScaler())
    >>> std_scaler.get_feature_names_in().tolist()
    ['ep', 'x0', 'x1', 'u0']
    >>> std_scaler.get_feature_names_out().tolist()
    ['ep', 'StandardScaler(x0)', 'StandardScaler(x1)', 'StandardScaler(u0)']
    >>> X_msd_pp = std_scaler.transform(X_msd)
    >>> np.mean(X_msd_pp[:, 1:], axis=0)
    array([...])
    >>> np.std(X_msd_pp[:, 1:], axis=0)
    array([...])
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

    def _transform_feature_names(
        self,
        feature_names: np.ndarray,
        format: str = None,
    ) -> np.ndarray:
        # noqa: D102
        if format == 'latex':
            fn = rf'\mathrm{{{self.transformer_.__class__.__name__}}}'
        else:
            fn = self.transformer_.__class__.__name__
        names_out = []
        for k in range(self.n_features_in_):
            if self.episode_feature_ and (k == 0):
                names_out.append(feature_names[k])
            else:
                names_out.append(f'{fn}({feature_names[k]})')
        feature_names_out = np.array(names_out, dtype=object)
        return feature_names_out


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
    feature_names_in_ : np.ndarray
        Array of input feature name strings.

    Examples
    --------
    Apply polynomial features to mass-spring-damper data

    >>> poly = pykoop.PolynomialLiftingFn(order=2)
    >>> poly.fit(X_msd, n_inputs=1, episode_feature=True)
    PolynomialLiftingFn(order=2)
    >>> poly.get_feature_names_in().tolist()
    ['ep', 'x0', 'x1', 'u0']
    >>> poly.get_feature_names_out().tolist()
    ['ep', 'x0', 'x1', 'x0^2', 'x0*x1', 'x1^2', 'u0', 'x0*u0', 'x1*u0', 'u0^2']
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
            include_bias=False,
        )
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
                np.all(
                    self.transformer_.powers_ == eye[i, :],
                    axis=1,
                ))
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
            np.any(
                self.transformer_.powers_[:, self.n_states_in_:] != 0,
                axis=1,
            ))[0].astype(int)
        # Do a set difference to remove the unlifted input features:
        other_inputs = np.setdiff1d(all_inputs, original_input)
        # Figure out which other lifted states contain states, but are not the
        # original states themselves. Accomplish this by subtracting off
        # all the other types of features we found. That is, if it's not in
        # ``original_state_features``, ``original_input_features``, or
        # ``other_input_features``, it must be in ``other_state_features``.
        other_states = np.setdiff1d(
            np.arange(self.transformer_.powers_.shape[0]),
            np.union1d(
                np.union1d(original_states, original_input),
                other_inputs,
            ),
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

    def _transform_feature_names(
        self,
        feature_names: np.ndarray,
        format: str = None,
    ) -> np.ndarray:
        # noqa: D102
        if format == 'latex':
            times = ' '
            pre = '{'
            post = '}'
        else:
            times = '*'
            pre = ''
            post = ''
        # Deal with episode feature
        if self.episode_feature_:
            names_in = feature_names[1:]
            ep = feature_names[[0]]
        else:
            names_in = feature_names
            ep = None
        # Transform feature names
        names_tf = []
        powers = self.transformer_.powers_
        for ft_out in range(powers.shape[0]):
            str_ = []
            for ft_in in range(powers.shape[1]):
                if powers[ft_out, ft_in] == 0:
                    pass
                elif powers[ft_out, ft_in] == 1:
                    str_.append(f'{names_in[ft_in]}')
                else:
                    exp = f'{pre}{powers[ft_out, ft_in]}{post}'
                    str_.append(f'{names_in[ft_in]}^{exp}')
            names_tf.append(times.join(str_))
        names_tf_arr = np.asarray(names_tf, dtype=object)
        names_out = names_tf_arr[self.transform_order_]
        if ep is not None:
            feature_names_out = np.concatenate((ep, names_out), dtype=object)
        else:
            feature_names_out = names_out.astype(object)
        return feature_names_out


class BilinearInputLiftingFn(koopman_pipeline.EpisodeIndependentLiftingFn):
    """Lifting function to generate bilinear products of the state and input.

    As proposed in [BFV20]_. Given a state ``x`` and input::

        u = np.array([
            [u1],
            [u2],
            [u3],
        ])

    the bilinear lifted state has the form::

        psi = np.array([
            [x],
            [u],
            [x * u1],
            [x * u2],
            [x * u3],
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
    feature_names_in_ : np.ndarray
        Array of input feature name strings.

    Examples
    --------
    Apply bilinear input features to mass-spring-damper data

    >>> bilin = pykoop.BilinearInputLiftingFn()
    >>> bilin.fit(X_msd, n_inputs=1, episode_feature=True)
    BilinearInputLiftingFn()
    >>> bilin.get_feature_names_in().tolist()
    ['ep', 'x0', 'x1', 'u0']
    >>> bilin.get_feature_names_out().tolist()
    ['ep', 'x0', 'x1', 'u0', 'x0*u0', 'x1*u0']
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
        features = [states, inputs]
        for k in range(self.n_inputs_in_):
            features.append(states * inputs[:, [k]])
        Xt = np.hstack(features)
        return Xt

    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        Xt = np.hstack((
            X[:, :self.n_states_in_],
            X[:, self.n_states_in_:self.n_states_in_ + self.n_inputs_in_],
        ))
        return Xt

    def _validate_parameters(self) -> None:
        # No parameters to validate
        pass

    def _transform_feature_names(
        self,
        feature_names: np.ndarray,
        format: str = None,
    ) -> np.ndarray:
        # noqa: D102
        if format == 'latex':
            times = ' '
        else:
            times = '*'
        names_out = []
        # Deal with episode feature
        if self.episode_feature_:
            names_in = feature_names[1:]
            names_out.append(feature_names[0])
        else:
            names_in = feature_names
        # Add states and inputs
        for ft in range(self.n_states_in_ + self.n_inputs_in_):
            names_out.append(names_in[ft])
        # Add products
        for ft_u in range(self.n_states_in_,
                          self.n_states_in_ + self.n_inputs_in_):
            for ft_x in range(self.n_states_in_):
                names_out.append(f'{names_in[ft_x]}{times}{names_in[ft_u]}')
        feature_names_out = np.array(names_out, dtype=object)
        return feature_names_out


class RbfLiftingFn(koopman_pipeline.EpisodeIndependentLiftingFn):
    """Lifting function using radial basis function (RBF) features.

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
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    rbf_ : Callable[[np.ndarray], np.ndarray]
        Radial basis function as a callable.
    centers_ : centers.Centers
        Fit centers estimator.
    offset_ : float
        Offset used to calculate radius. Differs from ``offset`` only if
        ``offset=None``.

    Examples
    --------
    Gaussian RBF lifting functions with normally distributed centers

    >>> rbf = pykoop.RbfLiftingFn(
    ...     rbf='gaussian',
    ...     centers=pykoop.GaussianRandomCenters(
    ...         n_centers=10,
    ...     ),
    ...     shape=0.1,
    ... )
    >>> rbf.fit(X_msd, n_inputs=1, episode_feature=True)
    RbfLiftingFn(centers=GaussianRandomCenters(n_centers=10), shape=0.1)

    Thin-plate RBF lifting functions with K-means clustered centers

    >>> tp = pykoop.RbfLiftingFn(
    ...     rbf='thin_plate',
    ...     centers=pykoop.ClusterCenters(
    ...         estimator=sklearn.cluster.KMeans(n_clusters=3)
    ...     ),
    ...     shape=10,
    ... )
    >>> tp.fit(X_msd, n_inputs=1, episode_feature=True)
    RbfLiftingFn(centers=ClusterCenters(estimator=KMeans(n_clusters=3)),
    rbf='thin_plate', shape=10)

    Inverse quadratic RBF lifting functions with Latin hypercube centers

    >>> iq = pykoop.RbfLiftingFn(
    ...     rbf='inverse_quadratic',
    ...     centers=pykoop.QmcCenters(
    ...         n_centers=10,
    ...         qmc=scipy.stats.qmc.LatinHypercube,
    ...     ),
    ... )
    >>> iq.fit(X_msd, n_inputs=1, episode_feature=True)
    RbfLiftingFn(centers=QmcCenters(n_centers=10,
    qmc=<class 'scipy.stats._qmc.LatinHypercube'>), rbf='inverse_quadratic')
    """

    def _rbf_exponential(r):
        """Exponential RBF."""
        return np.exp(-r)

    def _rbf_gaussian(r):
        """Gaussian RBF."""
        return np.exp(-r**2)

    def _rbf_multiquadric(r):
        """Multiquadric RBF."""
        return np.sqrt(1 + r**2)

    def _rbf_inverse_quadratic(r):
        """Inverse quadratic RBF."""
        return 1 / (1 + r**2)

    def _rbf_inverse_multiquadric(r):
        """Inverse multiquadric RBF."""
        return 1 / np.sqrt(1 + r**2)

    def _rbf_thin_plate(r):
        """Thin plate RBF."""
        return r**2 * np.log(r)

    def _rbf_bump_function(r):
        """Bump function RBF."""

        def bump(s):
            return np.exp(-1 / (1 - s**2))

        return np.piecewise(
            r,
            [r < 1],
            [bump, 0],
        )

    _rbf_lookup = {
        'exponential': {
            'callable': _rbf_exponential,
            'offset': 0,
        },
        'gaussian': {
            'callable': _rbf_gaussian,
            'offset': 0,
        },
        'multiquadric': {
            'callable': _rbf_multiquadric,
            'offset': 0,
        },
        'inverse_quadratic': {
            'callable': _rbf_inverse_quadratic,
            'offset': 0
        },
        'inverse_multiquadric': {
            'callable': _rbf_inverse_multiquadric,
            'offset': 0,
        },
        'thin_plate': {
            'callable': _rbf_thin_plate,
            'offset': 1e-3,
        },
        'bump_function': {
            'callable': _rbf_bump_function,
            'offset': 0,
        },
    }
    """Lookup table mapping RBF name to callable and default offset.

    The default offset should only be nonzero if the function is not defined
    for ``r = 0``.

    The private functions here are not `@staticmethod`s because that would
    prevent `scikit-learn` from pickling them, which would make the estimator
    check fail.
    """

    def __init__(
        self,
        rbf: Union[str, Callable[[np.ndarray], np.ndarray]] = 'gaussian',
        centers: centers.Centers = None,
        shape: float = 1,
        offset: float = None,
    ) -> None:
        """Instantiate :class:`RbfLiftingFn`.

        Attributes
        ----------
        rbf : Union[str, Callable[[np.ndarray], np.ndarray]]
            Radial basis function type. These are functions of the radius
            ``r = shape * ||x - c|| + offset``, where ``c`` is a constant
            center. Commonly used ones can be specified by their names,

            - ``'exponential'`` -- ``exp(-r)`` (see [WHDKR16]_),
            - ``'gaussian'`` -- ``exp(-r^2)`` (see [WHDKR16]_),
            - ``'multiquadric'`` -- ``sqrt(1 + r^2)``,
            - ``'inverse_quadratic'`` -- ``1 / (1 + r^2)``,
            - ``'inverse_multiquadric'`` -- ``1 / sqrt(1 + r^2)``,
            - ``'thin_plate'`` -- ``r^2 * ln(r)`` (see [DTK20]_ and [CHH19]_),
              or
            - ``'bump_function'`` -- ``exp(-1 / (1 - r^2)) if r < 1, else 0``.

            Alternatively, a callable representing the basis function ``R(r)``
            can be used. It must be vectorized, i.e., it must be callable
            with an array of radii and operate on it elementwise.

        centers : centers.Centers
            Estimator to generate centers from data. Number of lifting
            functions is controlled by the number of centers generated.
            Defaults to :class:`QmcCenters` with its default arguments.

        shape : float
            Shape parameter. Must be greater than zero. Larger numbers produce
            "sharper" basis functions. Default is ``1``.

        offset : float
            Offset to apply to the norm. Not needed unless RBF is not defined
            for zero radius. Default is ``None``, where zero is used for all
            ``rbf`` values except ``'thin_plate'``, where ``1e-3`` is used.
        """
        self.rbf = rbf
        self.centers = centers
        self.shape = shape
        self.offset = offset

    def _fit_one_ep(self, X: np.ndarray) -> Tuple[int, int]:
        # Set basis function type
        if isinstance(self.rbf, str):
            self.rbf_ = self._rbf_lookup[self.rbf]['callable']
        else:
            self.rbf_ = self.rbf
        # Set and fit centers
        if self.centers is None:
            self.centers_ = centers.QmcCenters()
        else:
            self.centers_ = sklearn.base.clone(self.centers)
        self.centers_.fit(X)
        # Set offset
        if self.offset is None:
            if isinstance(self.rbf, str):
                self.offset_ = self._rbf_lookup[self.rbf]['offset']
            else:
                self.offset_ = 0
        else:
            self.offset_ = self.offset
        # Calculate number of features
        if self.n_inputs_in_ == 0:
            n_states_out = self.n_states_in_ + self.centers_.n_centers_
            n_inputs_out = self.n_inputs_in_
        else:
            n_states_out = self.n_states_in_
            n_inputs_out = self.n_inputs_in_ + self.centers_.n_centers_
        return (n_states_out, n_inputs_out)

    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Separate states and inputs.
        X_x = X[:, :self.n_states_in_]
        X_u = X[:, self.n_states_in_:]
        # Calculate difference b/t states and centers. Leverages broadcasting:
        # ``(n_samples, 1, n_features) - (n_centers, n_features)``.
        # The middle dimension gets broadcast (expanded) to ``n_centers``. The
        # result has shape ``(n_samples, n_centers, n_features)``.
        diff = X[:, np.newaxis, :] - self.centers_.centers_
        # Calculate radii. The norm is taken over the last dimension to get
        # shape ``(n_samples, n_centers)``.
        radii = self.shape * linalg.norm(diff, axis=-1) + self.offset_
        # Evaluate basis function elementwise
        X_r = self.rbf_(radii)
        # Stack results. ``X_r`` always goes at the bottom because it always
        # involves the lifted inputs if present.
        Xt = np.hstack((X_x, X_u, X_r))
        return Xt

    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Separate states and inputs.
        X_x = X[:, :self.n_states_out_]
        X_u = X[:, self.n_states_out_:]
        # Extract original states and inputs from first features
        Xt_x = X_x[:, :self.n_states_in_]
        Xt_u = X_u[:, :self.n_inputs_in_]
        # Combine extracted states
        Xt = np.hstack((Xt_x, Xt_u))
        return Xt

    def _validate_parameters(self) -> None:
        if isinstance(self.rbf, str):
            if self.rbf not in self._rbf_lookup.keys():
                raise ValueError('`rbf` must be one of '
                                 f'{self._rbf_lookup.keys()} or a '
                                 '`Callable[[np.ndarray], np.ndarray]`.')
        if self.shape <= 0:
            raise ValueError('`shape` must be greater than zero.')
        if (self.offset is not None) and (self.offset < 0):
            raise ValueError('`offset` must be greater than or equal to zero.')

    def _transform_feature_names(
        self,
        feature_names: np.ndarray,
        format: str = None,
    ) -> np.ndarray:
        if format == 'latex':
            pre = '{'
            post = '}'
            if self.n_inputs_in_ == 0:
                arg = r'{\bf x}'
            else:
                arg = r'{\bf x}, {\bf u}'
        else:
            pre = ''
            post = ''
            if self.n_inputs_in_ == 0:
                arg = 'x'
            else:
                arg = 'x, u'
        names_out = []
        # Deal with episode feature
        if self.episode_feature_:
            names_in = feature_names[1:]
            names_out.append(feature_names[0])
        else:
            names_in = feature_names
        # Add states and inputs
        for ft in range(self.n_states_in_ + self.n_inputs_in_):
            names_out.append(names_in[ft])
        for i in range(self.centers_.n_centers_):
            names_out.append(f'R_{pre}{i}{post}({arg})')
        feature_names_out = np.array(names_out, dtype=object)
        return feature_names_out


KernelApproxEstimator = Union[
    kernel_approximation.KernelApproximation,
    sklearn.kernel_approximation.AdditiveChi2Sampler,
    sklearn.kernel_approximation.Nystroem,
    sklearn.kernel_approximation.PolynomialCountSketch,
    sklearn.kernel_approximation.RBFSampler,
    sklearn.kernel_approximation.SkewedChi2Sampler]
"""Type alias for supported kernel approximation methods."""


class KernelApproxLiftingFn(koopman_pipeline.EpisodeIndependentLiftingFn):
    """Lifting function using random kernel approximation.

    If you are looking for random Fourier features, this is the lifting
    function to use. Randomly binned features are also supported, but are
    experimental. See [RR07]_ for more details.

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
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    kernel_approx_ : KernelApproxEstimator
        Fit kernel approximation estimator.
    n_features_kernel_ : int
        Number of features in the kernel approximation.

    Examples
    --------
    Random Fourier features with :class:`pykoop.RandomFourierKernelApprox`

    >>> rff = pykoop.KernelApproxLiftingFn(
    ...     kernel_approx=pykoop.RandomFourierKernelApprox(
    ...         n_components=10,
    ...         random_state=1234,
    ...     )
    ... )
    >>> rff.fit(X_msd, n_inputs=1, episode_feature=True)
    KernelApproxLiftingFn(kernel_approx=RandomFourierKernelApprox(n_components=10,
    random_state=1234))

    Random Fourier features with
    :class:`sklearn.kernel_approximation.RBFSampler`

    >>> rff = pykoop.KernelApproxLiftingFn(
    ...     kernel_approx=sklearn.kernel_approximation.RBFSampler(
    ...         n_components=10,
    ...         random_state=1234,
    ...     )
    ... )
    >>> rff.fit(X_msd, n_inputs=1, episode_feature=True)
    KernelApproxLiftingFn(kernel_approx=RBFSampler(n_components=10,
    random_state=1234))

    Randomly binned features with :class:`pykoop.RandomBinningKernelApprox`

    >>> rb = pykoop.KernelApproxLiftingFn(
    ...     kernel_approx=pykoop.RandomBinningKernelApprox(
    ...         n_components=10,
    ...         random_state=1234,
    ...     )
    ... )
    >>> rb.fit(X_msd, n_inputs=1, episode_feature=True)
    KernelApproxLiftingFn(kernel_approx=RandomBinningKernelApprox(n_components=10,
    random_state=1234))
    """

    def __init__(self, kernel_approx: KernelApproxEstimator = None) -> None:
        """Instantiate :class:`KernelApproxLiftingFn`.

        Attributes
        ----------
        kernel_approx : KernelApproxEstimator
            Estimator that approximates feature maps from kernels. Can be an
            instance of :class:`pykoop.KernelApprox` or one of the estimators
            from :mod:`sklearn.kernel_approximation`. Defaults to
            :class:`pykoop.RandomFourierKernelApprox` with its default
            arguments.
        """
        self.kernel_approx = kernel_approx

    def _fit_one_ep(self, X: np.ndarray) -> Tuple[int, int]:
        # Clone and fit subestimator
        if self.kernel_approx is None:
            self.kernel_approx_ = \
                kernel_approximation.RandomFourierKernelApprox()
        else:
            self.kernel_approx_ = sklearn.base.clone(self.kernel_approx)
        self.kernel_approx_.fit(X)
        # Calculate number of random features
        if hasattr(self.kernel_approx_, 'n_features_out_'):
            # :class:`pykoop.KernelApprox` supports ``n_features_out_``
            self.n_features_kernel_ = self.kernel_approx_.n_features_out_
        else:
            # Fallback for estimators in :mod:`sklearn.kernel_approximation`
            self.n_features_kernel_ = \
                self.kernel_approx_.get_feature_names_out().size
        # Calculate number of state and input features
        if self.n_inputs_in_ == 0:
            n_states_out = self.n_states_in_ + self.n_features_kernel_
            n_inputs_out = self.n_inputs_in_
        else:
            n_states_out = self.n_states_in_
            n_inputs_out = self.n_inputs_in_ + self.n_features_kernel_
        return (n_states_out, n_inputs_out)

    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Separate states and inputs.
        X_x = X[:, :self.n_states_in_]
        X_u = X[:, self.n_states_in_:]
        # Transform states
        X_kern = self.kernel_approx_.transform(X)
        # Stack results. ``X_kern`` always goes at the bottom because it always
        # involves the lifted inputs if present.
        Xt = np.hstack((X_x, X_u, X_kern))
        return Xt

    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        # Separate states and inputs.
        X_x = X[:, :self.n_states_out_]
        X_u = X[:, self.n_states_out_:]
        # Extract original states and inputs from first features
        Xt_x = X_x[:, :self.n_states_in_]
        Xt_u = X_u[:, :self.n_inputs_in_]
        # Combine extracted states
        Xt = np.hstack((Xt_x, Xt_u))
        return Xt

    def _validate_parameters(self) -> None:
        # No parameters to validate
        pass

    def _transform_feature_names(
        self,
        feature_names: np.ndarray,
        format: str = None,
    ) -> np.ndarray:
        if format == 'latex':
            pre = '{'
            post = '}'
            if self.n_inputs_in_ == 0:
                arg = r'{\bf x}'
            else:
                arg = r'{\bf x}, {\bf u}'
        else:
            pre = ''
            post = ''
            if self.n_inputs_in_ == 0:
                arg = 'x'
            else:
                arg = 'x, u'
        names_out = []
        # Deal with episode feature
        if self.episode_feature_:
            names_in = feature_names[1:]
            names_out.append(feature_names[0])
        else:
            names_in = feature_names
        # Add states and inputs
        for ft in range(self.n_states_in_ + self.n_inputs_in_):
            names_out.append(names_in[ft])
        for i in range(self.n_features_kernel_):
            names_out.append(f'z_{pre}{i}{post}({arg})')
        feature_names_out = np.array(names_out, dtype=object)
        return feature_names_out


class ConstantLiftingFn(koopman_pipeline.EpisodeIndependentLiftingFn):
    """Lifting function that appends a constant term to the input features.

    This is an alternative to allowing bias/offset terms in other lifting
    functions (e.g. :class:`PolynomialLiftingFn`). The aim of explicitly
    including the constant term as a lifting function is to prevent
    accidentally adding it in multiple places when cascading multiple types of
    lifting functions.

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
    feature_names_in_ : np.ndarray
        Array of input feature name strings.

    Examples
    --------
    Add a constant lifted state

    >>> const = pykoop.ConstantLiftingFn()
    >>> const.fit(X_msd, n_inputs=1, episode_feature=True)
    ConstantLiftingFn()
    >>> const.get_feature_names_in().tolist()
    ['ep', 'x0', 'x1', 'u0']
    >>> const.get_feature_names_out().tolist()
    ['ep', 'x0', 'x1', '1', 'u0']
    >>> Xt_msd = const.transform(X_msd[:2, :])
    """

    def __init__(self) -> None:
        """Instantiate :class:`ConstantLiftingFn`."""
        pass

    def _fit_one_ep(self, X: np.ndarray) -> Tuple[int, int]:
        n_states_out = self.n_states_in_ + 1
        n_inputs_out = self.n_inputs_in_
        return (n_states_out, n_inputs_out)

    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        states = X[:, :self.n_states_in_]
        inputs = X[:, self.n_states_in_:]
        ones = np.ones((X.shape[0], 1))
        Xt = np.hstack((states, ones, inputs))
        return Xt

    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        input_start = self.n_states_in_ + 1
        input_stop = input_start + self.n_inputs_in_
        Xt = np.hstack((
            X[:, :self.n_states_in_],
            X[:, input_start:input_stop],
        ))
        return Xt

    def _validate_parameters(self) -> None:
        # No parameters to validate
        pass

    def _transform_feature_names(
        self,
        feature_names: np.ndarray,
        format: str = None,
    ) -> np.ndarray:
        # Deal with episode feature
        if self.episode_feature_:
            names_in = feature_names[1:]
            ep = feature_names[[0]]
        else:
            names_in = feature_names
            ep = None
        # Extract states and inputs
        states = names_in[:self.n_states_in_]
        inputs = names_in[self.n_states_in_:]
        one = np.array(['1'], dtype=object)
        if ep is not None:
            feature_names_out = np.concatenate(
                (ep, states, one, inputs),
                dtype=object,
            )
        else:
            feature_names_out = np.concatenate(
                (states, one, inputs),
                dtype=object,
            )
        return feature_names_out


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
    feature_names_in_ : np.ndarray
        Array of input feature name strings.

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
    >>> delay.get_feature_names_in().tolist()
    ['ep', 'x0', 'x1', 'u0']
    >>> delay.get_feature_names_out().tolist()
    ['ep', 'x0', 'x1', 'D1(x0)', 'D1(x1)', 'u0', 'D1(u0)']
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

    def _transform_feature_names(
        self,
        feature_names: np.ndarray,
        format: str = None,
    ) -> np.ndarray:
        # noqa: D102
        if format == 'latex':
            fn = 'D_'
            pre = '{'
            post = '}'
        else:
            fn = 'D'
            pre = ''
            post = ''
        names_out = []
        # Deal with episode feature
        if self.episode_feature_:
            names_in = feature_names[1:]
            names_out.append(feature_names[0])
        else:
            names_in = feature_names
        # Add state delays
        for delay in range(self.n_delays_state + 1):
            for state in range(self.n_states_in_):
                if delay == 0:
                    names_out.append(f'{names_in[state]}')
                else:
                    names_out.append(
                        f'{fn}{pre}{delay}{post}({names_in[state]})')
        # Add input delays
        for delay in range(self.n_delays_input + 1):
            for input_ in range(self.n_inputs_in_):
                if delay == 0:
                    names_out.append(f'{names_in[self.n_states_in_ + input_]}')
                else:
                    names_out.append(
                        f'{fn}{pre}{delay}{post}'
                        f'({names_in[self.n_states_in_ + input_]})')
        feature_names_out = np.array(names_out, dtype=object)
        return feature_names_out

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
