"""Koopman pipeline meta-estimators and related interfaces."""

import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas
import sklearn.base
import sklearn.metrics
from deprecated import deprecated
from matplotlib import pyplot as plt
from scipy import linalg

from ._sklearn_metaestimators import metaestimators

# Create logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class KoopmanLiftingFn(
        sklearn.base.BaseEstimator,
        sklearn.base.TransformerMixin,
        metaclass=abc.ABCMeta,
):
    """Base class for Koopman lifting functions.

    All attributes with a trailing underscore must be set in the subclass'
    :func:`fit`.

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
    """

    @abc.abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        n_inputs: int = 0,
        episode_feature: bool = False,
    ) -> 'KoopmanLiftingFn':
        """Fit the lifting function.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.
        n_inputs : int
            Number of input features at the end of ``X``.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.

        Returns
        -------
        KoopmanLiftingFn
            Instance of itself.

        Raises
        -----
        ValueError
            If constructor or fit parameters are incorrect.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Transformed data matrix.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert transformed data.

        Parameters
        ----------
        X : np.ndarray
            Transformed data matrix.

        Returns
        -------
        np.ndarray
            Inverted transformed data matrix.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def n_samples_in(self, n_samples_out: int = 1) -> int:
        """Calculate number of input samples required for given output length.

        Parameters
        ----------
        n_samples_out : int
            Number of samples needed at the output.

        Returns
        -------
        int
            Number of samples needed at the input.
        """
        raise NotImplementedError()

    def lift(self, X: np.ndarray, episode_feature: bool = None) -> np.ndarray:
        """Lift state and input.

        Potentially more convenient alternative to calling :func:`transform`.

        Parameters
        ----------
        X : np.ndarray
            State and input.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            Lifted state and input.
        """
        if ((episode_feature == self.episode_feature_)
                or (episode_feature is None)):
            # Can use ``transform`` without modification because
            # ``episode_feature`` is either ``None``, or it's the same as
            # ``self.episode_feature_``.
            Xt = self.transform(X)
        else:
            # ``episode_feature`` and ``self.episode_feature_`` differ, so the
            # input and output need to be padded and/or stripped.
            if self.episode_feature_:
                # Estimator was fit with an episode feature, but the input does
                # not have one. Need to add a fake one and them remove it after
                # transforming.
                X_ep = np.hstack((np.zeros((X.shape[0], 1)), X))
                Xt_ep = self.transform(X_ep)
                Xt = Xt_ep[:, 1:]
            else:
                # Estimator was fit without an episode feature, but the input
                # has one. Need to break up and recombine episodes.
                eps = split_episodes(X, episode_feature=episode_feature)
                eps_t = []
                for (i, X_i) in eps:
                    Xt_i = self.transform(X_i)
                    eps_t.append((i, Xt_i))
                Xt = combine_episodes(eps_t, episode_feature=episode_feature)
        return Xt

    def retract(
        self,
        X: np.ndarray,
        episode_feature: bool = None,
    ) -> np.ndarray:
        """Retract lifted state and input.

        Potentially more convenient alternative to calling
        :func:`inverse_transform`.

        Parameters
        ----------
        X : np.ndarray
            Lifted state and input.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            State and input.
        """
        if ((episode_feature == self.episode_feature_)
                or (episode_feature is None)):
            # Can use ``inverse_transform`` without modification because
            # ``episode_feature`` is either ``None``, or it's the same as
            # ``self.episode_feature_``.
            Xt = self.inverse_transform(X)
        else:
            # ``episode_feature`` and ``self.episode_feature_`` differ, so the
            # input and output need to be padded and/or stripped.
            if self.episode_feature_:
                # Estimator was fit with an episode feature, but the input does
                # not have one. Need to add a fake one and them remove it after
                # inverse-transforming.
                X_ep = np.hstack((np.zeros((X.shape[0], 1)), X))
                Xt_ep = self.inverse_transform(X_ep)
                Xt = Xt_ep[:, 1:]
            else:
                # Estimator was fit without an episode feature, but the input
                # has one. Need to break up and recombine episodes.
                eps = split_episodes(X, episode_feature=episode_feature)
                eps_t = []
                for (i, X_i) in eps:
                    Xt_i = self.inverse_transform(X_i)
                    eps_t.append((i, Xt_i))
                Xt = combine_episodes(eps_t, episode_feature=episode_feature)
        return Xt

    def lift_state(
        self,
        X: np.ndarray,
        episode_feature: bool = None,
    ) -> np.ndarray:
        """Lift state only.

        More convenient alternative to padding the state with dummy inputs,
        calling :func:`transform`, then stripping the unwanted lifted inputs.

        Parameters
        ----------
        X : np.ndarray
            State.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            Lifted state.
        """
        # Pad fake inputs
        X_pad = np.hstack((X, np.zeros((X.shape[0], self.n_inputs_in_))))
        # Lift states with fake inputs
        Xt_pad = self.lift(X_pad, episode_feature=episode_feature)
        # Strip fake lifted inputs
        Xt = Xt_pad[:, :self.n_states_out_ + (1 if episode_feature else 0)]
        return Xt

    def retract_state(
        self,
        X: np.ndarray,
        episode_feature: bool = None,
    ) -> np.ndarray:
        """Retract lifted state only.

        More convenient alternative to padding the lifted state with dummy
        lifted inputs, calling :func:`inverse_transform`.

        Parameters
        ----------
        X : np.ndarray
            Lifted state.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            State.
        """
        # Pad fake lifted inputs
        X_pad = np.hstack((X, np.zeros((X.shape[0], self.n_inputs_out_))))
        # Retract states with fake inputs
        Xt_pad = self.retract(X_pad, episode_feature=episode_feature)
        # Strip fake inputs
        Xt = Xt_pad[:, :self.n_states_in_ + (1 if episode_feature else 0)]
        return Xt

    def lift_input(
        self,
        X: np.ndarray,
        episode_feature: bool = None,
    ) -> np.ndarray:
        """Lift input only.

        More convenient alternative to calling :func:`transform`, then
        stripping the unwanted lifted states.

        Parameters
        ----------
        X : np.ndarray
            State and input.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            Lifted input.
        """
        # Lift states and inputs
        Xt_pad = self.lift(X, episode_feature=episode_feature)
        # Strip lifted states while retaining episode feature as needed
        if episode_feature:
            Xt = np.hstack((
                Xt_pad[:, [0]],
                Xt_pad[:, self.n_states_out_ + 1:],
            ))
        else:
            Xt = Xt_pad[:, self.n_states_out_:]
        return Xt

    def retract_input(
        self,
        X: np.ndarray,
        episode_feature: bool = None,
    ) -> np.ndarray:
        """Retract lifted input only.

        More convenient alternative to padding the lifted state with dummy
        lifted states, calling :func:`inverse_transform`, then stripping the
        unwanted states.

        Parameters
        ----------
        X : np.ndarray
            Lifted input.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            Input.
        """
        # Pad fake lifted states
        if episode_feature:
            X_pad = np.hstack((
                X[:, [0]],
                np.zeros((X.shape[0], self.n_states_out_)),
                X[:, 1:],
            ))
        else:
            X_pad = np.hstack((np.zeros((X.shape[0], self.n_states_out_)), X))
        # Retract inputs with fake states
        Xt_pad = self.retract(X_pad, episode_feature=episode_feature)
        # Strip fake states
        if episode_feature:
            Xt = np.hstack((Xt_pad[:, [0]], Xt_pad[:, self.n_states_in_ + 1:]))
        else:
            Xt = Xt_pad[:, self.n_states_in_:]
        return Xt

    def plot_lifted_trajectory(
        self,
        X: np.ndarray,
        episode_feature: bool = None,
        episode_style: str = None,
        subplots_kw: Dict[str, Any] = None,
        plot_kw: Dict[str, Any] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot lifted data matrix.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.
        episode_style : str
            If ``'columns'``, each episode is a column (default). If
            ``'overlay'``, states from each episode are plotted overtop of each
            other in different colors.
        subplots_kw : Dict[str, Any] = None,
            Keyword arguments for :func:`plt.subplots()`.
        plot_kw : Dict[str, Any] = None,
            Keyword arguments for Matplotlib :func:`plt.Axes.plot()`.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Matplotlib :class:`plt.Figure` object and two-dimensional array of
            :class:`plt.Axes` objects.
        """
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        # Transform data
        Xt = self.lift(X, episode_feature=episode_feature)
        # Split episodes
        eps = split_episodes(
            Xt,
            episode_feature=(self.episode_feature_
                             if episode_feature is None else episode_feature),
        )
        # Figure out dimensions
        n_row = eps[0][1].shape[1]
        n_eps = len(eps)
        n_col = 1 if episode_style == 'overlay' else n_eps
        # Create figure
        subplots_args = {} if subplots_kw is None else subplots_kw
        subplots_args.update({
            'squeeze': False,
            'constrained_layout': True,
            'sharex': 'col',
            'sharey': 'row',
        })
        fig, ax = plt.subplots(n_row, n_col, **subplots_args)
        # Set up plot arguments
        plot_args = {} if plot_kw is None else plot_kw
        plot_args.pop('label', None)
        # Plot results
        for row in range(n_row):
            for ep in range(n_eps):
                if episode_style == 'overlay':
                    ax[row, 0].plot(
                        eps[ep][1][:, row],
                        label=f'Ep. {int(eps[ep][0])}',
                        **plot_args,
                    )
                else:
                    ax[row, ep].plot(eps[ep][1][:, row], **plot_args)
        # Set y labels
        names = self.get_feature_names_out(
            symbols_only=True,
            format='latex',
            episode_feature=False,
        )
        for row in range(n_row):
            ax[row, 0].set_ylabel(f'${names[row]}$')
        # Set x labels and titles
        for col in range(n_col):
            if episode_style != 'overlay':
                ax[0, col].set_title(f'Ep. {int(eps[col][0])}')
            ax[-1, col].set_xlabel('$k$')
        # Set legend
        if episode_style == 'overlay':
            fig.legend(*ax[0, 0].get_legend_handles_labels(),
                       loc='upper right')
        fig.align_labels()
        return fig, ax

    @abc.abstractmethod
    def _transform_feature_names(
        self,
        feature_names: np.ndarray,
        format: str = None,
    ) -> np.ndarray:
        """Transform feature names.

        Parameters
        ----------
        feature_names : np.ndarray
            Feature names.
        format : str
            Feature name formatting method. Possible values are ``'plaintext'``
            (default if ``None``) or ``'latex'``.

        Returns
        -------
        np.ndarray
            Transformed feature names.
        """
        raise NotImplementedError()

    def get_feature_names_out(
        self,
        input_features: np.ndarray = None,
        symbols_only: bool = False,
        format: str = None,
        episode_feature: bool = None,
    ) -> np.ndarray:
        """Get output feature names.

        Parameters
        ----------
        input_features : np.ndarray
            Array of string input feature names. If provided, they are checked
            against ``feature_names_in_``. If ``None``, ignored.
        symbols_only : bool
            If true, only return symbols (``theta_0``, ``upsilon_0``, etc.).
            Otherwise, returns the full equations (default).
        format : str
            Feature name formatting method. Possible values are ``'plaintext'``
            (default if ``None``) or ``'latex'``.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            Output feature names.
        """
        # Handle episode feature
        if episode_feature is None:
            episode_feature = self.episode_feature_
        # Generate features
        if symbols_only:
            feature_names_out = self._generate_feature_names(
                lifted=True,
                format=format,
                episode_feature=episode_feature,
            )
        else:
            feature_names_in = self.get_feature_names_in(format)
            feature_names_tf = self._transform_feature_names(
                feature_names_in,
                format,
            )
            # Handle episode feature after the fact
            if episode_feature and not self.episode_feature_:
                ep = self._generate_feature_names(
                    format=format,
                    episode_feature=True,
                )[[0]]
                feature_names_out = np.concatenate((ep, feature_names_tf))
            elif self.episode_feature_ and not episode_feature:
                feature_names_out = np.delete(feature_names_tf, 0)
            else:
                feature_names_out = feature_names_tf
        return feature_names_out

    def get_feature_names_in(
        self,
        format: str = None,
        episode_feature: bool = None,
    ) -> np.ndarray:
        """Automatically generate input feature names.

        Parameters
        ----------
        format : str
            Feature name formatting method. Possible values are ``'plaintext'``
            (default if ``None``) or ``'latex'``.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            Automatically generated input feaure names.
        """
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        # Handle episode feature
        if episode_feature is None:
            episode_feature = self.episode_feature_
        # Generate features
        if self.feature_names_in_ is None:
            feature_names_in = self._generate_feature_names(
                lifted=False,
                format=format,
                episode_feature=episode_feature,
            )
        else:
            feature_names_in = self.feature_names_in_
        return feature_names_in

    def _validate_feature_names(self, X: np.ndarray) -> None:
        """Validate that input feature names are correct.

        Parameters
        ----------
        X : np.ndarray
            Input array.

        Raises
        ------
        ValueError
            If input feature names do not match fit ones in
            ``feature_names_in_``.
        """
        if not np.all(_extract_feature_names(X) == self.feature_names_in_):
            raise ValueError('Input features do not match fit features.')

    def _generate_feature_names(
        self,
        lifted: bool = False,
        format: str = None,
        episode_feature: bool = None,
    ) -> np.ndarray:
        """Generate feature names.

        Parameters
        ----------
        lifted : bool
            If true, return lifted feature names. If false, return unlifted
            feature names (default).
        format : str
            Feature name formatting method. Possible values are ``'plaintext'``
            (default if ``None``) or ``'latex'``.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            Generated states.
        """
        # Handle episode feature
        if episode_feature is None:
            episode_feature = self.episode_feature_
        names = []
        if lifted:
            if format == 'latex':
                if episode_feature:
                    names.append(r'\mathrm{episode}')
                for k in range(self.n_states_out_):
                    names.append(rf'\vartheta_{{{k}}}')
                for k in range(self.n_inputs_out_):
                    names.append(rf'\upsilon_{{{k}}}')
            else:
                if episode_feature:
                    names.append('ep')
                for k in range(self.n_states_out_):
                    names.append(f'theta{k}')
                for k in range(self.n_inputs_out_):
                    names.append(f'upsilon{k}')
        else:
            if format == 'latex':
                if episode_feature:
                    names.append(r'\mathrm{episode}')
                for k in range(self.n_states_in_):
                    names.append(f'x_{{{k}}}')
                for k in range(self.n_inputs_in_):
                    names.append(f'u_{{{k}}}')
            else:
                if episode_feature:
                    names.append('ep')
                for k in range(self.n_states_in_):
                    names.append(f'x{k}')
                for k in range(self.n_inputs_in_):
                    names.append(f'u{k}')
        feature_names_in = np.array(names, dtype=object)
        return feature_names_in


class EpisodeIndependentLiftingFn(KoopmanLiftingFn):
    """Base class for Koopman lifting functions that are episode-independent.

    Episode-independent lifting functions can be applied to a complete data
    matrix while ignoring the episode feature.

    For example, when rescaling a data matrix, it does not matter which episode
    a sample comes from.

    Attributes
    ----------
    n_features_in_ : int
        Number of features before transformation, including episode feature.
        Set by :func:`fit`.
    n_states_in_ : int
        Number of states before transformation.
        Set by :func:`fit`.
    n_inputs_in_ : int
        Number of inputs before transformation.
        Set by :func:`fit`.
    n_features_out_ : int
        Number of features after transformation, including episode feature.
        Set by :func:`fit`.
    n_states_out_ : int
        Number of states after transformation.
        Set by :func:`fit`.
    n_inputs_out_ : int
        Number of inputs after transformation.
        Set by :func:`fit`.
    min_samples_ : int
        Minimum number of samples needed to use the transformer.
        Set by :func:`fit`.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
        Set by :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    """

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = False) -> 'EpisodeIndependentLiftingFn':
        # noqa: D102
        # Validate constructor parameters
        self._validate_parameters()
        # Set feature names
        self.feature_names_in_ = _extract_feature_names(X)
        # Validate fit parameters
        if n_inputs < 0:
            raise ValueError('`n_inputs` must be greater than or equal to 0.')
        # Save presence of episode feature
        self.episode_feature_ = episode_feature
        # Set up array checks. If you have an episode feature, you need at
        # least one other feature!
        self._check_array_params = {
            'ensure_min_features': 2 if episode_feature else 1,
        }
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Set numbre of input features (including episode feature)
        self.n_features_in_ = X.shape[1]
        # Extract episode feature
        if self.episode_feature_:
            X = X[:, 1:]
        # Set states and inputs in
        self.n_inputs_in_ = n_inputs
        self.n_states_in_ = X.shape[1] - n_inputs
        # Episode independent lifting functions only need one sample.
        n_x, n_u = self._fit_one_ep(X)
        self.n_states_out_ = n_x
        self.n_inputs_out_ = n_u
        self.n_features_out_ = (self.n_states_out_ + self.n_inputs_out_ +
                                (1 if self.episode_feature_ else 0))
        # Episode-independent lifting functions only ever need one sample.
        self.min_samples_ = 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # noqa: D102
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        # Check feature names
        self._validate_feature_names(X)
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'{self.__class__.__name__} `fit()` called '
                             f'with {self.n_features_in_} features, but '
                             f'`transform()` called with {X.shape[1]} '
                             'features.')
        return self._apply_transform_or_inverse(X, 'transform')

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        # noqa: D102
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_out_:
            raise ValueError(f'{self.__class__.__name__} `fit()` output '
                             f'{self.n_features_out_} features, but '
                             '`inverse_transform()` called with '
                             f'{X.shape[1]} features.')
        return self._apply_transform_or_inverse(X, 'inverse_transform')

    def n_samples_in(self, n_samples_out: int = 1) -> int:
        # noqa: D102
        # Episode-independent lifting functions have an input for every output.
        return n_samples_out

    def _apply_transform_or_inverse(self, X: np.ndarray,
                                    transform: str) -> np.ndarray:
        """Strip episode feature, apply transform or inverse, then put it back.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        transform : str
            ``'transform'`` to apply transform or ``'inverse_transform'`` to
            apply inverse transform.

        Returns
        -------
        np.ndarray
            Transformed or inverse transformed data matrix.

        Raises
        ------
        ValueError
            If ``transform`` is not  ``'transform'`` or
            ``'inverse_transform'``.
        """
        # Extract episode feature
        if self.episode_feature_:
            X_ep = X[:, [0]]
            X = X[:, 1:]
        # Transform or inverse transform data
        if transform == 'transform':
            Xt = self._transform_one_ep(X)
        elif transform == 'inverse_transform':
            Xt = self._inverse_transform_one_ep(X)
        else:
            raise ValueError("Parameter `transform` must be one of "
                             "['transform', 'inverse_transform']")
        # Put feature back if needed
        if self.episode_feature_:
            Xt = np.hstack((X_ep, Xt))
        return Xt

    @abc.abstractmethod
    def _fit_one_ep(self, X: np.ndarray) -> Tuple[int, int]:
        """Fit lifting function using a single episode.

        Expects and returns data without an episode header. Data is assumed to
        belong to a single episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        Tuple[int, int]
            Tuple containing the number of state features and input features in
            the transformed data.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        """Transform data using a single episode.

        Expects and returns data without an episode header. Data is assumed to
        belong to a single episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Transformed data matrix.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        """Invert transformed data using a single episode.

        Expects and returns data without an episode header. Data is assumed to
        belong to a single episode.

        Parameters
        ----------
        X : np.ndarray
            Transformed data matrix.

        Returns
        -------
        np.ndarray
            Inverted transformed data matrix.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _validate_parameters(self) -> None:
        """Validate parameters passed in constructor.

        Raises
        ------
        ValueError
            If constructor parameters are incorrect.
        """
        raise NotImplementedError()


class EpisodeDependentLiftingFn(KoopmanLiftingFn):
    """Base class for Koopman lifting functions that are episode-dependent.

    Episode-dependent lifting functions cannot be applied to a complete data
    matrix. The data matrix must be split into episodes, and the lifting
    function must be applied to each one. The resulting lifted episodes are
    then concatenated.

    For example, when applying delay coordinates to a data matrix, samples from
    different episodes must not be intermingled. While a sample from episode 0
    and a sample from episode 1 are adjacent in the data matrix, they did not
    take place one timestep apart! Episode-dependent lifting functions take
    this requirement into account.

    Attributes
    ----------
    n_features_in_ : int
        Number of features before transformation, including episode feature.
        Set by :func:`fit`.
    n_states_in_ : int
        Number of states before transformation.
        Set by :func:`fit`.
    n_inputs_in_ : int
        Number of inputs before transformation.
        Set by :func:`fit`.
    n_features_out_ : int
        Number of features after transformation, including episode feature.
        Set by :func:`fit`.
    n_states_out_ : int
        Number of states after transformation.
        Set by :func:`fit`.
    n_inputs_out_ : int
        Number of inputs after transformation.
        Set by :func:`fit`.
    min_samples_ : int
        Minimum number of samples needed to use the transformer.
        Set by :func:`fit`.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
        Set by :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.

    Notes
    -----
    When :func:`fit` is called with multiple episodes, it only considers the
    first episode. It is assumed that the first episode contains all the
    information needed to properly fit the transformer. Typically, :func:`fit`
    just needs to know the dimensions of the data, so this is a reasonable
    assumption. When :func:`transform` and :func:`inverse_transform` are
    called, they apply the fit transformer to each episode individually.
    """

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = False) -> 'EpisodeDependentLiftingFn':
        # noqa: D102
        # Validate constructor parameters
        self._validate_parameters()
        # Set feature names
        self.feature_names_in_ = _extract_feature_names(X)
        # Validate fit parameters
        if n_inputs < 0:
            raise ValueError('`n_inputs` must be greater than or equal to 0.')
        # Save presence of episode feature
        self.episode_feature_ = episode_feature
        # Set up array checks. If you have an episode feature, you need at
        # least one other feature!
        self._check_array_params = {
            'ensure_min_features': 2 if episode_feature else 1,
        }
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Set number of input features (including episode feature)
        self.n_features_in_ = X.shape[1]
        # Split episodes
        episodes = split_episodes(X, episode_feature=self.episode_feature_)
        first_episode = episodes[0]
        X_first = first_episode[1]
        # Set states and inputs in
        self.n_inputs_in_ = n_inputs
        self.n_states_in_ = X_first.shape[1] - n_inputs
        n_x, n_u, n_k = self._fit_one_ep(X_first)
        self.n_states_out_ = n_x
        self.n_inputs_out_ = n_u
        self.n_features_out_ = (self.n_states_out_ + self.n_inputs_out_ +
                                (1 if self.episode_feature_ else 0))
        self.min_samples_ = n_k
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # noqa: D102
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        # Check feature names
        self._validate_feature_names(X)
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'{self.__class__.__name__} `fit()` called '
                             f'with {self.n_features_in_} features, but '
                             f'`transform()` called with {X.shape[1]} '
                             'features.')
        return self._apply_transform_or_inverse(X, 'transform')

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        # noqa: D102
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self)
        # Validate data
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_out_:
            raise ValueError(f'{self.__class__.__name__} `fit()` output '
                             f'{self.n_features_out_} features, but '
                             '`inverse_transform()` called with '
                             f'{X.shape[1]} features.')
        return self._apply_transform_or_inverse(X, 'inverse_transform')

    def _apply_transform_or_inverse(self, X: np.ndarray,
                                    transform: str) -> np.ndarray:
        """Strip episode feature, apply transform or inverse, then put it back.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        transform : str
            ``'transform'`` to apply transform or ``'inverse_transform'`` to
            apply inverse transform.

        Returns
        -------
        np.ndarray
            Transformed or inverse transformed data matrix.

        Raises
        ------
        ValueError
            If ``transform`` is not  ``'transform'`` or
            ``'inverse_transform'``.
        """
        # Split episodes
        episodes = split_episodes(X, episode_feature=self.episode_feature_)
        # Transform each episode
        transformed_episodes = []
        for (i, X_i) in episodes:
            # Apply transform or inverse to individual episodes
            if transform == 'transform':
                transformed_episode = self._transform_one_ep(X_i)
            elif transform == 'inverse_transform':
                transformed_episode = self._inverse_transform_one_ep(X_i)
            else:
                raise ValueError("Parameter `transform` must be one of "
                                 "['transform', 'inverse_transform']")
            # Add new episode feature back if needed. This is necessary because
            # some transformations may modify the episode length.
            transformed_episodes.append((i, transformed_episode))
        # Concatenate the transformed episodes
        Xt = combine_episodes(transformed_episodes,
                              episode_feature=self.episode_feature_)
        return Xt

    @abc.abstractmethod
    def _fit_one_ep(self, X: np.ndarray) -> Tuple[int, int, int]:
        """Fit lifting function using a single episode.

        Expects and returns data without an episode header. Data is assumed to
        belong to a single episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        Tuple[int, int, int]
            Tuple containing the number of state features in the transformed
            data, the number of input features in the transformed data, and
            the minimum number of samples required to use the transformer.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        """Transform data using a single episode.

        Expects and returns data without an episode header. Data is assumed to
        belong to a single episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Transformed data matrix.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _inverse_transform_one_ep(self, X: np.ndarray) -> np.ndarray:
        """Invert transformed data using a single episode.

        Expects and returns data without an episode header. Data is assumed to
        belong to a single episode.

        Parameters
        ----------
        X : np.ndarray
            Transformed data matrix.

        Returns
        -------
        np.ndarray
            Inverted transformed data matrix.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _validate_parameters(self) -> None:
        """Validate parameters passed in constructor.

        Raises
        ------
        ValueError
            If constructor parameters are incorrect.
        """
        raise NotImplementedError()


class KoopmanRegressor(sklearn.base.BaseEstimator,
                       sklearn.base.RegressorMixin,
                       metaclass=abc.ABCMeta):
    """Base class for Koopman regressors.

    All attributes with a trailing underscore are set by :func:`fit`.

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
    coef_ : np.ndarray
        Fit coefficient matrix.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    """

    # Array check parameters for :func:`fit` when ``X`` and ``y` are given
    _check_X_y_params: Dict[str, Any] = {
        'multi_output': True,
        'y_numeric': True,
    }

    # Array check parameters for :func:`predict` and :func:`fit` when only
    # ``X`` is given
    _check_array_params: Dict[str, Any] = {
        'dtype': 'numeric',
    }

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = False) -> 'KoopmanRegressor':
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
        KoopmanRegressor
            Instance of itself.

        Raises
        -----
        ValueError
            If constructor or fit parameters are incorrect.
        """
        # Set feature names
        self.feature_names_in_ = _extract_feature_names(X)
        # Check ``X`` differently depending on whether ``y`` is given
        if y is None:
            X = sklearn.utils.validation.check_array(
                X, **self._check_array_params)
        else:
            X, y = sklearn.utils.validation.check_X_y(X, y,
                                                      **self._check_X_y_params)
        # Validate constructor parameters
        self._validate_parameters()
        # Compute fit attributes
        self.n_features_in_ = X.shape[1]
        self.n_inputs_in_ = n_inputs
        self.n_states_in_ = (X.shape[1] - n_inputs -
                             (1 if episode_feature else 0))
        self.episode_feature_ = episode_feature
        # Split ``X`` if needed
        if y is None:
            X_unshifted, X_shifted = shift_episodes(
                X,
                n_inputs=self.n_inputs_in_,
                episode_feature=self.episode_feature_)
        else:
            X_unshifted = X
            X_shifted = y
        # Strip episode feature if present
        if self.episode_feature_:
            X_unshifted_noep = X_unshifted[:, 1:]
            X_shifted_noep = X_shifted[:, 1:]
        else:
            X_unshifted_noep = X_unshifted
            X_shifted_noep = X_shifted
        # Call fit from subclass
        self.coef_ = self._fit_regressor(X_unshifted_noep, X_shifted_noep)
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
        # Check if fitted
        sklearn.utils.validation.check_is_fitted(self)
        # Check feature names
        self._validate_feature_names(X)
        # Validate array
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Split episodes
        episodes = split_episodes(X, episode_feature=self.episode_feature_)
        # Predict for each episode
        predictions = []
        for (i, X_i) in episodes:
            predictions.append((i, X_i @ self.coef_))
        # Combine and return
        X_pred = combine_episodes(
            predictions,
            episode_feature=self.episode_feature_,
        )
        return X_pred

    @abc.abstractmethod
    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        """Fit the regressor using shifted and unshifted data matrices.

        The input data matrices must not have episode features.

        Parameters
        ----------
        X_unshifted : np.ndarray
            Unshifted data matrix without episode feature.
        X_shifted : np.ndarray
            Shifted data matrix without episode feature.

        Returns
        -------
        np.ndarray
            Fit coefficient matrix.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _validate_parameters(self) -> None:
        """Validate parameters passed in constructor.

        Raises
        ------
        ValueError
            If constructor parameters are incorrect.
        """
        raise NotImplementedError()

    def plot_bode(
        self,
        t_step: float,
        f_min: float = 0,
        f_max: float = None,
        n_points: int = 1000,
        decibels: bool = True,
        subplots_kw: Dict[str, Any] = None,
        plot_kw: Dict[str, Any] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot frequency response of Koopman system.

        Parameters
        ----------
        t_step : float
            Sampling timestep.
        f_min : float
            Minimum frequency to plot.
        f_max : float
            Maximum frequency to plot.
        n_points : int
            Number of frequecy points to plot.
        decibels : bool
            Plot gain in dB (default is true).
        subplots_kw : Dict[str, Any] = None,
            Keyword arguments for :func:`plt.subplots()`.
        plot_kw : Dict[str, Any] = None,
            Keyword arguments for Matplotlib :func:`plt.Axes.plot()`.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Matplotlib :class:`plt.Figure` object and two-dimensional array of
            :class:`plt.Axes` objects.

        Raises
        ------
        ValueError
            If ``f_min`` is less than zero or ``f_max`` is greater than the
            Nyquist frequency.
        """
        sklearn.utils.validation.check_is_fitted(self)
        # Get Koopman ``A`` and ``B`` matrices
        koop_mat = self.coef_.T
        A = koop_mat[:, :koop_mat.shape[0]]
        B = koop_mat[:, koop_mat.shape[0]:]

        def _sigma_bar_G(f):
            """Compute Bode plot at given frequency."""
            z = np.exp(1j * 2 * np.pi * f * t_step)
            G = linalg.solve((np.diag([z] * A.shape[0]) - A), B)
            return linalg.svdvals(G)[0]

        # Generate frequency response data
        f_samp = 1 / t_step
        if f_min < 0:
            raise ValueError('`f_min` must be at least 0.')
        if f_max is None:
            f_max = f_samp / 2
        if f_max > f_samp / 2:
            raise ValueError(
                '`f_max` must be less than the Nyquist frequency.')
        f_plot = np.linspace(f_min, f_max, n_points)
        bode = []
        for k in range(f_plot.size):
            bode.append(_sigma_bar_G(f_plot[k]))
        mag = np.array(bode)
        mag_db = 20 * np.log10(mag)
        # Create figure
        subplots_args = {} if subplots_kw is None else subplots_kw
        subplots_args.update({
            'constrained_layout': True,
        })
        fig, ax = plt.subplots(**subplots_args)
        # Set up plot arguments
        plot_args = {} if plot_kw is None else plot_kw
        # Plot data
        ylabel = r'$\bar{\sigma}\left({\bf G}(e^{j \theta})\right)$'
        if decibels:
            ax.semilogx(f_plot, mag_db, **plot_args)
            ax.set_ylabel(f'{ylabel} (dB)')
        else:
            ax.semilogx(f_plot, mag, **plot_args)
            ax.set_ylabel(f'{ylabel} (unitless gain)')
        ax.set_xlabel(r'$f$ (Hz)')
        return fig, ax

    def plot_eigenvalues(
        self,
        unit_circle: bool = True,
        figure_kw: Dict[str, Any] = None,
        subplot_kw: Dict[str, Any] = None,
        plot_kw: Dict[str, Any] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot eigenvalues of Koopman ``A`` matrix.

        Parameters
        ----------
        figure_kw : Dict[str, Any] = None,
            Keyword arguments for :func:`plt.figure()`.
        subplot_kw : Dict[str, Any] = None,
            Keyword arguments for :func:`plt.subplot()`.
        plot_kw : Dict[str, Any] = None,
            Keyword arguments for Matplotlib :func:`plt.Axes.plot()`.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Matplotlib :class:`plt.Figure` object and two-dimensional array of
            :class:`plt.Axes` objects.
        """
        sklearn.utils.validation.check_is_fitted(self)
        # Get Koopman ``A`` matrix
        koop_mat = self.coef_.T
        A = koop_mat[:, :koop_mat.shape[0]]
        # Calculate eigenvalues
        eigv = linalg.eig(A)[0]
        # Create figure
        figure_args = {} if figure_kw is None else figure_kw
        fig = plt.figure(**figure_args)
        subplot_args = {} if subplot_kw is None else subplot_kw
        subplot_args.pop('projection', None)
        ax = plt.subplot(projection='polar', **subplot_args)
        plot_args = {} if plot_kw is None else plot_kw
        # Plot eigenvalues
        ax.scatter(np.angle(eigv), np.absolute(eigv), **plot_args)
        # Plot unit circle
        if unit_circle:
            plot_args.pop('linestyle', None)
            plot_args.pop('color', None)
            th = np.linspace(0, 2 * np.pi)
            ax.plot(
                th,
                np.ones(th.shape),
                linestyle='--',
                color='k',
                **plot_args,
            )
        # Set labels
        ax.set_xlabel(r'$\mathrm{Re}(\lambda)$')
        ax.set_ylabel(r'$\mathrm{Im}(\lambda)$', labelpad=30)
        return fig, ax

    def plot_koopman_matrix(
        self,
        subplots_kw: Dict[str, Any] = None,
        plot_kw: Dict[str, Any] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot heatmap of Koopman matrices.

        Parameters
        ----------
        subplots_kw : Dict[str, Any] = None,
            Keyword arguments for :func:`plt.subplots()`.
        plot_kw : Dict[str, Any] = None,
            Keyword arguments for Matplotlib :func:`plt.Axes.plot()`.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Matplotlib :class:`plt.Figure` object and two-dimensional array of
            :class:`plt.Axes` objects.
        """
        sklearn.utils.validation.check_is_fitted(self)
        U = self.coef_.T
        p_theta, p = U.shape
        # Create figure
        subplots_args = {} if subplots_kw is None else subplots_kw
        subplots_args.update({
            'constrained_layout': True,
        })
        fig, ax = plt.subplots(**subplots_args)
        # Get max magnitude for colorbar
        mag = np.max(np.abs(U))
        plot_args = {} if plot_kw is None else plot_kw
        plot_args.update({
            'vmin': -mag,
            'vmax': mag,
            'cmap': 'seismic',
        })
        im = ax.matshow(U, **plot_args)
        # Plot line to separate ``A`` and ``B``
        ax.vlines(
            p_theta - 0.5,
            -0.5,
            p_theta - 0.5,
            linestyle='--',
            color='k',
            linewidth=2,
        )
        ax.text(0, p_theta, r'${\bf A}$')
        ax.text(p_theta, p_theta, r'${\bf B}$')
        fig.colorbar(im, ax=ax, orientation='horizontal')
        return fig, ax

    def plot_svd(
        self,
        subplots_kw: Dict[str, Any] = None,
        plot_kw: Dict[str, Any] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot singular values of Koopman matrices.

        Parameters
        ----------
        subplots_kw : Dict[str, Any] = None,
            Keyword arguments for :func:`plt.subplots()`.
        plot_kw : Dict[str, Any] = None,
            Keyword arguments for Matplotlib :func:`plt.Axes.plot()`.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Matplotlib :class:`plt.Figure` object and two-dimensional array of
            :class:`plt.Axes` objects.
        """
        sklearn.utils.validation.check_is_fitted(self)
        koop_mat = self.coef_.T
        A = koop_mat[:, :koop_mat.shape[0]]
        B = koop_mat[:, koop_mat.shape[0]:]
        # Create figure
        subplots_args = {} if subplots_kw is None else subplots_kw
        subplots_args.update({
            'squeeze': True,
            'constrained_layout': True,
            'sharey': 'row',
        })
        fig, ax = plt.subplots(1, 3, **subplots_args)
        # Compute singular values
        sv_U = linalg.svdvals(koop_mat)
        sv_A = linalg.svdvals(A)
        sv_B = linalg.svdvals(B)
        # Plot singular values
        plot_args = {} if plot_kw is None else plot_kw
        plot_args.update({
            'marker': '.',
        })
        ax[0].semilogy(sv_U, **plot_args)
        ax[1].semilogy(sv_A, **plot_args)
        ax[2].semilogy(sv_B, **plot_args)
        ax[0].set_xlabel(r'$i$')
        ax[1].set_xlabel(r'$i$')
        ax[2].set_xlabel(r'$i$')
        ax[0].set_ylabel(r'$\sigma_i({\bf U})$')
        ax[1].set_ylabel(r'$\sigma_i({\bf A})$')
        ax[2].set_ylabel(r'$\sigma_i({\bf B})$')
        return fig, ax

    def _validate_feature_names(self, X: np.ndarray) -> None:
        """Validate that input feature names are correct.

        Parameters
        ----------
        X : np.ndarray
            Input array.

        Raises
        ------
        ValueError
            If input feature names do not match fit ones in
            ``feature_names_in_``.
        """
        if not np.all(_extract_feature_names(X) == self.feature_names_in_):
            raise ValueError('Input features do not match fit features.')

    # Extra estimator tags
    # https://scikit-learn.org/stable/developers/develop.html#estimator-tags
    def _more_tags(self):
        return {
            'multioutput': True,
            'multioutput_only': True,
        }


class SplitPipeline(metaestimators._BaseComposition, KoopmanLiftingFn):
    """Meta-estimator for lifting states and inputs separately.

    Only works with episode-independent lifting functions! It's too complicated
    to make this work with :class:`DelayLiftingFn`, especially when you can
    just set ``n_delays_input=0``.

    Attributes
    ----------
    lifting_functions_state_: List[Tuple[str, EpisodeIndependentLiftingFn]]
        Fit state lifting functions (and their names).
    lifting_functions_input_: List[Tuple[str, EpisodeIndependentLiftingFn]]
        Fit input lifting functions (and their names).
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
    Apply split pipeline to mass-spring-damper data

    >>> kp = pykoop.SplitPipeline(
    ...     lifting_functions_state=[
    ...         ('pl', pykoop.PolynomialLiftingFn(order=2))
    ...     ],
    ...     lifting_functions_input=None,
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)
    SplitPipeline(lifting_functions_state=[('pl',
    PolynomialLiftingFn(order=2))])
    >>> Xt_msd = kp.transform(X_msd[:2, :])
    """

    # Array check parameters for :func:`predict` and :func:`fit` when only
    # ``X`` is given
    _check_array_params = {
        'dtype': 'numeric',
    }

    def __init__(
        self,
        lifting_functions_state: List[Tuple[
            str, EpisodeIndependentLiftingFn]] = None,
        lifting_functions_input: List[Tuple[
            str, EpisodeIndependentLiftingFn]] = None,
    ) -> None:
        """Instantiate :class:`SplitPipeline`.

        Parameters
        ----------
        lifting_functions_state : List[Tuple[str, EpisodeIndependentLiftingFn]]
            Lifting functions to apply to the state features (and their names).
        lifting_functions_input : List[Tuple[str, EpisodeIndependentLiftingFn]]
            Lifting functions to apply to the input features (and their names).
        """
        self.lifting_functions_state = lifting_functions_state
        self.lifting_functions_input = lifting_functions_input

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = False) -> 'SplitPipeline':
        # noqa: D102
        # Set feature names
        self.feature_names_in_ = _extract_feature_names(X)
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Save state of episode feature
        self.episode_feature_ = episode_feature
        # Set number of features
        self.n_features_in_ = X.shape[1]
        self.n_states_in_ = (X.shape[1] - n_inputs -
                             (1 if episode_feature else 0))
        self.n_inputs_in_ = n_inputs
        # Clone state lifting functions
        used_keys = []
        self.lifting_functions_state_ = []
        if self.lifting_functions_state is not None:
            for key, lf in self.lifting_functions_state:
                used_keys.append(key)
                self.lifting_functions_state_.append(
                    tuple((key, sklearn.base.clone(lf))))
        # Clone input lifting functions
        self.lifting_functions_input_ = []
        if self.lifting_functions_input is not None:
            for key, lf in self.lifting_functions_input:
                used_keys.append(key)
                self.lifting_functions_input_.append(
                    tuple((key, sklearn.base.clone(lf))))
        # Check names
        self._validate_names(used_keys)
        # Separate episode feature
        if self.episode_feature_:
            X_ep = X[:, [0]]
            X = X[:, 1:]
        # Split state and input
        X_state = X[:, :self.n_states_in_]
        X_input = X[:, self.n_states_in_:]
        # Put back episode feature if needed
        if self.episode_feature_:
            X_state = np.hstack((
                X_ep,
                X_state,
            ))
            X_input = np.hstack((
                X_ep,
                X_input,
            ))
        # Fit and transform states
        X_out_state = X_state
        for _, lf in self.lifting_functions_state_:
            X_out_state = lf.fit_transform(
                X_out_state,
                n_inputs=0,
                episode_feature=self.episode_feature_,
            )
        # Fit and transform inputs
        X_out_input = X_input
        for _, lf in self.lifting_functions_input_:
            X_out_input = lf.fit_transform(
                X_out_input,
                n_inputs=X_out_input.shape[1],
                episode_feature=self.episode_feature_,
            )
        # Compute output dimensions for states
        if len(self.lifting_functions_state_) > 0:
            # Compute number of output states
            last_tf = self.lifting_functions_state_[-1][1]
            if last_tf.n_inputs_out_ != 0:
                raise RuntimeError(f'Lifting function {last_tf} was called '
                                   'with `n_inputs=0` but `n_inputs_out_` is '
                                   'not 0. Is it implemented correctly?')
            self.n_states_out_ = last_tf.n_states_out_
        else:
            self.n_states_out_ = self.n_states_in_
        # Compute output dimensions for inputs
        if len(self.lifting_functions_input_) > 0:
            # Compute number of output states
            last_tf = self.lifting_functions_input_[-1][1]
            if last_tf.n_states_out_ != 0:
                raise RuntimeError(f'Lifting function {last_tf} was called '
                                   f'with `n_inputs={last_tf.n_inputs_in_}` '
                                   'but `n_states_out_` is not 0. Is it '
                                   'implemented correctly?')
            self.n_inputs_out_ = last_tf.n_inputs_out_
        else:
            self.n_inputs_out_ = self.n_inputs_in_
        # Compute number of features and minimum samples needed
        self.n_features_out_ = (self.n_states_out_ + self.n_inputs_out_ +
                                (1 if self.episode_feature_ else 0))
        # Since all lifting functions are episode-independent, we only ever
        # need one sample.
        self.min_samples_ = 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # noqa: D102
        # Check if fitted
        sklearn.utils.validation.check_is_fitted(self)
        # Check feature names
        self._validate_feature_names(X)
        # Validate input array
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'{self.__class__.__name__} `fit()` called '
                             f'with {self.n_features_in_} features, but '
                             f'`transform()` called with {X.shape[1]} '
                             'features.')
        # Separate episode feature
        if self.episode_feature_:
            X_ep = X[:, [0]]
            X = X[:, 1:]
        # Split state and input
        X_state = X[:, :self.n_states_in_]
        X_input = X[:, self.n_states_in_:]
        # Put back episode feature if needed
        if self.episode_feature_:
            X_state = np.hstack((
                X_ep,
                X_state,
            ))
            X_input = np.hstack((
                X_ep,
                X_input,
            ))
        # Fit and transform states
        X_out_state = X_state
        for _, lf in self.lifting_functions_state_:
            X_out_state = lf.transform(X_out_state)
        # Fit and transform inputs
        X_out_input = X_input
        for _, lf in self.lifting_functions_input_:
            X_out_input = lf.transform(X_out_input)
        if self.episode_feature_:
            Xt = np.hstack((
                X_out_state,
                X_out_input[:, 1:],
            ))
        else:
            Xt = np.hstack((
                X_out_state,
                X_out_input,
            ))
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        # noqa: D102
        sklearn.utils.validation.check_is_fitted(self)
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_out_:
            raise ValueError(f'{self.__class__.__name__} `fit()` output '
                             f'{self.n_features_out_} features, but '
                             '`inverse_transform()` called with '
                             f'{X.shape[1]} features.')
        if self.episode_feature_:
            X_ep = X[:, [0]]
            X = X[:, 1:]
        # Split state and input
        X_state = X[:, :self.n_states_out_]
        X_input = X[:, self.n_states_out_:]
        # Put back episode feature if needed
        if self.episode_feature_:
            X_state = np.hstack((
                X_ep,
                X_state,
            ))
            X_input = np.hstack((
                X_ep,
                X_input,
            ))
        # Fit and inverse transform states
        X_out_state = X_state
        for _, lf in self.lifting_functions_state_[::-1]:
            X_out_state = lf.inverse_transform(X_out_state)
        # Fit and transform inputs
        X_out_input = X_input
        for _, lf in self.lifting_functions_input_[::-1]:
            X_out_input = lf.inverse_transform(X_out_input)
        if self.episode_feature_:
            Xt = np.hstack((
                X_out_state,
                X_out_input[:, 1:],
            ))
        else:
            Xt = np.hstack((
                X_out_state,
                X_out_input,
            ))
        return Xt

    def n_samples_in(self, n_samples_out: int = 1) -> int:
        # noqa: D102
        # Since this pipeline only works with episode-independent lifting
        # functions, we know ``n_samples_in == n_samples_out``.
        return n_samples_out

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        # noqa: D102
        # A bit inefficient to do this twice but it's not the end of the world.
        state = self._get_params('lifting_functions_state', deep=deep)
        input = self._get_params('lifting_functions_input', deep=deep)
        state.update(input)
        return state

    def set_params(self, **kwargs) -> 'SplitPipeline':
        # noqa: D102
        # A bit inefficient to do this twice but it's not the end of the world.
        self._set_params('lifting_functions_state', **kwargs)
        self._set_params('lifting_functions_input', **kwargs)
        return self

    def _transform_feature_names(
        self,
        feature_names: np.ndarray,
        format: str = None,
    ) -> np.ndarray:
        # noqa: D102
        # Deal with episode feature
        if self.episode_feature_:
            names_in = feature_names[1:]
            ep = feature_names[[0]]
        else:
            names_in = feature_names
            ep = None
        # Split states and inputs
        # breakpoint()
        names_in_state = names_in[:self.n_states_in_]
        names_in_input = names_in[self.n_states_in_:]
        if self.episode_feature_:
            names_in_state = np.hstack((ep, names_in_state))
            names_in_input = np.hstack((ep, names_in_input))
        # Transform state and input
        names_out_state = names_in_state
        for _, lf in self.lifting_functions_state_:
            names_out_state = lf._transform_feature_names(
                names_out_state,
                format,
            )
        names_out_input = names_in_input
        for _, lf in self.lifting_functions_input_:
            names_out_input = lf._transform_feature_names(
                names_out_input,
                format,
            )
        # Recombine
        if self.episode_feature_:
            feature_names_out = np.concatenate((
                names_out_state,
                names_out_input[1:],
            ),
                                               dtype=object)
        else:
            feature_names_out = np.concatenate((
                names_out_state,
                names_out_input,
            ),
                                               dtype=object)
        return feature_names_out


class KoopmanPipeline(metaestimators._BaseComposition, KoopmanLiftingFn):
    """Meta-estimator for chaining lifting functions with an estimator.

    Attributes
    ----------
    liting_functions_ : List[Tuple[str, KoopmanLiftingFn]]
        Fit lifting functions (and their names).
    regressor_ : KoopmanRegressor
        Fit regressor.
    transformers_fit_ : bool
        True if lifting functions have been fit.
    regressor_fit_ : bool
        True if regressor has been fit.
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
    Apply a basic Koopman pipeline to mass-spring-damper data

    >>> kp = pykoop.KoopmanPipeline(
    ...     lifting_functions=[('pl', pykoop.PolynomialLiftingFn(order=2))],
    ...     regressor=pykoop.Edmd(),
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)
    KoopmanPipeline(lifting_functions=[('pl', PolynomialLiftingFn(order=2))],
    regressor=Edmd())
    >>> kp.get_feature_names_in().tolist()
    ['ep', 'x0', 'x1', 'u0']
    >>> kp.get_feature_names_out().tolist()
    ['ep', 'x0', 'x1', 'x0^2', 'x0*x1', 'x1^2', 'u0', 'x0*u0', 'x1*u0', 'u0^2']

    Apply more sophisticated Koopman pipeline to mass-spring-damper data

    >>> kp = KoopmanPipeline(
    ...     lifting_functions=[
    ...         ('ma', pykoop.SkLearnLiftingFn(
    ...                    sklearn.preprocessing.MaxAbsScaler())),
    ...         ('pl', pykoop.PolynomialLiftingFn(order=2)),
    ...         ('ss', pykoop.SkLearnLiftingFn(
    ...                    sklearn.preprocessing.StandardScaler())),
    ...     ],
    ...     regressor=pykoop.Edmd(),
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)
    KoopmanPipeline(lifting_functions=[('ma',
    SkLearnLiftingFn(transformer=MaxAbsScaler())),
    ('pl', PolynomialLiftingFn(order=2)),
    ('ss', SkLearnLiftingFn(transformer=StandardScaler()))],
    regressor=Edmd())
    >>> Xt_msd = kp.transform(X_msd[:2, :])

    Apply bilinear Koopman pipeline to mass-spring-damper data

    >>> kp = KoopmanPipeline(
    ...     lifting_functions=[
    ...         ('ma', pykoop.SkLearnLiftingFn(
    ...                    sklearn.preprocessing.MaxAbsScaler())),
    ...         ('sp', pykoop.SplitPipeline(
    ...             lifting_functions_state=[
    ...                 ('pl', pykoop.PolynomialLiftingFn(order=2)),
    ...             ],
    ...             lifting_functions_input=None,
    ...         )),
    ...         ('bi', pykoop.BilinearInputLiftingFn()),
    ...         ('ss', pykoop.SkLearnLiftingFn(
    ...                    sklearn.preprocessing.StandardScaler())),
    ...     ],
    ...     regressor=pykoop.Edmd(),
    ... )
    >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)
    KoopmanPipeline(lifting_functions=[('ma',
    SkLearnLiftingFn(transformer=MaxAbsScaler())),
    ('sp', SplitPipeline(lifting_functions_state=[('pl',
    PolynomialLiftingFn(order=2))])),
    ('bi', BilinearInputLiftingFn()),
    ('ss', SkLearnLiftingFn(transformer=StandardScaler()))],
    regressor=Edmd())
    """

    # Array check parameters for :func:`predict` and :func:`fit` when only
    # ``X`` is given
    _check_array_params = {
        'dtype': 'numeric',
    }

    def __init__(
        self,
        lifting_functions: List[Tuple[str, KoopmanLiftingFn]] = None,
        regressor: KoopmanRegressor = None,
    ) -> None:
        """Instantiate for :class:`KoopmanPipeline`.

        Parameters
        ----------
        lifting_functions : List[Tuple[str, KoopmanLiftingFn]]
            List of names and lifting function objects.
        regressor : KoopmanRegressor
            Koopman regressor.
        """
        self.lifting_functions = lifting_functions
        self.regressor = regressor

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            n_inputs: int = 0,
            episode_feature: bool = False) -> 'KoopmanPipeline':
        """Fit the Koopman pipeline.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.
        n_inputs : int
            Number of input features at the end of ``X``.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.

        Returns
        -------
        KoopmanPipeline
            Instance of itself.

        Raises
        -----
        ValueError
            If constructor or fit parameters are incorrect.
        """
        # Deliberately do not overwrite ``X`` with result, because
        # ``self.fit_transformers()`` and ``self.regressor_.fit()`` also
        # call ``sklearn.utils.validation.check_array()``, and we don't want to
        # strip the feature names for them.
        sklearn.utils.validation.check_array(
            X,
            ensure_min_samples=2,
            **self._check_array_params,
        )
        if self.regressor is None:
            raise ValueError(
                '`regressor` must be specified in order to use `fit()`.')
        # Clone regressor
        self.regressor_ = sklearn.base.clone(self.regressor)
        # Fit transformers and transform input
        self.fit_transformers(
            X,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        Xt = self.transform(X)
        # Fit the regressor
        self.regressor_.fit(
            Xt,
            n_inputs=self.n_inputs_out_,
            episode_feature=self.episode_feature_,
        )
        self.regressor_fit_ = True
        return self

    def fit_transformers(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        n_inputs: int = 0,
        episode_feature: bool = False,
    ) -> 'KoopmanPipeline':
        """Fit only the lifting functions in the pipeline.

        .. todo:: Rename to ``partial_fit``.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.
        n_inputs : int
            Number of input features at the end of ``X``.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.

        Returns
        -------
        KoopmanPipeline
            Instance of itself.

        Raises
        -----
        ValueError
            If constructor or fit parameters are incorrect.
        """
        # Set feature names
        self.feature_names_in_ = _extract_feature_names(X)
        # Validate input array
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Save state of episode feature
        self.episode_feature_ = episode_feature
        # Set number of features
        self.n_features_in_ = X.shape[1]
        self.n_states_in_ = (X.shape[1] - n_inputs -
                             (1 if episode_feature else 0))
        self.n_inputs_in_ = n_inputs
        used_keys = []
        self.lifting_functions_ = []
        if self.lifting_functions is not None:
            for key, lf in self.lifting_functions:
                used_keys.append(key)
                self.lifting_functions_.append(
                    tuple((key, sklearn.base.clone(lf))))
        # Check names
        self._validate_names(used_keys)
        # Fit and transform lifting functions
        X_out = X
        n_inputs_out = n_inputs
        for _, lf in self.lifting_functions_:
            X_out = lf.fit_transform(X_out,
                                     n_inputs=n_inputs_out,
                                     episode_feature=episode_feature)
            n_inputs_out = lf.n_inputs_out_
        # Set output dimensions
        if len(self.lifting_functions_) > 0:
            # Find the last transformer and use it to get output dimensions
            last_pp = self.lifting_functions_[-1][1]
            self.n_features_out_ = last_pp.n_features_out_
            self.n_states_out_ = last_pp.n_states_out_
            self.n_inputs_out_ = last_pp.n_inputs_out_
        else:
            # Fall back on input dimensions
            self.n_features_out_ = self.n_features_in_
            self.n_states_out_ = self.n_states_in_
            self.n_inputs_out_ = self.n_inputs_in_
        self.min_samples_ = self.n_samples_in(1)
        self.transformers_fit_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Transformed data matrix.
        """
        # Check if fitted
        sklearn.utils.validation.check_is_fitted(self, 'transformers_fit_')
        # Check feature names
        self._validate_feature_names(X)
        # Validate input array
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'{self.__class__.__name__} `fit()` called '
                             f'with {self.n_features_in_} features, but '
                             f'`transform()` called with {X.shape[1]} '
                             'features.')
        # Apply lifting functions
        X_out = X
        for _, lf in self.lifting_functions_:
            X_out = lf.transform(X_out)
        return X_out

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert transformed data.

        Parameters
        ----------
        X : np.ndarray
            Transformed data matrix.

        Returns
        -------
        np.ndarray
            Inverted transformed data matrix.
        """
        sklearn.utils.validation.check_is_fitted(self, 'transformers_fit_')
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_out_:
            raise ValueError(f'{self.__class__.__name__} `fit()` output '
                             f'{self.n_features_out_} features, but '
                             '`inverse_transform()` called with '
                             f'{X.shape[1]} features.')
        # Apply inverse lifting functions in reverse order
        X_out = X
        for _, lf in self.lifting_functions_[::-1]:
            X_out = lf.inverse_transform(X_out)
        return X_out

    def n_samples_in(self, n_samples_out: int = 1) -> int:
        # noqa: D102
        # Compute minimum number of samples needed by transformer.
        # Each transformer knows how many input samples it needs to produce
        # a given number of output samples.  Knowing we just want one
        # sample at the output, we work backwards to figure out how many
        # samples we need at the beginning of the pipeline.
        n_samples_in = n_samples_out
        if self.lifting_functions is not None:
            for _, tf in self.lifting_functions[::-1]:
                n_samples_in = tf.n_samples_in(n_samples_in)
        return n_samples_in

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform a single-step prediction for each state in each episode.

        Lifts the state, preforms a single-step prediction in the lifted space,
        then retracts to the original state space.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Predicted data matrix.
        """
        # Check if fitted
        sklearn.utils.validation.check_is_fitted(self, 'regressor_fit_')
        # Check feature names
        self._validate_feature_names(X)
        # Validate input array
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Lift data matrix
        X_trans = self.transform(X)
        # Predict in lifted space
        X_pred = self.regressor_.predict(X_trans)
        # Pad inputs wth zeros to do inverse
        if self.n_inputs_out_ != 0:
            X_pred_pad = np.hstack((
                X_pred,
                np.zeros((X_pred.shape[0], self.n_inputs_out_)),
            ))
        else:
            X_pred_pad = X_pred
        # Invert lifting functions
        X_pred_pad_inv = self.inverse_transform(X_pred_pad)
        # Strip zero inputs
        if self.n_inputs_in_ != 0:
            X_pred_inv = X_pred_pad_inv[:, :(self.n_features_in_
                                             - self.n_inputs_in_)]
        else:
            X_pred_inv = X_pred_pad_inv
        return X_pred_inv

    def score(self, X: np.ndarray, y: np.ndarray = None) -> float:
        """Calculate prediction score.

        For more flexible scoring, see :func:`make_scorer`.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.

        Returns
        -------
        float
            Mean squared error prediction score.
        """
        # Check if fitted
        sklearn.utils.validation.check_is_fitted(self, 'regressor_fit_')
        # Check feature names
        self._validate_feature_names(X)
        # Validate input array
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        scorer = KoopmanPipeline.make_scorer()
        score = scorer(self, X, None)
        return score

    @deprecated('Use `predict_trajectory` instead')
    def predict_multistep(self, X: np.ndarray) -> np.ndarray:
        """Perform a multi-step prediction for the first state of each episode.

        This function takes the first ``min_samples_`` states of the input,
        along with all of its inputs, and predicts the next ``X.shape[0]``
        states of the system. This action is performed on a per-episode basis.
        The state features of ``X`` (other than the first ``min_samples_``
        features) are not used at all.

        If prediction fails numerically, missing predictions are filled with
        ``np.nan``.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Predicted data matrix.

        Raises
        ------
        ValueError
            If an episode is shorter than ``min_samples_``.

        Warning
        -------
        Deprecated in favour of
        :func:`pykoop.KoopmanPipeline.predict_trajectory`.

        """
        sklearn.utils.validation.check_is_fitted(self, 'regressor_fit_')
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Split episodes
        episodes = split_episodes(X, episode_feature=self.episode_feature_)
        # Loop over episodes
        predictions = []
        for (i, X_i) in episodes:
            # Check length of episode.
            if X_i.shape[0] < self.min_samples_:
                raise ValueError(f'Episode {i} has {X_i.shape[0]} samples but '
                                 f'`min_samples_`={self.min_samples_} samples '
                                 'are required.')
            # Index where prediction blows up (if it does)
            crash_index = None
            # Extract initial state and input
            x0 = X_i[:self.min_samples_, :self.n_states_in_]
            u = X_i[:, self.n_states_in_:]
            # Create array to hold predicted states
            X_pred_i = np.zeros((X_i.shape[0], self.n_states_in_))
            # Set the initial condition
            X_pred_i[:self.min_samples_, :] = x0
            # Predict all time steps
            for k in range(self.min_samples_, X_i.shape[0]):
                # Stack episode feature, previous predictions, and input
                X_ik = combine_episodes(
                    [(i,
                      np.hstack((
                          X_pred_i[(k - self.min_samples_):k, :],
                          X_i[(k - self.min_samples_):k, self.n_states_in_:],
                      )))],
                    episode_feature=self.episode_feature_)
                # Predict next step
                try:
                    X_pred_ik = self.predict(X_ik)[[-1], :]
                except ValueError:
                    crash_index = k
                    break
                # Extract data matrix from prediction
                X_pred_i[[k], :] = split_episodes(
                    X_pred_ik, episode_feature=self.episode_feature_)[0][1]
            if crash_index is not None:
                X_pred_i[crash_index:, :] = np.nan
            predictions.append((i, X_pred_i))
        # Combine episodes
        X_p = combine_episodes(predictions,
                               episode_feature=self.episode_feature_)
        return X_p

    def predict_trajectory(
        self,
        X0_or_X: np.ndarray,
        U: np.ndarray = None,
        relift_state: bool = True,
        return_lifted: bool = False,
        return_input: bool = False,
        episode_feature: bool = None,
    ) -> np.ndarray:
        """Predict state trajectory given input for each episode.

        Parameters
        ----------
        X0_or_X : np.ndarray
            Initial state if ``U`` is specified. If ``U`` is ``None``, then
            treated as the initial state and full input in one matrix, where
            the remaining states are ignored.
        U : np.ndarray
            Input. Length of prediction is governed by length of input. If
            ``None``, input is taken from last features of ``X0_or_X``.
        relift_state : bool
            If true, retract and re-lift state between prediction steps
            (default). Otherwise, only retract the state after all predictions
            are made. Correspond to the local and global error definitions of
            [MAM22]_.
        return_lifted : bool
            If true, return the lifted state. If false, return the original
            state (default).
        return_input : bool
            If true, return the input as well as the state. If false, return
            only the original state (default).
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.

        Returns
        -------
        np.ndarray
            Predicted state. If ``return_input``, input is appended to the
            array. If ``return_lifted``, the predicted state (and input) are
            returned in the lifted space.

        Raises
        ------
        ValueError
            If an episode is shorter than ``min_samples_``.

        Examples
        --------
        Predict trajectory with one argument

        >>> kp = pykoop.KoopmanPipeline(
        ...     lifting_functions=[
        ...         ('pl', pykoop.PolynomialLiftingFn(order=2)),
        ...     ],
        ...     regressor=pykoop.Edmd(),
        ... )
        >>> kp.fit(X_msd, n_inputs=1, episode_feature=True)
        KoopmanPipeline(lifting_functions=[('pl',
        PolynomialLiftingFn(order=2))], regressor=Edmd())
        >>> X_pred = kp.predict_trajectory(X_msd)

        Predict trajectory with two arguments
        >>> x0 = pykoop.extract_initial_conditions(
        ...     X_msd,
        ...     min_samples=kp.min_samples_,
        ...     n_inputs=1,
        ...     episode_feature=True,
        ... )
        >>> u = pykoop.extract_input(
        ...     X_msd,
        ...     n_inputs=1,
        ...     episode_feature=True,
        ... )
        >>> X_pred = kp.predict_trajectory(x0, u)
        """
        # Check fit
        sklearn.utils.validation.check_is_fitted(self, 'regressor_fit_')
        # Set episode feature if unspecified
        if episode_feature is None:
            episode_feature = self.episode_feature_
        # Get Koopman ``A`` and ``B`` matrices
        koop_mat = self.regressor_.coef_.T
        A = koop_mat[:, :koop_mat.shape[0]]
        B = koop_mat[:, koop_mat.shape[0]:]
        # Split episodes
        episodes = self._split_state_input_episodes(
            X0_or_X,
            U,
            episode_feature=episode_feature,
        )
        # Predict for each episode
        predictions: List[Tuple[float, np.ndarray]] = []
        for (i, X0_i, U_i) in episodes:
            crash_index = None
            # Iterate over episode and make predictions
            if relift_state:
                # Number of steps in episode
                n_steps_i = U_i.shape[0]
                # Initial conditions
                X_i = np.zeros((n_steps_i, self.n_states_in_))
                X_i[:self.min_samples_, :] = X0_i
                for k in range(self.min_samples_, n_steps_i):
                    try:
                        # Lift state and input
                        window = np.s_[(k - self.min_samples_):k]
                        Theta_ikm1 = self.lift_state(
                            X_i[window, :],
                            episode_feature=False,
                        )
                        Upsilon_ikm1 = self.lift_input(
                            np.hstack((X_i[window, :], U_i[window, :])),
                            episode_feature=False,
                        )
                        # Predict
                        Theta_ik = Theta_ikm1 @ A.T + Upsilon_ikm1 @ B.T
                        # Retract. If more than one sample is returned by
                        # ``retract_state``, take only the last one. This will
                        # happen if there's a delay lifting function.
                        X_i[[k], :] = self.retract_state(
                            Theta_ik,
                            episode_feature=False,
                        )[[-1], :]
                    except ValueError as ve:
                        if (np.all(np.isfinite(Theta_ikm1))
                                and np.all(np.isfinite(X_i))
                                and np.all(np.isfinite(U_i))
                                and np.all(np.isfinite(Upsilon_ikm1))
                                and np.all(np.isfinite(Theta_ik))):
                            raise ve
                        else:
                            crash_index = k - 1
                            X_i[crash_index:, :] = 0
                            break
                Theta_i = self.lift_state(X_i, episode_feature=False)
                Upsilon_i = self.lift_input(
                    np.hstack((X_i, U_i)),
                    episode_feature=False,
                )
            else:
                # Number of steps in episode
                n_steps_i = U_i.shape[0] - self.min_samples_ + 1
                # Initial conditions
                Theta_i = np.zeros((n_steps_i, self.n_states_out_))
                Upsilon_i = np.zeros((n_steps_i, self.n_inputs_out_))
                X_i = np.zeros((U_i.shape[0], self.n_states_in_))
                Theta_i[[0], :] = self.lift_state(X0_i, episode_feature=False)
                X_i[:self.min_samples_, :] = X0_i
                for k in range(1, n_steps_i + 1):
                    try:
                        # Lift input. We need to use the retracted state here
                        # because some lifting functions are state-dependent.
                        # ``Theta_i`` should not have any lifting and
                        # retracting happening, but for ``Upsilon_i``, it is
                        # unavoidable.
                        window_km1 = np.s_[(k - 1):(k + self.min_samples_ - 1)]
                        Upsilon_i[[k - 1], :] = self.lift_input(
                            np.hstack((
                                X_i[window_km1, :],
                                U_i[window_km1, :],
                            )),
                            episode_feature=False,
                        )
                        if k < n_steps_i:
                            # Predict next lifted state
                            Theta_i[[k], :] = (Theta_i[[k - 1], :] @ A.T
                                               + Upsilon_i[[k - 1], :] @ B.T)
                            # Retract and store state. ``k`` is index of
                            # ``Theta_i``, which is shorter than ``X_i`` when
                            # there are delays.
                            #
                            # If ``min_samples_=3``, ``Theta_i`` is two entries
                            # shorter than ``X_i``. The first entry of
                            # ``Theta_i`` occurs at the same "time" as the
                            # third entry of ``X_i``. Thus ``Theta_i[k]``
                            # corresponds to
                            # ``X_i[k + self.min_samples_ - 1]``.
                            #
                            # Let ``self.min_samples_=3``. An illustration:
                            #
                            #            X_i[0:3]
                            #            vvvvv
                            #     X_i: [ | | | . . . . . . . ]
                            # Theta_i:     [ | . . . . . . . ]
                            #                ^
                            #                Theta_i[0]
                            X_ik = self.retract_state(
                                Theta_i[[k], :],
                                episode_feature=False,
                            )
                            X_i[[k + self.min_samples_ - 1], :] = X_ik[[-1], :]
                    except ValueError as ve:
                        if (np.all(np.isfinite(X_ikm1))
                                and np.all(np.isfinite(Theta_i))
                                and np.all(np.isfinite(Upsilon_i))
                                and np.all(np.isfinite(U_i))):
                            raise ve
                        else:
                            crash_index = k - 1
                            Theta_i[crash_index:, :] = 0
                            Upsilon_i[crash_index:, :] = 0
                            break
            # If prediction crashed, set remaining entries to NaN
            if crash_index is not None:
                log.warning(f'Prediction diverged at index {crash_index}. '
                            'Remaining entries set to `NaN`.')
                # Don't set ``U_i`` to NaN since it's a known input
                X_i[crash_index:, :] = np.nan
                Theta_i[crash_index:, :] = np.nan
                Upsilon_i[crash_index:, :] = np.nan
            # Choose what to return
            if return_lifted:
                if return_input:
                    predictions.append((i, np.hstack((Theta_i, Upsilon_i))))
                else:
                    predictions.append((i, Theta_i))
            else:
                if return_input:
                    predictions.append((i, np.hstack((X_i, U_i))))
                else:
                    predictions.append((i, X_i))
        # Combine episodes
        combined_episodes = combine_episodes(
            predictions,
            episode_feature=episode_feature,
        )
        return combined_episodes

    def plot_predicted_trajectory(
        self,
        X0_or_X: np.ndarray,
        U: np.ndarray = None,
        relift_state: bool = True,
        plot_lifted: bool = False,
        plot_input: bool = False,
        episode_feature: bool = None,
        plot_ground_truth: bool = True,
        episode_style: str = None,
        subplots_kw: Dict[str, Any] = None,
        plot_kw: Dict[str, Any] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot predicted trajectory.

        Parameters
        ----------
        X0_or_X : np.ndarray
            Initial state if ``U`` is specified. If ``U`` is ``None``, then
            treated as the ground truth trajectory from which the initial state
            and full input are extracted.
        U : np.ndarray
            Input. Length of prediction is governed by length of input. If
            ``None``, input is taken from last features of ``X0_or_X``.
        relift_state : bool
            If true, retract and re-lift state between prediction steps
            (default). Otherwise, only retract the state after all predictions
            are made. Correspond to the local and global error definitions of
            [MAM22]_.
        plot_lifted : bool
            If true, plot the lifted state. If false, plot the original
            state (default).
        plot_input : bool
            If true, plot the input as well as the state. If false, plot
            only the original state (default).
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.
            If ``None``, ``self.episode_feature_`` is used.
        plot_ground_truth : bool
            Plot contents of ``X0_or_X`` as ground truth if ``U`` is ``None``.
            Ignored if ``U`` is not ``None``.
        episode_style : str
            If ``'columns'``, each episode is a column (default). If
            ``'overlay'``, states from each episode are plotted overtop of each
            other in different colors.
        subplots_kw : Dict[str, Any] = None,
            Keyword arguments for :func:`plt.subplots()`.
        plot_kw : Dict[str, Any] = None,
            Keyword arguments for Matplotlib :func:`plt.Axes.plot()`.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Matplotlib :class:`plt.Figure` object and two-dimensional array of
            :class:`plt.Axes` objects.
        """
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self, 'regressor_fit_')
        # Set episode feature if unspecified
        if episode_feature is None:
            episode_feature = self.episode_feature_
        # Predict trajectory
        Xp = self.predict_trajectory(
            X0_or_X,
            U,
            relift_state=relift_state,
            return_lifted=plot_lifted,
            return_input=plot_input,
            episode_feature=episode_feature,
        )
        # Split episodes. `eps`` only contains inputs if they are to be
        # plotted. ``eps_gt`` truth always contains inputs, even if they are
        # not plotted.
        eps = split_episodes(Xp, episode_feature=episode_feature)
        if plot_ground_truth and (U is None):
            if plot_lifted:
                X_gt = self.lift(X0_or_X, episode_feature=episode_feature)
            else:
                X_gt = X0_or_X
            eps_gt = split_episodes(X_gt, episode_feature=episode_feature)
        else:
            eps_gt = None
        # Figure out dimensions.
        if plot_lifted:
            if plot_input:
                n_row = self.n_states_out_ + self.n_inputs_out_
            else:
                n_row = self.n_states_out_
        else:
            if plot_input:
                n_row = self.n_states_in_ + self.n_inputs_in_
            else:
                n_row = self.n_states_in_
        n_eps = len(eps)
        n_col = 1 if episode_style == 'overlay' else n_eps
        n_states = self.n_states_out_ if plot_lifted else self.n_states_in_
        # Create figure
        subplots_args = {} if subplots_kw is None else subplots_kw
        subplots_args.update({
            'squeeze': False,
            'constrained_layout': True,
            'sharex': 'col',
            'sharey': 'row',
        })
        fig, ax = plt.subplots(n_row, n_col, **subplots_args)
        # Set up plot arguments
        plot_args = {} if plot_kw is None else plot_kw
        plot_args.pop('label', None)
        plot_args.pop('color', None)
        plot_args.pop('linestyle', None)
        # Plot results
        for row in range(n_row):
            for ep in range(n_eps):
                if episode_style == 'overlay':
                    line_pred = ax[row, 0].plot(
                        eps[ep][1][:, row],
                        label=f'Ep. {int(eps[ep][0])} prediction',
                        **plot_args,
                    )
                    if eps_gt is not None and row < n_states:
                        ax[row, 0].plot(
                            eps_gt[ep][1][:, row],
                            label=f'Ep. {int(eps[ep][0])} ground truth',
                            linestyle='--',
                            color=line_pred[0].get_color(),
                            **plot_args,
                        )
                else:
                    line_pred = ax[row, ep].plot(
                        eps[ep][1][:, row],
                        label=f'Prediction',
                        **plot_args,
                    )
                    if eps_gt is not None and row < n_states:
                        ax[row, ep].plot(
                            eps_gt[ep][1][:, row],
                            label=f'Ground truth',
                            linestyle='--',
                            **plot_args,
                        )
        # Set y labels
        if plot_lifted:
            names = self.get_feature_names_out(
                symbols_only=True,
                format='latex',
                episode_feature=False,
            )
        else:
            names = self.get_feature_names_in(
                format='latex',
                episode_feature=False,
            )
        for row in range(n_row):
            ax[row, 0].set_ylabel(f'${names[row]}$')
        for col in range(n_col):
            if episode_style != 'overlay':
                ax[0, col].set_title(f'Ep. {int(eps[col][0])}')
            ax[-1, col].set_xlabel('$k$')
        # Set legend
        fig.legend(
            *ax[0, 0].get_legend_handles_labels(),
            loc='upper right',
            bbox_to_anchor=(1, 1) if episode_style == 'overlay' else (1, 0.95),
        )
        return fig, ax

    @staticmethod
    def make_scorer(
        n_steps: int = None,
        discount_factor: float = 1,
        regression_metric: str = 'neg_mean_squared_error',
        multistep: bool = True,
        relift_state: bool = True,
    ) -> Callable[['KoopmanPipeline', np.ndarray, Optional[np.ndarray]],
                  float]:
        """Make a Koopman pipeline scorer.

        A ``scikit-learn`` scorer accepts the parameters ``(estimator, X, y)``
        and returns a float representing the prediction quality of
        ``estimator`` on ``X`` with reference to ``y``. Uses existing
        ``scikit-learn`` regression metrics [#sc]_. Higher numbers are better.
        Metrics corresponding to losses are negated.

        Technically, the scorer will predict the entire episode, regardless of
        how ``n_steps`` is set. It will then assign a zero weight to all errors
        beyond ``n_steps``.

        Parameters
        ----------
        n_steps : int
            Number of steps ahead to predict. If ``None`` or longer than the
            episode, will score the entire episode.

        discount_factor : float
            Discount factor used to weight the error timeseries. Should be
            positive, with magnitude 1 or slightly less. The error at each
            timestep is weighted by ``discount_factor**k``, where ``k`` is the
            timestep.

        regression_metric : str
            Regression metric to use. One of

            - ``'explained_variance'``,
            - ``'neg_mean_absolute_error'``,
            - ``'neg_mean_squared_error'``,
            - ``'neg_mean_squared_log_error'``,
            - ``'neg_median_absolute_error'``,
            - ``'r2'``, or
            - ``'neg_mean_absolute_percentage_error'``,

            which are existing ``scikit-learn`` regression metrics [#sc]_.

        multistep : bool
            If true, predict using :func:`predict_trajectory`. Otherwise,
            predict using :func:`predict` (one-step-ahead prediction).
            Multistep prediction is highly recommended unless debugging. If
            one-step-ahead prediciton is used, `n_steps` and `discount_factor`
            are ignored.

        relift_state : bool
            If true, retract and re-lift state between prediction steps
            (default). Otherwise, only retract the state after all predictions
            are made. Correspond to the local and global error definitions of
            [MAM22]_. Ignored if ``multistep`` is false.

        Returns
        -------
        Callable[[KoopmanPipeline, np.ndarray, Optional[np.ndarray]], float]
            Scorer compatible with ``scikit-learn``.

        Raises
        ------
        ValueError
            If ``discount_factor`` is negative or greater than one.

        References
        ----------
        .. [#sc] https://scikit-learn.org/stable/modules/model_evaluation.html
        """

        def koopman_pipeline_scorer(
            estimator: KoopmanPipeline,
            X: np.ndarray,
            y: np.ndarray = None,
        ) -> float:
            # Shift episodes
            X_unshifted, X_shifted = shift_episodes(
                X,
                n_inputs=estimator.n_inputs_in_,
                episode_feature=estimator.episode_feature_,
            )
            # Predict
            if multistep:
                # Get initial conditions for each episode
                x0 = extract_initial_conditions(
                    X_unshifted,
                    min_samples=estimator.min_samples_,
                    n_inputs=estimator.n_inputs_in_,
                    episode_feature=estimator.episode_feature_,
                )
                # Get inputs for each episode
                u = extract_input(
                    X_unshifted,
                    n_inputs=estimator.n_inputs_in_,
                    episode_feature=estimator.episode_feature_,
                )
                # Predict state for each episode
                X_predicted = estimator.predict_trajectory(
                    x0,
                    u,
                    relift_state=relift_state,
                )
                # Score prediction
                score = score_trajectory(
                    X_predicted,
                    X_shifted,
                    n_steps=n_steps,
                    discount_factor=discount_factor,
                    regression_metric=regression_metric,
                    min_samples=estimator.min_samples_,
                    episode_feature=estimator.episode_feature_,
                )
            else:
                # Warn about ignored non-default arguments
                if not relift_state:
                    log.info('Ignoring `relift_state` since `multistep` is '
                             'false.')
                if n_steps is not None:
                    log.info('Ignoring `n_steps` since `multistep` is false.')
                if discount_factor != 1:
                    log.info('Ignoring `discount_factor` since `multistep` is '
                             'false.')
                # Perform single-step prediction
                X_predicted = estimator.predict(X_unshifted)
                # Score prediction
                score = score_trajectory(
                    X_predicted,
                    X_shifted,
                    n_steps=None,
                    discount_factor=1,
                    regression_metric=regression_metric,
                    min_samples=estimator.min_samples_,
                    episode_feature=estimator.episode_feature_,
                )
            return score

        return koopman_pipeline_scorer

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        # noqa: D102
        return self._get_params('lifting_functions', deep=deep)

    def set_params(self, **kwargs) -> 'KoopmanPipeline':
        # noqa: D102
        self._set_params('lifting_functions', **kwargs)
        return self

    def _transform_feature_names(
        self,
        feature_names: np.ndarray,
        format: str = None,
    ) -> np.ndarray:
        """Transform feature names.

        Parameters
        ----------
        feature_names : np.ndarray
            Feature names.
        format : str
            Feature name formatting method. Possible values are ``'plaintext'``
            (default if ``None``) or ``'latex'``.

        Returns
        -------
        np.ndarray
            Transformed feature names.
        """
        names_out = feature_names
        for _, lf in self.lifting_functions_:
            names_out = lf._transform_feature_names(names_out, format)
        return names_out

    def _split_state_input_episodes(
        self,
        X0_or_X: np.ndarray,
        U: np.ndarray = None,
        episode_feature: bool = False,
    ) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """Break initial conditions and inputs into episodes.

        Parameters
        ----------
        X0_or_X : np.ndarray
            Initial state if ``U`` is specified. If ``U`` is ``None``, then
            treated as the initial state and full input in one matrix, where
            the remaining states are ignored.
        U : np.ndarray
            Input. Length of prediction is governed by length of input. If
            ``None``, input is taken from last features of ``X0_or_X``.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.

        Returns
        -------
        List[Tuple[float, np.ndarray, np.ndarray]]
            List of episode indices, initial conditions, and inputs.

        Raises
        ------
        ValueError
            If input dimensions are incorrect.
        """
        if U is None:
            if X0_or_X.shape[1] != self.n_features_in_:
                raise ValueError('Invalid dimensions for ``X0_or_X``. If '
                                 '``U=None``, ``X0_or_X`` must contain '
                                 'states and inputs.')
            ep_X = split_episodes(X0_or_X, episode_feature=episode_feature)
            episodes = [(
                ex[0],
                ex[1][:self.min_samples_, :self.n_states_in_],
                ex[1][:, self.n_states_in_:],
            ) for ex in ep_X]
        else:
            ep = 1 if episode_feature else 0
            if X0_or_X.shape[1] != self.n_states_in_ + ep:
                raise ValueError('Invalid dimensions for ``X0_or_X``. If '
                                 '``U`` is specified, ``X0_or_X`` must '
                                 'contain only states.')
            if U.shape[1] != self.n_inputs_in_ + ep:
                raise ValueError('Invalid dimensions for ``U``. If ``U`` is '
                                 'specified, it must contain only inputs.')
            ep_X0 = split_episodes(X0_or_X, episode_feature=episode_feature)
            ep_U = split_episodes(U, episode_feature=episode_feature)
            episodes = [(ex[0], ex[1], eu[1]) for (ex, eu) in zip(ep_X0, ep_U)]
        # Check length of episode.
        for (i, X0_i, U_i) in episodes:
            if X0_i.shape[0] != self.min_samples_:
                raise ValueError(f'Initial condition in episode {i} has '
                                 f'{X0_i.shape[0]} samples but `min_samples_`='
                                 f'{self.min_samples_} samples are required.')
            if U_i.shape[0] < self.min_samples_:
                raise ValueError(f'Input in episode {i} has {U_i.shape[0]} '
                                 'samples but at least `min_samples_`='
                                 f'{self.min_samples_} samples are required.')
        return episodes

    def plot_bode(
        self,
        t_step: float,
        f_min: float = 0,
        f_max: float = None,
        n_points: int = 1000,
        decibels: bool = True,
        subplots_kw: Dict[str, Any] = None,
        plot_kw: Dict[str, Any] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot frequency response of Koopman system.

        Parameters
        ----------
        t_step : float
            Sampling timestep.
        f_min : float
            Minimum frequency to plot.
        f_max : float
            Maximum frequency to plot.
        n_points : int
            Number of frequecy points to plot.
        decibels : bool
            Plot gain in dB (default is true).
        subplots_kw : Dict[str, Any] = None,
            Keyword arguments for :func:`plt.subplots()`.
        plot_kw : Dict[str, Any] = None,
            Keyword arguments for Matplotlib :func:`plt.Axes.plot()`.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Matplotlib :class:`plt.Figure` object and two-dimensional array of
            :class:`plt.Axes` objects.

        Raises
        ------
        ValueError
            If ``f_min`` is less than zero or ``f_max`` is greater than the
            Nyquist frequency.
        """
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self, 'regressor_fit_')
        return self.regressor_.plot_bode(
            t_step,
            f_min,
            f_max,
            n_points,
            decibels,
            subplots_kw,
            plot_kw,
        )

    def plot_eigenvalues(
        self,
        unit_circle: bool = True,
        figure_kw: Dict[str, Any] = None,
        subplot_kw: Dict[str, Any] = None,
        plot_kw: Dict[str, Any] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot eigenvalues of Koopman ``A`` matrix.

        Parameters
        ----------
        figure_kw : Dict[str, Any] = None,
            Keyword arguments for :func:`plt.figure()`.
        subplot_kw : Dict[str, Any] = None,
            Keyword arguments for :func:`plt.subplot()`.
        plot_kw : Dict[str, Any] = None,
            Keyword arguments for Matplotlib :func:`plt.Axes.plot()`.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Matplotlib :class:`plt.Figure` object and two-dimensional array of
            :class:`plt.Axes` objects.
        """
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self, 'regressor_fit_')
        return self.regressor_.plot_eigenvalues(
            unit_circle,
            figure_kw,
            subplot_kw,
            plot_kw,
        )

    def plot_koopman_matrix(
        self,
        subplots_kw: Dict[str, Any] = None,
        plot_kw: Dict[str, Any] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot heatmap of Koopman matrices.

        Parameters
        ----------
        subplots_kw : Dict[str, Any] = None,
            Keyword arguments for :func:`plt.subplots()`.
        plot_kw : Dict[str, Any] = None,
            Keyword arguments for Matplotlib :func:`plt.Axes.plot()`.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Matplotlib :class:`plt.Figure` object and two-dimensional array of
            :class:`plt.Axes` objects.
        """
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self, 'regressor_fit_')
        return self.regressor_.plot_koopman_matrix(subplots_kw, plot_kw)

    def plot_svd(
        self,
        subplots_kw: Dict[str, Any] = None,
        plot_kw: Dict[str, Any] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot singular values of Koopman matrices.

        Parameters
        ----------
        subplots_kw : Dict[str, Any] = None,
            Keyword arguments for :func:`plt.subplots()`.
        plot_kw : Dict[str, Any] = None,
            Keyword arguments for Matplotlib :func:`plt.Axes.plot()`.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Matplotlib :class:`plt.Figure` object and two-dimensional array of
            :class:`plt.Axes` objects.
        """
        # Ensure fit has been done
        sklearn.utils.validation.check_is_fitted(self, 'regressor_fit_')
        return self.regressor_.plot_svd(subplots_kw, plot_kw)


def score_trajectory(
    X_predicted: np.ndarray,
    X_expected: np.ndarray,
    n_steps: int = None,
    discount_factor: float = 1,
    regression_metric: str = 'neg_mean_squared_error',
    min_samples: int = 1,
    episode_feature: bool = False,
) -> float:
    """Score a predicted data matrix compared to an expected data matrix.

    Parameters
    ----------
    X_predicted : np.ndarray
        Predicted state data matrix.

    X_expected : np.ndarray
        Expected state data matrix.

    n_steps : int
        Number of steps ahead to predict. If ``None`` or longer than the
        episode, will score the entire episode.

    discount_factor : float
        Discount factor used to weight the error timeseries. Should be
        positive, with magnitude 1 or slightly less. The error at each
        timestep is weighted by ``discount_factor**k``, where ``k`` is the
        timestep.

    regression_metric : str
        Regression metric to use. One of

        - ``'explained_variance'``,
        - ``'neg_mean_absolute_error'``,
        - ``'neg_mean_squared_error'``,
        - ``'neg_mean_squared_log_error'``,
        - ``'neg_median_absolute_error'``,
        - ``'r2'``, or
        - ``'neg_mean_absolute_percentage_error'``,

        which are existing ``scikit-learn`` regression metrics [#sc]_.

    min_samples : int
        Number of samples in initial condition.

    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    float
        Score (greater is better).

    References
    ----------
    .. [#sc] https://scikit-learn.org/stable/modules/model_evaluation.html
    """
    # Valid ``regression_metric`` values:
    regression_metrics = {
        'explained_variance':
        sklearn.metrics.explained_variance_score,
        'r2':
        sklearn.metrics.r2_score,
        'neg_mean_absolute_error':
        sklearn.metrics.mean_absolute_error,
        'neg_mean_squared_error':
        sklearn.metrics.mean_squared_error,
        'neg_mean_squared_log_error':
        sklearn.metrics.mean_squared_log_error,
        'neg_median_absolute_error':
        sklearn.metrics.median_absolute_error,
        'neg_mean_absolute_percentage_error':
        sklearn.metrics.mean_absolute_percentage_error,
    }
    # Scores that do not need inversion
    greater_is_better = ['explained_variance', 'r2']
    # Strip episode feature and initial conditions
    X_expected = strip_initial_conditions(
        X_expected,
        min_samples=min_samples,
        episode_feature=episode_feature,
    )
    X_predicted = strip_initial_conditions(
        X_predicted,
        min_samples=min_samples,
        episode_feature=episode_feature,
    )
    # Compute weights
    weights = _weights_from_data_matrix(
        X_expected,
        n_steps=n_steps,
        discount_factor=discount_factor,
        episode_feature=episode_feature,
    )
    # Strip episode feature if present
    if episode_feature:
        X_expected = X_expected[:, 1:]
        X_predicted = X_predicted[:, 1:]
    # Calculate score
    score = regression_metrics[regression_metric](
        X_expected,
        X_predicted,
        sample_weight=weights,
        multioutput='uniform_average',
    )
    # Invert losses
    if regression_metric not in greater_is_better:
        score *= -1
    return score


def extract_initial_conditions(
    X: np.ndarray,
    min_samples: int = 1,
    n_inputs: int = 0,
    episode_feature: bool = False,
) -> np.ndarray:
    """Extract initial conditions from each episode.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    min_samples : int
        Number of samples in initial condition.
    n_inputs : int
        Number of input features at the end of ``X``.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    np.ndarray
        Initial conditions from each episode.
    """
    episodes = split_episodes(X, episode_feature=episode_feature)
    # Strip each episode
    initial_conditions = []
    for (i, X_i) in episodes:
        if n_inputs == 0:
            initial_condition = X_i[:min_samples, :]
        else:
            initial_condition = X_i[:min_samples, :-n_inputs]
        initial_conditions.append((i, initial_condition))
    # Concatenate the initial conditions
    X0 = combine_episodes(initial_conditions, episode_feature=episode_feature)
    return X0


def extract_input(
    X: np.ndarray,
    n_inputs: int = 0,
    episode_feature: bool = False,
) -> np.ndarray:
    """Extract input from a data matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    n_inputs : int
        Number of input features at the end of ``X``.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    np.ndarray
        Input extracted from data matrix.
    """
    episodes = split_episodes(X, episode_feature=episode_feature)
    # Strip each episode
    inputs = []
    for (i, X_i) in episodes:
        if n_inputs == 0:
            input_ = np.zeros((X_i.shape[0], 0))
        else:
            n_states = X_i.shape[1] - n_inputs
            input_ = X_i[:, n_states:]
        inputs.append((i, input_))
    # Concatenate the inputs
    u = combine_episodes(inputs, episode_feature=episode_feature)
    return u


def strip_initial_conditions(X: np.ndarray,
                             min_samples: int = 1,
                             episode_feature: bool = False) -> np.ndarray:
    """Strip initial conditions from each episode.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    min_samples : int
        Number of samples in initial condition.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    np.ndarray
        Data matrix with initial conditions removed.
    """
    episodes = split_episodes(X, episode_feature=episode_feature)
    # Strip each episode
    stripped_episodes = []
    for (i, X_i) in episodes:
        stripped_episode = X_i[min_samples:, :]
        stripped_episodes.append((i, stripped_episode))
    # Concatenate the stripped episodes
    Xs = combine_episodes(stripped_episodes, episode_feature=episode_feature)
    return Xs


def shift_episodes(
        X: np.ndarray,
        n_inputs: int = 0,
        episode_feature: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Shift episodes and truncate shifted inputs.

    The Koopman matrix ``K`` approximately satisfies::

        Theta_+ = Psi @ K.T

    where ``Psi`` contains the unshifted states and inputs, and ``Theta_+``
    contains the shifted states.

    The regressors used in :class:`KoopmanPipeline` expect ``Psi`` as their
    ``X`` and ``Theta_+`` as their ``y``. This function breaks its input (also
    named ``X``) into ``Psi`` and ``Theta_+`` for use with these regressors.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    n_inputs : int
        Number of input features at the end of ``X``.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple whose first element is the unshifted array and whose second
        element is the shifted array with its inputs truncated. Both arrays
        have the same number of samples. Their episode features are stripped if
        present.
    """
    # Split episodes
    episodes = split_episodes(X, episode_feature=episode_feature)
    # Shift each episode
    unshifted_episodes = []
    shifted_episodes = []
    for i, X_i in episodes:
        # Get unshifted episode
        X_i_unshifted = X_i[:-1, :]
        # Get shifted episode. Strip input if present.
        if n_inputs == 0:
            X_i_shifted = X_i[1:, :]
        else:
            X_i_shifted = X_i[1:, :-n_inputs]
        # Append to episode list
        unshifted_episodes.append((i, X_i_unshifted))
        shifted_episodes.append((i, X_i_shifted))
    # Recombine and return
    X_unshifted = combine_episodes(unshifted_episodes,
                                   episode_feature=episode_feature)
    X_shifted = combine_episodes(shifted_episodes,
                                 episode_feature=episode_feature)
    return (X_unshifted, X_shifted)


def split_episodes(
        X: np.ndarray,
        episode_feature: bool = False) -> List[Tuple[float, np.ndarray]]:
    """Split a data matrix into episodes.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    List[Tuple[float, np.ndarray]]
        List of episode tuples. The first element of each tuple contains the
        episode index. The second element contains the episode data.
    """
    # Extract episode feature
    if episode_feature:
        X_ep = X[:, 0]
        X = X[:, 1:]
    else:
        X_ep = np.zeros((X.shape[0], ))
    # Split X into list of episodes. Each episode is a tuple containing
    # its index and its associated data matrix.
    episodes = []
    # ``pandas.unique`` is faster than ``np.unique`` and preserves order.
    for i in pandas.unique(X_ep):
        episodes.append((i, X[X_ep == i, :]))
    # Return list of episodes
    return episodes


def combine_episodes(episodes: List[Tuple[float, np.ndarray]],
                     episode_feature: bool = False) -> np.ndarray:
    """Combine episodes into a data matrix.

    Parameters
    ----------
    episodes : List[Tuple[float, np.ndarray]]
        List of episode tuples. The first element of each tuple contains the
        episode index. The second element contains the episode data.
    episode_feature : bool
        True if first feature of output should indicate which episode a
        timestep is from.

    Returns
    -------
    np.ndarray
        Combined data matrix.
    """
    combined_episodes = []
    for (i, X) in episodes:
        if episode_feature:
            combined_episodes.append(
                np.hstack((i * np.ones((X.shape[0], 1)), X)))
        else:
            combined_episodes.append(X)
    # Concatenate the combined episodes
    Xc = np.vstack(combined_episodes)
    return Xc


def _weights_from_data_matrix(
    X: np.ndarray,
    n_steps: int = None,
    discount_factor: float = 1,
    episode_feature: bool = False,
) -> np.ndarray:
    """Create an array of scoring weights from a data matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix
    n_steps : int
        Number of steps ahead to predict. If ``None`` or longer than the
        episode, will weight the entire episode.
    discount_factor : float
        Discount factor used to weight the error timeseries. Should be
        positive, with magnitude 1 or slightly less. The error at each
        timestep is weighted by ``discount_factor**k``, where ``k`` is the
        timestep.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    np.ndarray
        Array of weights use for scoring.

    Raises
    ------
    ValueError
        If ``discount_factor`` is not in [0, 1].
    """
    # Check discount factor
    if (discount_factor < 0) or (discount_factor > 1):
        raise ValueError('`discount_factor` must be positive and less '
                         'than one.')
    weights_list = []
    episodes = split_episodes(X, episode_feature=episode_feature)
    for i, X_i in episodes:
        # Compute number of nonzero weights needed
        n_samples_i = X_i.shape[0]
        if n_steps is None:
            n_nonzero_weights_i = n_samples_i
        else:
            n_nonzero_weights_i = min(n_steps, n_samples_i)
        # Compute weights. Weights after ``n_steps`` are 0.
        weights_i = np.array(
            [discount_factor**k for k in range(n_nonzero_weights_i)]
            + [0] * (n_samples_i - n_nonzero_weights_i))
        weights_list.append(weights_i)
    weights = np.concatenate(weights_list)
    return weights


def _extract_feature_names(
        X: Union[np.ndarray, pandas.DataFrame]) -> Optional[np.ndarray]:
    """Extract feature names from input array.

    Parameters
    ----------
    X : Union[np.ndarray, pandas.DataFrame]
        Input array.

    Returns
    -------
    Optional[np.ndarray]
        Feature names if present, ``None`` otherwise.

    Raises
    ------
    ValueError
        If feature names are not strings.
    """
    if isinstance(X, pandas.DataFrame):
        for name in X.columns:
            if not isinstance(name, str):
                log.warning(
                    'Feature names must all be strings. When ``scikit-learn`` '
                    'v1.2 comes out this will be upgraded to an exception.')
                return None
        return np.asarray(X.columns, dtype=object)
    else:
        return None
