import numpy as np
import sklearn.base


class KoopmanPipeline(sklearn.base.BaseEstimator):

    def __init__(self, delay, estimator):
        self.delay = delay
        self.estimator = estimator

    def fit(self, X, y=None, n_u=0):
        # Clone estimators
        self.delay_ = sklearn.base.clone(self.delay)
        self.estimator_ = sklearn.base.clone(self.estimator)
        # TODO Pre-processing
        # Delays
        self.delay_.fit(X[:, 1:], n_u=n_u)
        episodes = []
        for i in np.unique(X[:, 0]):
            episodes.append((i, X[X[:, 0] == i, 1:]))
        delayed_episodes = []
        # Delay episode
        for (i, ep) in episodes:
            delayed_ep = self.delay_.transform(ep)
            delayed_episodes.append(np.hstack((
                i * np.ones((delayed_ep.shape[0], 1)),
                delayed_ep,
            )))
        Xd = np.vstack(delayed_episodes)
        # TODO Lifting functions here
        # TODO Can do delays before OR after lifting
        Xt = Xd
        # Split into X and y
        transformed_episodes = []
        for i in np.unique(Xt[:, 0]):
            transformed_episodes.append((i, Xt[Xt[:, 0] == i, 1:]))
        Xt_unshifted = []
        Xt_shifted = []
        for (_, ep) in transformed_episodes:
            Xt_unshifted.append(ep[:-1, :])
            if self.delay_.n_ud_ == 0:
                Xt_shifted.append(ep[1:, :])
            else:
                Xt_shifted.append(ep[1:, :-self.delay_.n_ud_])
        Xt_unshifted = np.vstack(Xt_unshifted)
        Xt_shifted = np.vstack(Xt_shifted)
        # Fit estimator
        self.estimator_.fit(Xt_unshifted, Xt_shifted)

    def predict(self, X):
        group = X[0, 0]
        Xd = self.delay_.transform(X[:, 1:])
        # Pad inputs with zeros so inverse_transform has same dimension as
        # fit(). TODO Is this necessary?
        Xdp = np.hstack((
            self.estimator_.predict(Xd),
            np.zeros((1, self.delay_.n_ud_))
        ))
        Xp = self.delay_.inverse_transform(Xdp)
        # Take only most recent time step. Strip off dummy inputs.
        if self.delay_.n_u_ == 0:
            Xp_reduced = Xp[[-1], :]
        else:
            Xp_reduced = Xp[[-1], :-self.delay_.n_u_]
        Xp_reduced_group = np.hstack((
            group * np.ones((Xp_reduced.shape[0], 1)),
            Xp_reduced
        ))
        return Xp_reduced_group
