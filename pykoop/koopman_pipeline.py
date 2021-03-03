import numpy as np
import sklearn.base


class KoopmanPipeline(sklearn.base.BaseEstimator):

    def __init__(self, delay, estimator):
        self.delay = delay
        self.estimator = estimator

    def fit(self, X, y=None, n_u=0, episode_indices=None):
        # Clone estimators
        self.delay_ = sklearn.base.clone(self.delay)
        self.estimator_ = sklearn.base.clone(self.estimator)
        # TODO Pre-processing
        # Delays
        self.delay_.fit(X, n_u=n_u)
        episodes = np.split(X, [] if episode_indices is None else
                            episode_indices)
        delayed_episodes = []
        # Delay episode
        for ep in episodes:
            delayed_ep = self.delay_.transform(ep)
            delayed_episodes.append(delayed_ep)
        Xd = np.vstack(delayed_episodes)
        # Keep track of the new episode boundaries
        delayed_episode_indices = [delayed_episodes[0].shape[0]]
        for ep in delayed_episodes[1:-1]:
            delayed_episode_indices.append(delayed_episode_indices[-1]
                                           + ep.shape[0])
        # TODO Lifting functions here
        # TODO Can do delays before OR after lifting
        Xt = Xd
        # Split into X and y
        transformed_episodes = np.split(Xt, [] if episode_indices is None else
                                        delayed_episode_indices)
        Xt_unshifted = []
        Xt_shifted = []
        for ep in transformed_episodes:
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
        Xd = self.delay_.transform(X)
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
        return Xp_reduced
