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
        # Delays
        self.delay_.fit(X)
        if episode_indices is None:
            episode_indices = []
        episodes = np.split(X, episode_indices)
        delayed_episodes = []
        delayed_episode_indices = []
        for ep in episodes:
            # Delay episode
            delayed_ep = self.delay_.transform(ep)
            delayed_episodes.append(delayed_ep)
            # Keep track of the new episode boundaries
            if delayed_episode_indices:
                last_ep_idx = delayed_episode_indices[-1]
            else:
                last_ep_idx = 0
            delayed_episode_indices.append(last_ep_idx + delayed_ep.shape[0])
        Xd = np.vstack(delayed_episodes)
        # TODO Lifting functions here
        Xt = Xd
        # Split into X and y
        transformed_episodes = np.split(Xt, delayed_episode_indices)
        Xt_unshifted = []
        Xt_shifted = []
        for ep in transformed_episodes:
            Xt_unshifted.append(ep[:-1, :])
            Xt_shifted.append(ep[1:, :-n_u])
        Xt_unshifted = np.vstack(Xt_unshifted)
        Xt_shifted = np.vstack(Xt_shifted)
        # Fit estimator
        self.estimator_.fit(Xt_unshifted, Xt_shifted)

    def predict(self, X):
        pass
