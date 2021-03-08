import numpy as np
import sklearn.base
import sklearn.metrics


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
        # TODO HANDLE SPLITTING HERE?
        # TODO MOVE SPLIT AND DELAY TO HELPER FUNCTIONS?
        # TODO CREATE A TRANSFORMER?
        episodes = []
        for i in np.unique(X[:, 0]):
            episodes.append((i, X[X[:, 0] == i, 1:]))
        predictions = []
        for (i, ep) in episodes:
            Xd = self.delay_.transform(ep)
            # Pad inputs with zeros so inverse_transform has same dimension as
            # fit(). TODO Is this necessary?
            pred = self.estimator_.predict(Xd)
            Xdp = np.hstack((
                pred,
                np.zeros((pred.shape[0], self.delay_.n_ud_))
            ))
            Xp = self.delay_.inverse_transform(Xdp)
            # Take only most recent time step. Strip off dummy inputs.
            if self.delay_.n_u_ == 0:
                Xp_reduced = Xp[:, :]
            else:
                Xp_reduced = Xp[:, :-self.delay_.n_u_]
            Xp_reduced_group = np.hstack((
                i * np.ones((Xp_reduced.shape[0], 1)),
                Xp_reduced
            ))
            predictions.append(Xp_reduced_group)
        return np.vstack(predictions)

    def score_single_step(self, X, y=None):
        episodes = []
        for i in np.unique(X[:, 0]):
            episodes.append((i, X[X[:, 0] == i, 1:]))
        X_unshifted = []
        X_shifted = []
        for (i, ep) in episodes:
            X_unshifted.append(np.hstack((
                i * np.ones((ep.shape[0]-1, 1)),
                ep[:-1, :]
            )))
            if self.delay_.n_ud_ == 0:
                X_shifted.append(ep[1:, :])
            else:
                X_shifted.append(ep[1:, :-self.delay_.n_ud_])
        X_unshifted = np.vstack(X_unshifted)
        X_shifted = np.vstack(X_shifted)
        # Predict
        X_predicted = self.predict(X_unshifted)
        return sklearn.metrics.r2_score(X_shifted, X_predicted[:, 1:])

    def score(self, X, y=None):
        episodes = []
        for i in np.unique(X[:, 0]):
            episodes.append((i, X[X[:, 0] == i, 1:]))
        n_samp = self.delay_.n_samples_needed_
        X_validation = []
        X_predicted = []
        for (i, ep) in episodes:
            X_pred = np.empty(ep.shape)
            X_pred[:n_samp, :] = ep[:n_samp, :]
            for k in range(n_samp, X_pred.shape[0]):
                Xk = np.hstack((
                    i * np.ones((n_samp, 1)),
                    X_pred[(k-n_samp):k, :],
                ))
                X_pred[[k], :] = self.predict(Xk)[[-1], 1:]
            X_validation.append(ep[n_samp:, :])
            X_predicted.append(X_pred[n_samp:, :])
        X_validation = np.vstack(X_validation)
        X_predicted = np.vstack(X_predicted)
        # Predict
        return sklearn.metrics.r2_score(X_validation, X_predicted)
