import numpy as np
import sklearn.base
import sklearn.metrics


class KoopmanPipeline(sklearn.base.BaseEstimator):

    def __init__(self, preprocessing=None, delay=None, lifting_function=None,
                 estimator=None):
        self.preprocessing = preprocessing
        self.delay = delay
        self.lifting_function = lifting_function
        self.estimator = estimator

    def fit(self, X, y=None, n_u=0):
        # Clone estimators
        self.delay_ = sklearn.base.clone(self.delay)
        self.estimator_ = sklearn.base.clone(self.estimator)
        self.preprocessing_ = sklearn.base.clone(self.preprocessing)
        self.lifting_function_ = sklearn.base.clone(self.lifting_function)
        # Save number of inputs
        self.n_x_ = X.shape[1] - n_u - 1
        self.n_u_ = n_u
        # TODO Pre-processing
        Xp = np.hstack((
            X[:, [0]],
            self.preprocessing_.fit_transform(X[:, 1:])
        ))
        # Delays
        self.delay_.fit(Xp[:, 1:], n_u=n_u)
        episodes = []
        for i in np.unique(Xp[:, 0]):
            episodes.append((i, Xp[Xp[:, 0] == i, 1:]))
        delayed_episodes = []
        # Delay episode
        for (i, ep) in episodes:
            delayed_ep = self.delay_.transform(ep)
            delayed_episodes.append(np.hstack((
                i * np.ones((delayed_ep.shape[0], 1)),
                delayed_ep,
            )))
        Xd = np.vstack(delayed_episodes)
        # TODO Lifting functions
        Xt = np.hstack((
            Xd[:, [0]],
            self.lifting_function_.fit_transform(Xd[:, 1:],
                                                 n_u=self.delay_.n_ud_)
        ))
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
            Xp = self.preprocessing_.transform(ep)
            Xd = self.delay_.transform(Xp)
            Xt = self.lifting_function_.transform(Xd)
            # Pad inputs with zeros so inverse_transform has same dimension as
            # fit(). TODO Is this necessary?
            pred = self.estimator_.predict(Xt)
            Xdp = np.hstack((
                pred,
                np.zeros((pred.shape[0], self.lifting_function_.n_ul_))
            ))
            Xp = self.preprocessing_.inverse_transform(
                self.delay_.inverse_transform(
                    self.lifting_function_.inverse_transform(
                        Xdp)))
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

    def score(self, X, y=None):
        episodes = []
        for i in np.unique(X[:, 0]):
            episodes.append((i, X[X[:, 0] == i, 1:]))
        n_samp = self.delay_.n_samples_needed_
        X_validation = []
        X_predicted = []
        for (i, ep) in episodes:
            X_pred = np.empty((ep.shape[0], self.n_x_))
            X_pred[:n_samp, :] = ep[:n_samp, :self.n_x_]
            for k in range(n_samp, X_pred.shape[0]):
                Xk = np.hstack((
                    i * np.ones((n_samp, 1)),
                    X_pred[(k-n_samp):k, :],
                    ep[(k-n_samp):k, self.n_x_:],
                ))
                X_pred[[k], :] = self.predict(Xk)[[-1], 1:]
            X_validation.append(ep[n_samp:, :self.n_x_])
            X_predicted.append(X_pred[n_samp:, :])
        X_validation = np.vstack(X_validation)
        X_predicted = np.vstack(X_predicted)
        # Predict
        return sklearn.metrics.r2_score(X_validation, X_predicted)
