import numpy as np


class GaussianClassifier:
    """
    Gaussian maximum-likelihood classifier with class priors.

    Each class is modeled as:
        x | C_k ~ N(mu_k, Sigma_k)
    """

    def __init__(self, regularization=1e-6):
        self.regularization = float(regularization)
        self.classes_ = None
        self.means_ = {}
        self.covariances_ = {}
        self.priors_ = {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        n = len(y)
        for c in self.classes_:
            Xi = X[y == c]
            mu = Xi.mean(axis=0)
            cov = np.cov(Xi, rowvar=False)
            cov = np.atleast_2d(cov)
            cov += self.regularization * np.eye(cov.shape[0])

            self.means_[c] = mu
            self.covariances_[c] = cov
            self.priors_[c] = len(Xi) / n
        return self

    def _log_likelihood(self, X, c):
        mu = self.means_[c]
        cov = self.covariances_[c]
        inv_cov = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("Covariance matrix must be positive definite.")

        d = X - mu
        quad = np.einsum("...i,ij,...j->...", d, inv_cov, d)
        return (
            -0.5 * quad
            -0.5 * logdet
            -0.5 * X.shape[1] * np.log(2 * np.pi)
            + np.log(self.priors_[c])
        )

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.column_stack(
            [self._log_likelihood(X, c) for c in self.classes_]
        )
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))
