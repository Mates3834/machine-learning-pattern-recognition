import numpy as np


class PCAFromScratch:
    """Compact PCA implementation using eigen-decomposition."""

    def __init__(self, n_components=2, standardize=False):
        self.n_components = int(n_components)
        self.standardize = bool(standardize)

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        if self.standardize:
            self.scale_ = Xc.std(axis=0, ddof=1)
            self.scale_[self.scale_ == 0] = 1.0
            Xc = Xc / self.scale_
        else:
            self.scale_ = np.ones(X.shape[1])

        cov = np.cov(Xc, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]

        self.eigenvalues_ = eigvals[order]
        self.components_ = eigvecs[:, order[:self.n_components]].T

        total = np.sum(self.eigenvalues_)
        self.explained_variance_ratio_ = (
            self.eigenvalues_[:self.n_components] / total
        )
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Xc = (X - self.mean_) / self.scale_
        return Xc @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
