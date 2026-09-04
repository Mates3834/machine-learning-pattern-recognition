import numpy as np


class KMeansFromScratch:
    """Educational K-Means implementation."""

    def __init__(self, n_clusters=3, max_iter=200, tol=1e-6, seed=None):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rng = np.random.default_rng(seed)

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        idx = self.rng.choice(
            len(X),
            size=self.n_clusters,
            replace=False,
        )
        centers = X[idx].copy()

        for _ in range(self.max_iter):
            distances = np.linalg.norm(
                X[:, None, :] - centers[None, :, :],
                axis=2,
            )
            labels = np.argmin(distances, axis=1)

            new_centers = centers.copy()
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members):
                    new_centers[k] = members.mean(axis=0)

            if np.linalg.norm(new_centers - centers) < self.tol:
                centers = new_centers
                break
            centers = new_centers

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(
            np.sum((X - centers[labels]) ** 2)
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = np.linalg.norm(
            X[:, None, :] - self.cluster_centers_[None, :, :],
            axis=2,
        )
        return np.argmin(distances, axis=1)
