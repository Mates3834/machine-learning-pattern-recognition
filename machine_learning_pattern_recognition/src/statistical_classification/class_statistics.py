import numpy as np


def class_statistics(X, y):
    """
    Compute class-wise mean vectors and covariance matrices.

    Returns:
        dict[class_label] = {
            "mean": ...,
            "covariance": ...,
            "count": ...
        }
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    out = {}
    for label in np.unique(y):
        Xi = X[y == label]
        out[label] = {
            "mean": Xi.mean(axis=0),
            "covariance": np.cov(Xi, rowvar=False),
            "count": Xi.shape[0],
        }
    return out
