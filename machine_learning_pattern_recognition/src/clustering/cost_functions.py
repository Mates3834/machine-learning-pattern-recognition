import numpy as np


def within_cluster_sse(X, labels, centers=None):
    """Compute total within-cluster sum of squared errors."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    unique = np.unique(labels)
    total = 0.0

    for c in unique:
        members = X[labels == c]
        if len(members) == 0:
            continue
        if centers is None:
            center = members.mean(axis=0)
        else:
            center = np.asarray(centers)[int(c)]
        total += np.sum((members - center) ** 2)

    return float(total)
