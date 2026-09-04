from itertools import combinations
import numpy as np


def all_feature_pairs(n_features):
    """Return all unique two-feature index combinations."""
    return list(combinations(range(n_features), 2))


def select_pair(X, pair):
    """Select two columns from a feature matrix."""
    X = np.asarray(X)
    return X[:, list(pair)]
