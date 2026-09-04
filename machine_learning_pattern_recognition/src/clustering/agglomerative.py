import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def agglomerative_labels(X, n_clusters=3, linkage="ward"):
    """Return agglomerative-clustering labels."""
    X = np.asarray(X, dtype=float)
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
    )
    return model.fit_predict(X)


def select_cluster_count(X, candidates=(2, 3, 4, 5), linkage="ward"):
    """
    Select a cluster count using silhouette score.
    Returns best_k and score dictionary.
    """
    X = np.asarray(X, dtype=float)
    scores = {}

    for k in candidates:
        labels = agglomerative_labels(X, k, linkage)
        if len(np.unique(labels)) > 1:
            scores[int(k)] = float(silhouette_score(X, labels))

    if not scores:
        raise ValueError("No valid clustering configuration found.")

    best_k = max(scores, key=scores.get)
    return best_k, scores
