import numpy as np
import matplotlib.pyplot as plt

from src.clustering.kmeans import KMeansFromScratch
from src.clustering.agglomerative import select_cluster_count

rng = np.random.default_rng(0)

X = np.vstack([
    rng.normal(loc=(-2, -2), scale=0.6, size=(80, 2)),
    rng.normal(loc=(2, 0), scale=0.7, size=(80, 2)),
    rng.normal(loc=(0, 3), scale=0.5, size=(80, 2)),
])

for k in (2, 3, 4, 5):
    model = KMeansFromScratch(n_clusters=k, seed=0).fit(X)
    print(f"K-Means K={k}: SSE={model.inertia_:.3f}")

best_k, scores = select_cluster_count(
    X,
    candidates=(2, 3, 4, 5),
)
print("Agglomerative silhouette scores:", scores)
print("Selected cluster count:", best_k)

model = KMeansFromScratch(n_clusters=3, seed=0).fit(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Synthetic Clustering Example")
plt.grid(True)
plt.show()
