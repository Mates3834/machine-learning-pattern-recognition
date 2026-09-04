import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from src.dimensionality_reduction.pca import PCAFromScratch
from src.clustering.kmeans import KMeansFromScratch

iris = load_iris()
X = iris.data

pca = PCAFromScratch(n_components=2, standardize=True)
Z = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)

for k in (2, 3, 4):
    model = KMeansFromScratch(n_clusters=k, seed=0).fit(Z)
    print(f"K={k}, SSE={model.inertia_:.3f}")

model = KMeansFromScratch(n_clusters=3, seed=0).fit(Z)

plt.figure()
plt.scatter(Z[:, 0], Z[:, 1], c=model.labels_)
plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="x",
    s=120,
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA + K-Means")
plt.grid(True)
plt.show()
