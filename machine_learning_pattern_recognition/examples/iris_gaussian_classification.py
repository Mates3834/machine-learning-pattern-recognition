import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.statistical_classification.class_statistics import class_statistics
from src.statistical_classification.gaussian_classifier import GaussianClassifier
from src.statistical_classification.feature_pair_analysis import (
    all_feature_pairs,
    select_pair,
)

iris = load_iris()
X = iris.data
y = iris.target

stats = class_statistics(X, y)
for label, info in stats.items():
    print(f"Class {label}")
    print("Mean:", info["mean"])
    print("Covariance:\n", info["covariance"])

pair = all_feature_pairs(X.shape[1])[0]
X2 = select_pair(X, pair)

X_train, X_test, y_train, y_test = train_test_split(
    X2, y, test_size=0.3, random_state=0, stratify=y
)

clf = GaussianClassifier().fit(X_train, y_train)
print("Test accuracy:", clf.score(X_test, y_test))

plt.figure()
for label in sorted(set(y)):
    pts = X2[y == label]
    plt.scatter(pts[:, 0], pts[:, 1], label=iris.target_names[label])
plt.xlabel(iris.feature_names[pair[0]])
plt.ylabel(iris.feature_names[pair[1]])
plt.title("Iris - Two-Feature View")
plt.legend()
plt.grid(True)
plt.show()
