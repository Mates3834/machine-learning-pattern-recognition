# Machine Learning and Pattern Recognition

A collection of graduate-level studies in **machine learning, statistical pattern recognition, dimensionality reduction, and unsupervised learning**.

The project investigates data representation and classification using statistical properties, followed by dimensionality reduction and clustering techniques.

---

## Overview

The repository covers:

- Statistical pattern analysis
- Feature selection
- Mean and covariance analysis
- Gaussian-based classification
- Decision boundaries
- Principal Component Analysis (PCA)
- K-Means clustering
- Agglomerative clustering
- Cluster evaluation

The experiments are primarily performed using the **Iris dataset**, together with synthetic datasets for clustering analysis.

---

# 1. Statistical Pattern Analysis

The Iris dataset is divided according to its class labels.

For each class, statistical properties of the feature vectors are calculated.

These include:

- Mean vectors
- Covariance matrices
- Feature distributions
- Pairwise feature combinations

The general workflow is:

```text
Dataset
   ↓
Class Separation
   ↓
Feature Selection
   ↓
Mean & Covariance Estimation
   ↓
Statistical Analysis
```

---

# 2. Feature Pair Analysis

Different two-dimensional combinations of the available features are investigated.

The objective is to examine how effectively different feature combinations separate the classes.

```text
Feature Space
     ↓
Select Feature Pair
     ↓
Class Distributions
     ↓
Statistical Modeling
     ↓
Decision Regions
```

This allows the discriminative characteristics of different feature combinations to be compared.

---

# 3. Gaussian Classification

Class statistics are used to investigate Gaussian-based pattern classification.

For each class, the estimated:

- Mean vector
- Covariance matrix

are used to characterize the corresponding feature distribution.

The resulting distributions can be visualized in the selected feature spaces and used to analyze classification boundaries.

---

# 4. Principal Component Analysis

**Principal Component Analysis (PCA)** is used for dimensionality reduction.

Before PCA, the data are centered around their mean.

Two configurations are investigated.

### One-Dimensional PCA

The dataset is projected onto the **first principal component**.

```text
Original Features
       ↓
 Mean Centering
       ↓
      PCA
       ↓
First Principal Component
       ↓
  1D Representation
```

### Two-Dimensional PCA

The first two principal components are used to obtain a two-dimensional representation.

```text
Original Feature Space
         ↓
        PCA
         ↓
PC1 + PC2 Representation
```

This enables clustering behavior to be examined in a reduced-dimensional feature space.

---

# 5. K-Means Clustering

K-Means clustering is applied to the PCA-transformed data.

Different numbers of clusters are investigated:

- K = 2
- K = 3
- K = 4

The experiments are performed for both:

- 1D PCA representation
- 2D PCA representation

This makes it possible to investigate how dimensionality reduction and the selected number of clusters affect the resulting data partition.

---

# 6. Synthetic Dataset Experiments

In addition to the Iris dataset, clustering experiments are performed using synthetic data.

K-Means is evaluated for:

- K = 2
- K = 3
- K = 4
- K = 5

Different clustering configurations are compared using clustering cost criteria.

---

# 7. Agglomerative Clustering

**Agglomerative clustering** is investigated as an alternative unsupervised learning approach.

Unlike K-Means, hierarchical clustering progressively combines samples or clusters according to their similarity.

Conceptually:

```text
Individual Samples
        ↓
Nearest Groups Combined
        ↓
Larger Clusters
        ↓
Hierarchical Structure
        ↓
Selected Number of Clusters
```

The method is used to investigate an appropriate number of clusters for the analyzed dataset.

---

# Experimental Pipeline

The overall machine-learning workflow can be summarized as:

```text
                 Dataset
                    │
                    ▼
             Preprocessing
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 Statistical Analysis        PCA
          │                   │
          ▼                   ▼
Gaussian Analysis      Reduced Features
                              │
                     ┌────────┴────────┐
                     ▼                 ▼
                  K-Means       Agglomerative
                     │            Clustering
                     └────────┬────────┘
                              ▼
                       Cluster Analysis
```

---

# Technologies

- MATLAB
- Machine Learning
- Pattern Recognition
- Principal Component Analysis
- K-Means
- Agglomerative Clustering
- Statistical Classification
- Data Visualization

---

# Research Areas

- Machine Learning
- Pattern Recognition
- Statistical Learning
- Unsupervised Learning
- Dimensionality Reduction
- Clustering
- Data Analysis

---

# Repository Structure

```text
machine-learning-pattern-recognition/
│
├── README.md
│
├── statistical-classification/
│   ├── feature-analysis/
│   ├── gaussian-modeling/
│   └── decision-boundaries/
│
├── dimensionality-reduction/
│   └── pca/
│
├── clustering/
│   ├── kmeans/
│   └── agglomerative/
│
├── datasets/
│
├── results/
│   ├── feature-distributions/
│   ├── pca/
│   └── clustering/
│
└── docs/
```

---

# Project Status

This repository consolidates graduate-level work in **Machine Learning and Artificial Neural Networks**.

The project focuses on statistical pattern analysis, dimensionality reduction, and unsupervised learning using both real and synthetic datasets.
