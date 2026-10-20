
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class MyKMeans:
    def __init__(self, k=3, max_iters=200, tol=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        np.random.seed(self.random_state)

        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            labels = self._assign_clusters(X)
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.k)])
            if np.all(np.abs(self.centroids - new_centroids) < self.tol):
                break
            self.centroids = new_centroids

        self.labels_ = labels

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self._assign_clusters(X)

    def getCost(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        distances = np.min(self._calculate_distances(X, self.centroids), axis=1)
        return np.sum(distances ** 2)

    def _assign_clusters(self, X):
        distances = self._calculate_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

    def _calculate_distances(self, X, centroids):
        distances = np.zeros((X.shape[0], self.k))
        for idx, centroid in enumerate(centroids):
            distances[:, idx] = np.linalg.norm(X - centroid, axis=1)
        return distances
