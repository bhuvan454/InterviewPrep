"""
Authored by: Bhuvan Chennoju
Created on: 28th July 2024

K-Means Algorithm

The K-means algorithm is the most widely used clustering algorithm that uses an explicit distance measure
to partition the data set into clusters.The main objective of the K-Means algorithm is to minimize the sum of distances between 
the points and their respective cluster centroid.

Algorithm:
1) Initialize K centroids randomly
2) Assign each data point to the nearest centroid
3) Recompute the centroids
4) Repeat steps 2 and 3 until convergence


"""

import numpy as np

class KMeans:

    def __init__(self, k=3, max_iters=100, random_seed=2024):
        self.k = k
        self.max_iters = max_iters
        self.random_seed = random_seed
        self.centroids = None
        self.clusters = None

    # Step 1: Initializing K centroids randomly
    def initialize_centroids(self, X):
        np.random.seed(self.random_seed)
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]
        return self.centroids
    
    # Step 2: Assigning each data point to the nearest centroid
    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    # Step 3: Recomputing the centroids
    def update_centroids(self, X):
        self.centroids = np.array([X[self.clusters == k].mean(axis=0) for k in range(self.k)])
        return self.centroids

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()
            self.clusters = self.assign_clusters(X)
            self.centroids = self.update_centroids(X)
            if np.all(old_centroids == self.centroids):
                break
        return self.centroids, self.clusters

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def get_centroids(self):
        return self.centroids

    def get_clusters(self):
        return self.clusters