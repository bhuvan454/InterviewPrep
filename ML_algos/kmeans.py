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


