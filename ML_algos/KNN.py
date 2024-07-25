
"""
Authored by: Bhuvan Chennoju
Created on: 18th July 2024

K-Nearest Neighbors Algorithm

K-nearest neighbors algorithm is a classification algorithm that classifies the data points based on the majority of the k-nearest data points.
This algorithm is a non-parametric algorithm, which means it doesn't make any assumptions about the data distribution. 
In this algorithm, the distance between the data points is calculated using the Euclidean distance formula, or Manhattan distance formula, or Minkowski distance formula,or Hamming distance formula, etc.

Algorithm:
1) Initialize the k value.
2) Calculate the distance between the test data point and all the training data points.
3) Sort the distances in ascending order.
4) Get the k-nearest data points.
5) Get the majority class of the k-ne
6) Predict the class of the test data point.


# the formula for Euclidean distance is:
d(x,y) = sprt( (x1 - y1)^2 + (x2 - y2)^2 + ... + (xn - yn)^2 
where x1,x2,..xn are the features of the data point x
      y1,y2,..yn are the features of the data point y

# the formula for Manhattan distance is:
d(x,y) = |x1 - y1| + |x2 - y2| + ... + |xn - yn|
where x1,x2,..xn are the features of the data point x
      y1,y2,..yn are the features of the data point y

# the formula for Minkowski distance is:
d(x,y) = (|x1 - y1|^p + |x2 - y2|^p + ... + |xn - yn|^p)^(1/p)
where x1,x2,..xn are the features of the data point x
      y1,y2,..yn are the features of the data point y
      p is the order of the Minkowski distance. 
      if p = 1, then it is Manhattan distance
      if p = 2, then it is Euclidean distance
      if p = inf, then it is Chebyshev distance

# the formula for Hamming distance is:
d(x,y) = sum(x != y)
where x,y are the binary vectors


Advantages
- Easy to implement: algorithm’s simplicity and accuracy.
- Adapts easily: As new training samples are added, the algorithm adjusts to account for any new data since all training data is stored into memory.

- Few hyperparameters: KNN only requires a k value and a distance metric.

Disadvantages
- Does not scale well: It stores all the training data in memory, which can be computationally expensive.

- Curse of dimensionality: The KNN algorithm tends to fall victim to the curse of dimensionality, which means that it doesn’t perform well with high-dimensional data inputs. 

- Prone to overfitting: Due to the “curse of dimensionality”, KNN is also more prone to overfitting.
 While feature selection and dimensionality reduction techniques are leveraged to prevent this from occurring, 
 the value of k can also impact the model’s behavior. Lower values of k can overfit the data, whereas higher values of k tend to “smooth out” 
 the prediction values since it is averaging the values over a greater area, or neighborhood. However, if the value of k is too high, then it can underfit the data. 

So how can you know the optimum value of “k”? We can decide based on the error calculation of a training and testing set. 
Separating the data into training and test sets allows for an objective model evaluation.

One popular approach is testing different numbers of “k” and measuring the resulting error, 
choosing the “k” value at which an increase will cause a very small decrease in the error sum, 
while a decrease will sharply increase the error sum. 
This point that defines the optimal number is known as the “elbow point”.

# how the KNN algorithm is implemented in Python using the scikit-learn library.
from sklearn.neighbors import KNeighborsClassifier
model_name = ‘K-Nearest Neighbor Classifier’
knnClassifier = KNeighborsClassifier(n_neighbors = 5, metric = ‘minkowski’, p=2)
knn_model = Pipeline(steps=[(‘preprocessor’, preprocessorForFeatures), (‘classifier’ , knnClassifier)])
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)


"""

import numpy as np

class KNN:
    def __init__(self,k = 3, distance = 'euclidean'):
        self.k = k
        self.distance = distance
        self.X_train = None
        self.y_train = None

    def distance_cal(self,x1,x2):
        if self.distance == 'euclidean':
            return np.sqrt(np.sum((x1-x2)**2))
        elif self.distance == 'manhattan':
            return np.sum(np.abs(x1-x2))
        elif self.distance == 'hamming':
            return np.sum(x1 != x2)
        
    def fit(self,X,y):
            self.X_train = X
            self.y_train = y

    def predict(self,X):
            y_pred = [self._predict(x) for x in X]
            return np.array(y_pred)
    
    def _predict(self,x):
         # step 1: calculate the distance between the given x and all the training data points
         distances = [self.distance_cal(x, x_train) for x_train in self.X_train]
          
         # step 2: sort the distances in ascending order, and take the top k data points
         k_indices = np.argsort(distances)[:self.k]

         # step 3: get the majority class of the k-nearest data points
         k_nearest_labels = [self.y_train[i] for i in k_indices]
         most_common = np.bincount(k_nearest_labels).argmax()
         return most_common