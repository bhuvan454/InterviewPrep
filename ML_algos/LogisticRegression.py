"""
author by: Bhuvan Chennoju
created on: 18th July 2024

Logistic Regression Algorithm

This algorithm calculates the model probabilities of the dependent variable(y) given the independent variable(x).
The logistic regression equation is similar to the linear regression equation but we apply the sigmoid activation
function to get the probabilities of the dependent variable y being binary(0 or 1), or multi-class(0,1,2,3,4,5,6,7,8,9).

so first define the sigmoid function:

sigmoid(x) = 1 / (1 + exp(-x)) Range (0,1) and differentiable at all points, and maximum at x = 0, and minimum at x = -inf, +inf

The logistic regression equation is:
y = sigmoid(Wx + b) + e
where y is the dependent variable,
      x is the independent variable,
      W is the weight,
      b is the bias,
      e is the error term that we can't explain with the model.

Algorithm:
1) Initialize the weights and bias with random values.
2) calcuate the predicted y with sigmoid(Wx + b)
3) calculate the loss function with the predicted y and actual y. 
4) calculate the gradient of the loss function with respect to the weights and bias. 
5) update the weights and bias with the learning rate and gradients.
6) repeat the steps 2-5 until the loss function is minimized.

"""

import numpy as np

class LogisticRegression:

    def __init__(self, lr_rate=0.01, n_iter=1000, error=0.0001):
        """
        Initialize the logistic regression model.
        """
        self.lr_rate = lr_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.error = error

    def sigmoid(self, x):
        """
        Compute the sigmoid of x.
        """
        return 1 / (1 + np.exp(-x))
    
    def predict(self, X):
        """
        Predict the probabilities for the given input X.
        """
        return self.sigmoid(np.dot(X, self.weights) + self.bias)
    
    def loss(self, y, y_pred):
        """
        Compute the log likelihood loss function.
        
        logloss = -1/n * sum(y * log(y_pred) + (1 - y) * log(1 - y_pred))
        """
        return -1/len(y) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def fit(self, X, y):
        """
        Fit the model to the data X and y.
        """
        # Step 1: Initialize weights and bias
        self.weights = np.random.rand(X.shape[1])
        self.bias = np.random.rand()
        
        # Step 2 - 6: Iterative optimization
        for i in range(self.n_iter):
            # Step 2: Calculate the predicted y
            y_pred = self.predict(X)
            
            # Step 3: Calculate the loss
            loss = self.loss(y, y_pred)
            print(f"Loss at iteration {i}: {loss}")
            
            # Step 4: Calculate the gradients
            dw = 1 / len(y) * np.dot(X.T, (y_pred - y))
            db = 1 / len(y) * np.sum(y_pred - y)
            
            # Step 5: Update weights and bias
            self.weights -= self.lr_rate * dw
            self.bias -= self.lr_rate * db
            
            # Early stopping criterion
            if loss < self.error:
                print(f"Early stopping at iteration {i} with loss {loss}")
                break

        return self.weights, self.bias
    
    def accuracy(self, y_true, y_pred, threshold=0.5):
        """
        Compute the accuracy of the model.
        """
        y_pred_labels = [1 if i > threshold else 0 for i in y_pred]
        accuracy = np.sum(y_true == y_pred_labels) / len(y_true)
        return accuracy