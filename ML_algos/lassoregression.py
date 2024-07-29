"""
author by: Bhuvan Chennoju
created on: 28th July 2024

Lasso Regression Algorithm

This is the linear regression algorithm with L1 regularization. Lasso regression is used when we have a large number 
of features and we want to avoid overfitting. Lasso regression adds the L1 regularization term to the loss function.

The loss function for Lasso regression is:
loss = 1/n * sum((y_pred - y)^2) + lambda*sum(|W|)

where lambda is the regularization parameter.

Algorithm:
1) Initialize the weights and bias with random values.
2) calcuate the predicted y with Wx + b
3) calcuate the loss; in this case I am using mean squared error loss; loss = 1/n * sum((y_pred - y)^2) + lambda*sum(|W|)
4) update the weights,bias with gradient of the loss wrt w,b. dw = d(loss)/dW, db = d(loss)/db, W = W - lr*dw, b = b - lr*db
5) either repeat the steps 2-4 until the loss is minimum or for fixed number of iterations.


"""

import numpy as np

class LassoRegression:

    def __init__(self, lr = 0.025,num_iter = 1000,error = 0.0001,lambda_ = 0.01):
        self.lr = lr
        self.num_iter = num_iter
        self.error = error
        self.weights = None
        self.bias = None
        self.loss = None
        self.lambda_ = lambda_

    def fit(self,X,y):
        num_samples, num_feats = X.shape
        # step 1: initialize the weights and bias
        self.weights = np.random.randn(num_feats)
        self.bias = np.random.randn()

        #step: 2-5
        for _ in range(self.num_iter):
            # step 2: predicted y 
            y_pred = self.predict(X)
            # step 3: calculate the loss
            self.loss = np.mean((y_pred - y)**2) + self.lambda_*np.sum(np.abs(self.weights)) # this is only variation from linear regression, added a L1 regularization term
            # step 4: update the weights and bias
            dw = (1/num_samples) * np.dot(X.T,(y_pred - y)) + self.lambda_*np.sign(self.weights)
            db = (1/num_samples) * np.sum(y_pred - y)
            self.weights -= self.lr*dw
            self.bias -= self.lr*db

            if self.loss < self.error:
                break

    def predict(self,X):
        return np.dot(X,self.weights) + self.bias
    
    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
