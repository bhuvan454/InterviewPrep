"""
authur by: Bhuvan Chennoju
created on: 16th July 2024

Linear Regression Algorithm

linear algorithm assumes that the linear relationship between the dependent variable(y) and the independent variable(x).

The simple linear regression is:
y = W_1*x_1 + W_2*x_2 + W_3*x_3 + ... + W_n*x_n + b
and more general form of linear regression is:
y = Wx + b + e 
where y is the dependent variable,
      x is the independent variable,
      W is the weight,
      b is the bias,
      e is the error term that we can't explain with the model.


Algorithm:
1) Initialize the weights and bias with random values.
2) calcuate the predicted y with Wx + b
3) calcuate the loss; in this case I am using mean squared error loss; loss = 1/n * sum((y_pred - y)^2)
4) update the weights,bias with gradient of the loss wrt w,b. dw = d(loss)/dW, db = d(loss)/db, W = W - lr*dw, b = b - lr*db
5) either repeat the steps 2-4 until the loss is minimum or for fixed number of iterations.
"""


import numpy as np

class LinearRegression:
    def __init__(self, lr = 0.025,num_iter = 1000,error = 0.0001):
        self.lr = lr
        self.num_iter = num_iter
        self.error = error
        self.weights = None
        self.bias = None
        self.loss = None



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
            self.loss = np.mean((y_pred - y)**2)
            # step 4: update the weights and bias
            dw = (1/num_samples) * np.dot(X.T,(y_pred - y))
            db = (1/num_samples) * np.sum(y_pred - y)
            self.weights -= self.lr*dw
            self.bias -= self.lr*db

            if self.loss < self.error:
                break
    
    def predict(self,X):
        y_pred = np.dot(X,self.weights) + self.bias
        return y_pred

