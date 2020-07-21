from collections import Counter
import numpy as np
import pandas as pd
import random
import math
from scipy import optimize
from gredient_decent import gradient_descent

x = np.array([  [1], 
                [2], 
                [3], 
                [4], 
                [5]], dtype=np.float64)

y = np.array([  [2],
                [5],
                [8],
                [10],
                [12]], dtype=np.float64)

class PolynomialRegression(object):

    def __init__(self, x, y, degree, starting_b=None,
        starting_m=None, learning_rate=None, num_iterations=None):
        self.x = x
        self.y = y
        self.degree = degree
        #self.points = points
        self.starting_b = starting_b
        self.starting_m = starting_m
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def polynomial_matrix(self):
        X = np.zeros((x.shape[0], self.degree))
        for i in range(X.shape[0]):
            count = 0
            for j in range(X.shape[1]):
                val = np.array([self.x[i]**abs(self.degree-count)])
                count = count + 1
                X[i][j] = float(val[0])
        return X

    def fit(self, points=None):
        #print(points, self.starting_b, self.starting_m, self.learning_rate,self.num_iterations)
        m, b = gradient_descent(points, self.starting_b, self.starting_m, 
        self.learning_rate, self.num_iterations)
        m = m.reshape(self.degree, 1)
        result = (points[:, :-1].dot(m)) + min(b)
        return result


starting_m = 0
starting_b = 0 
learning_rate = 0.0001
num_iterations = 1000

poly = PolynomialRegression(x=x, y=y, degree=2, starting_b=starting_b,starting_m=starting_m, 
learning_rate=learning_rate, num_iterations=num_iterations)
X_train = poly.polynomial_matrix()
points = np.append(X_train, y, axis=1)
predict = poly.fit(points=points)
print(predict)