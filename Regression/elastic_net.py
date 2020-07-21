import numpy as np 
import math as m

x = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5],
              [4, 5, 6], 
              [5, 6, 7]])

y = np.array([[1],
              [0],
              [1],
              [0],
              [0]])

w = np.zeros(y.shape)


class ElastNet(object):

    def __init__(self, x, y, w, alpha, elastic_lambda):
        self.x = x
        self.y = y
        self.w = w
        self.error = np.zeros(y.shape)
        self.alpha = alpha
        self.ridge_final_val = np.zeros(y.shape)
        self.elastic_lambda = elastic_lambda

    def cost_function(self):
        for i in range(len(self.y)):
            y_cap = np.sum(self.w[i] * self.x[i].T)
            self.error[i] = 1/(2*len(self.x))*(np.sum(m.pow((self.y[i] - y_cap), 2)))
        return self.error
    
    def elastic_fit(self):
        alpha_ridge = (1 - self.alpha)/2
        alpha_ridge = alpha_ridge * (np.sum(np.square(w))) 
        alpha_lasso = self.alpha * np.sum(w)
        elastic_lamd = self.elastic_lambda * (alpha_ridge + alpha_lasso)
        return self.error + elastic_lamd
        
elastic = ElastNet(x, y, w, alpha=0, elastic_lambda=1)
predict = elastic.elastic_fit()
print(predict)
