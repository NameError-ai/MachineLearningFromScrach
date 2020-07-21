import numpy as np 
import math as m


x = np.array([[1, 2, 3, 4, 5, 6],
              [2, 3, 4, 4, 5, 6],
              [3, 4, 5, 4, 5, 6],
              [4, 5, 6, 4, 5, 6], 
              [5, 6, 7, 4, 5, 6]])


y = np.array([[3.8],
              [4.2],
              [4.3],
              [2.5],
              [2.8]])


w = np.zeros(y.shape)


class LassoRegression(object):
    
    def __init__(self, x, y, w, alpha):
        self.x = x
        self.y = y
        self.w = w
        self.error = np.zeros(y.shape)
        self.alpha = alpha
        self.lasso_final_val = np.zeros(y.shape)

    def cost_function(self):
        for i in range(len(self.y)):
            y_cap = np.sum(self.w[i] * self.x[i].T)
            self.error[i] = 1/(2*len(self.x))*(np.sum(np.square(self.y[i] - y_cap)))
        return self.error

    def Lasso_fit(self):
        for i in range(len(self.y)):
            weight = self.alpha + np.sum(abs(self.w[i]))
            self.lasso_final_val[i] = error[i] + weight
        return self.lasso_final_val

lasso = LassoRegression(x=x, y=y, w=w, alpha=2)
error = lasso.cost_function()
predict = lasso.Lasso_fit()
print(predict)