import numpy as np


x = np.array([[1],
                [2],
                [3],
                [4],
                [5]])

y = np.array([[4],
                [12],
                [28],
                [50],
                [80]])



class LinearRegression(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.m = None
        self.b = None

    def finding_mandb(self):
        x1, x2  = self.x[0], self.x[-1]
        y1, y2  = self.y[0], self.y[-1]
        #Slope
        self.m = (y2-y1)/(x2-x1)
        #Bias
        #self.y0  = self.m[0]*self.x0 + (- self.m[0]*x1 + y1)
        self.b = - self.m[0]*x1 + y1
        return self.m[0], self.b[0]

    def finding_predicted(self):
        y0 = []

        for i in range(len(x)):
            y_pred = self.m[0]*self.x[i] + self.b[0]
            y0.append(y_pred)
        return y0
    
    def applying_lse(self):
        x_mean = np.mean(self.x)
        y_mean = np.mean(self.y)
        x_x_mean = []
        y_y_mean = []
        for i in range(len(self.x)):
            sep_mean = self.x[i] - x_mean
            x_x_mean.append(sep_mean)
            sep_mean_y = self.y[i] - y_mean
            y_y_mean.append(sep_mean_y)
        top = sum(np.array(y_y_mean)*np.array(x_x_mean))
        bot = sum(np.array(x_x_mean)**2)
        m = top/bot
        b = y_mean - m * x_mean

        y0_pre = []
        for i in range(len(x)):
            y_pred = m[0]*self.x[i] + b[0]
            y0_pre.append(y_pred)

        return y0_pre

    def mean_square_error(self, y, y_pred):
        error = (1/len(y)) * (sum(y_pred - y)**2)
        return error

obj = LinearRegression(x, y)
m, b = obj.finding_mandb()
y_pred = obj.finding_predicted()
y_pred_1 = obj.applying_lse()
print(y_pred_1)

error = obj.mean_square_error(y, y_pred_1)
print(error)

import matplotlib.pyplot as plt
plt.scatter(x, y, label="Normal values")
plt.plot(x, y)
plt.scatter(x, y_pred, label = "straight line Equation")
plt.plot(x, y_pred)
plt.scatter(x, y_pred_1, label = "Least Square Equation")
plt.plot(x, y_pred_1)
plt.legend(loc="upper left")
plt.title("Linear Regression")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.savefig("LinearRegression.png")
plt.show()