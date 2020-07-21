import numpy as np 


x = np.array(
    [
        [1, 4],
        [2, 5],
        [3, 5], 
        [6, 4]
    ]
)

y = np.array(
    [
        [1],
        [3.3],
        [3.4],
        [4.6]
    ]
)

class Multiple_Linear_Regression(object):

    def __init__(self, learning_rate, iterations):

        self.learning_rate = learning_rate     
        self.iterations = iterations
          
    def predict(self, x, val=False):
        
        if val==False:
            self.y_pred = self.X.dot(self.beta)
        else:
            x = np.insert(x,0,1,axis=1)
            self.y_pred = x.dot(self.beta)
        return self.y_pred
    
    def cost_func(self):
        self.cost = sum((self.y - self.y_pred)**2)/len(self.y)
        return self.cost
    
    def gradient_descent(self):
        self.dbeta = 2/len(x)*(self.X.T.dot(self.X).dot(self.beta) - self.X.T.dot(self.y))      
        self.beta = self.beta - self.learning_rate * self.dbeta

    def fit(self, x, y):       
        self.X = np.insert(x,0,1,axis=1)
        self.y = y
        self.beta = np.zeros((x.shape[1] + 1, 1))
        costs = []    
        for i in range(self.iterations):
            self.predict(x = [])
            self.cost_func()
            costs.append(self.cost)
            self.gradient_descent()
        return costs
    
    def r2_Score(self, x, y):
        ss_res = sum((self.predict(x, val=True) - y)**2)
        ss_tot = sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot
        return r2     

learning_rate = 0.0001
iterations = 1000
obj = Multiple_Linear_Regression(learning_rate=learning_rate, iterations=iterations)
costs = obj.fit(x, y)
error = obj.r2_Score(x, y)

import matplotlib.pyplot as plt
plt.plot([i for i in range(len(costs))], costs, color='r', label="Error Rate",lw=3)
#plt.scatter([i for i in range(len(costs))], costs, color='b', label="Error Rate")
plt.title('Multiple Linear Regression')
plt.xlabel('Iterations')
plt.ylabel('Cost function')
plt.legend(loc="center")
plt.savefig("Multiple_Linear_regression.png")
plt.show()