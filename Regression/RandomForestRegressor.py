import pandas as pd

#numpy is a numerical python uses to high dimentional feature calculations

import numpy as np

#pandas is python data frame library uses to manupulate data


data = pd.read_csv("/home/saireddy/NameError/DOCS/Regression/boston-house-prices/housing_data.csv")

#read_csv is the function from pandas which uses to read csv data 
#and we can manupulate the csv data files

data = data.drop(['Unnamed: 0'], axis=1)

print(data.isna().sum())

#isna function is from pandas which uses to find the null values in data 
#sum will give total number of number of columns in each column

print(data.describe()) 

#describe is EDA(Explorative Data Analysis) step it will all the statical analysis about the data


X = data.iloc[:, :-1].values

#features of the data 

y = data.iloc[:, -1].values

#labels of the data

print(data.dtypes)

#datypes is used to print all the data types of each column in data

from sklearn.model_selection import train_test_split

#importing tarin test split function from model_Section module in sklearn 

X_train,  X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

'''
X -- Features

X_train is for train data with 80%
X_test is for test data with 20%

y -- Labels

y_train is for training with 80%
y_test is for testing with 20%

test_size is how much percentage data we should send into the testing

random_state means it will select test data randomly
'''

from sklearn.ensemble import RandomForestRegressor

#importing Support Vector Regressor from sklearn tSupport Vector Machine

random = RandomForestRegressor() #intilizing Support Vector Regressor
random.fit(X_train, y_train) #fitting our model into the model

y_pred = random.predict(X_test)
#predicting the test data with help of the model

from sklearn.metrics import mean_squared_error
#from sklearn metrics we are import the mean square error
score = mean_squared_error(y_pred, y_test)
#returning the error
print("[ INFO ] score:-", score)

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot# Pull out one tree from the forest
tree = random.estimators_[5]# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot# Pull out one tree from the forest
tree = random.estimators_[5]# Export the image to a dot file
export_graphviz(tree, out_file = '/home/saireddy/NameError/DOCS/Regression/tree.dot', 
feature_names = data.columns[:-1], rounded = True, precision = 1)# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('/home/saireddy/NameError/DOCS/Regression/tree.dot')# Write graph to a png file
graph.write_png('tree.png')
