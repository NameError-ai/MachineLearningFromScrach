from sklearn.ensemble import BaggingClassifier
#import ting baggingclassifier from ensemble class
from sklearn.ensemble import RandomForestClassifier
#importing randomforestclassifier from ensemble class
import numpy as np
#numpy is a numerical python uses to high dimentional feature calculations
import pandas as pd
#pandas is python data frame library uses to manupulate data
import warnings
warnings.filterwarnings('ignore')




data = pd.read_csv("iris.csv", names=["sepal_length", "sepal_width", 
                                              "petal_length", "petal_width", "class"])
#read_csv is the function from pandas which uses to read csv data 
#and we can manupulate the csv data files


print(data.isna().sum())

#isna function is from pandas which uses to find the null values in data 
#sum will give total number of number of columns in each column

print(data.describe()) 

#describe is EDA(Explorative Data Analysis) step it will all the statical analysis about the data



X = data.iloc[:, [0, 2]].values
#features of the data 
y = data.iloc[:, -1].values
#labels of the data 
#print(y.shape)

from sklearn.model_selection import train_test_split
#importing tarin test split function from model_Section module in sklearn 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

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

ra = RandomForestClassifier() #intilizing randomforest classifier

bg = BaggingClassifier(base_estimator=ra, n_estimators=100) #intilizing Baggingclassifier
bg.fit(X_train, y_train) #fitting the data 


y_pred = bg.predict(X_test) #predicting the test data with help of the model


from sklearn.metrics import accuracy_score #importing metrics from the sklearn
print(accuracy_score(y_pred, y_test)) #printing the accuracy of the data 

