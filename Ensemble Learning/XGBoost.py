import numpy as np
#numpy is a numerical python uses to high dimentional feature calculations
import pandas as pd
#pandas is python data frame library uses to manupulate data
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv("/home/saireddy/NameError/DOCS/Ensemble Learning/iris.csv", names=["sepal_length", "sepal_width", 
                                              "petal_length", "petal_width", "class"])

#read_csv is the function from pandas which uses to read csv data 
#and we can manupulate the csv data files


print(data.isna().sum())

#isna function is from pandas which uses to find the null values in data 
#sum will give total number of number of columns in each column

print(data.describe()) 

#describe is EDA(Explorative Data Analysis) step it will all the statical analysis about the data


X = data.iloc[:, :-1].values
#features of the data 
y = data.iloc[:, -1].values
#labels of the data 

from sklearn.preprocessing import LabelEncoder
#labelencode function imported from sklearn
lb = LabelEncoder() #calling object of the class
y = lb.fit_transform(y) #transforming text into the int

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
import xgboost as xgb

xg = xgb.XGBClassifier() #initilized the XGboost clasifier
xg.fit(X_train, y_train) #fitting the data


y_pred = xg.predict(X_test) #predicting the test data with help of the model


from sklearn.metrics import accuracy_score  #importing metrics from the sklearn
print(accuracy_score(y_pred, y_test)) #printing the accuracy of the data 




data_dmatrix = xgb.DMatrix(data=X,label=y) #dataset into an optimized data structure called Dmatrix


params = {'booster':'gbtree','colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

#params which holds all the hyper-parameters and their values as key-value pairs

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10, 
                    as_pandas=True, seed=123)

#3-fold cross validation model by invoking XGBoost's cv()

print(cv_results.head()) #printing top 5 values of cross validation



#Visualize Boosting Trees and Feature Importance
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=25)


#Plotting the first tree with the matplotlib library:
import matplotlib.pyplot as plt
xgb.plot_tree(xg_reg,num_trees=0)
#plt.rcParams['figure.figsize'] = [50, 10]
plt.savefig("XGB_tree") #saving figure
plt.show() #showing figure


#plotting feature impotance with matplotlib
xgb.plot_importance(xg_reg) 
#plt.rcParams['figure.figsize'] = [5, 5]
plt.savefig("XGB_features") #saving figure
plt.show()  #showing figure
