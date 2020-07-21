import numpy as np 

#numpy is a numerical python uses to high dimentional feature calculations

import pandas as pd 

#pandas is python data frame library uses to manupulate data


data = pd.read_csv("./iris.csv", names=['sepal length', 'sepal width'
                                        , 'petal length', 'petal width', 'class'])


#read_csv is the function from pandas which uses to read csv data 
#and we can manupulate the csv data files

print(data.isna().sum())

#isna function is from pandas which uses to find the null values in data 
#sum will give total number of number of columns in each column

print(data.describe()) 

#describe is EDA(Explorative Data Analysis) step it will all the statical analysis about the data

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
data['class'] = lb.fit_transform(data['class'])

X = data.iloc[:, [1, 2]].values

#features of the data 

y = data.iloc[:, -1].values

#labels of the data

print(data.dtypes)

#datypes is used to print all the data types of each column in data

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
data['class'] = lb.fit_transform(data['class'])


from sklearn.model_selection import train_test_split

#importing tarin test split function from model_Section module in sklearn 

x_train,  x_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

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


from sklearn.svm import SVC

#importing Support Vector Classifier from sklearn Support Vector Machine 


model = SVC() #intilizing Support Vector Classifier
model.fit(x_train, y_train) #fitting our model into the model


y_pred = model.predict(x_test)
#predicting the test data with help of the model


from sklearn.metrics import accuracy_score
#importing metrics from the sklearn
print(accuracy_score(y_pred, y_test))
#printing the accuracy of the data

import matplotlib.pyplot as plt

print(x_train[:, (0,1)].shape)
m,n = x_train.shape[0],x_train.shape[1]
positive = (y_train==1).reshape(m,1)
negative = (y_train==0).reshape(m,1)
plt.figure(figsize=(8,6))
plt.scatter(x_train[positive[:,0],0],x_train[positive[:,0],1],c="r",marker="*",s=50)
plt.scatter(x_train[negative[:,0],0],x_train[negative[:,0],1],c="y",marker="o",s=50)
# plotting the decision  by taking minmum and maximum values from frist column and second column in the x data
X_1,X_2 = np.meshgrid(np.linspace(x_train[:,0].min(),x_train[:,1].max(),num=100),np.linspace(x_train[:,1].min(),
                                                                                             x_train[:,1].max(),num=100))
plt.contour(X_1,X_2,model.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),0,colors="b")
plt.xlim(1.5, 6)
plt.ylim(0.5, 6)
plt.title("how the svc learn the things using linear kernel")
plt.savefig("SVc.png")
plt.show()