from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import math
import numpy as np
from statistics import mode, mean
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class EnsembleLearning(object):
    
    def __init__(self, hard=None, soft=None, weight=None, average=None):
        self.hard = hard
        self.soft = soft
        self.weight = weight
        self.average = average
    
    def data_prep(self, file_name):
        self.data = pd.read_csv("iris.csv", names=["sepal_length", "sepal_width", 
                                              "petal_length", "petal_width", "class"])
        X, y = self.data.iloc[:, 1:4].values, self.data.iloc[:, -1].values
        y = LabelEncoder().fit_transform(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, 
                                                        y, test_size=0.2, random_state=0)
        #print(set(self.data['class']))
        
    def fit(self, algos):
        algos = algos
        self.models = [None, None, None]
        for i in range(len(algos)):
            self.models[i] = algos[i].fit(self.X_train, self.y_train)
            
    def predict(self):
        predict_data = {}
        for prediction in self.models:
            predict_data[prediction.__class__.__name__] = prediction.predict(self.X_test)
        self.predicted = pd.DataFrame(data=predict_data)
        return self.predicted
    
    def prob_predict(self):
        self.soft_predict_data = {}
        self.lis1, self.lis2, self.lis3 = [], [], []
        
        for j, prediction in enumerate(self.models):
            self.soft_predict_data[prediction.__class__.__name__] = [prediction.predict_proba(self.X_test)]
            #print(prediction.predict_proba(self.X_test))
            for i in range(len(self.y_test)):
                lis = np.argmax(prediction.predict_proba(self.X_test)[i])
                if j == 0:
                    self.lis1.append(lis)
                elif j==1:
                    self.lis2.append(lis)
                else:
                    self.lis3.append(lis)
                    
        #print(f"lis1 :{lis1}, lis2 :{lis2}, lis3 :{lis3}")
        self.soft_predicted = pd.DataFrame(data=self.soft_predict_data)
        self.li = np.array([])
        final=(self.soft_predicted.iloc[:, 0][0]+self.soft_predicted.iloc[:, 1][0]+self.soft_predicted.iloc[:, 2][0])/3
        for i in range(len(final)):
            value = np.argmax(final[i])
            #print(value)
            self.li = np.append(self.li, value)
            
        self.soft_predict_data['LogisticRegression'] = self.lis1
        self.soft_predict_data['KNeighborsClassifier'] = self.lis2
        self.soft_predict_data['DecisionTreeClassifier'] = self.lis3
        self.soft_predict_data["Ensemble"] = self.li
        return self.soft_predict_data
    
    def average_predict(self):
        average_predict_data = {}
        for prediction in self.models:
            average_predict_data[prediction.__class__.__name__] = prediction.predict(self.X_test)
        self.average_predicted = pd.DataFrame(data=average_predict_data)
        
        print(self.average_predicted)
        lis_W = []
        value = (self.average_predicted.iloc[:, 0]+self.average_predicted.iloc[:, 1]+self.average_predicted.iloc[:, 2])/3
        return value
    
    def weighted_predict(self):
        #np.random.seed(555)
        weight = list(np.random.random_sample((30,)))
        weighted_predict_data = {}
        for prediction in self.models:
            weighted_predict_data[prediction.__class__.__name__] = prediction.predict(self.X_test)
        self.weighted_predicted = pd.DataFrame(data=weighted_predict_data)
        self.weighted_predicted["weight"] = weight
        #print(self.weighted_predicted)
        lis_W = []
        value = (self.weighted_predicted.iloc[:, 0]*weight+self.weighted_predicted.iloc[:, 1]*weight+self.weighted_predicted.iloc[:, 2]*weight)/3
        for i in range(len(value)):
            value[i] = math.ceil(value[i])
        return value
    
    def Ensemble_select(self):
        if self.hard:
            final = []
            hard_predicted = self.predict()
            for i in range(len(hard_predicted)):
                Ensemble = final.append(mode(hard_predicted.iloc[i,:]))
            hard_predicted["EnsembleLearning"] = final
            score_s = {}
            for i in range(len(hard_predicted.columns)):
                score_s[hard_predicted.columns[i]] = accuracy_score(hard_predicted.iloc[:, i], self.y_test)
            return hard_predicted, score_s
        elif self.soft:
            final_data = self.prob_predict()
            soft_predicted_data = pd.DataFrame(data=final_data)
            soft_predicted_data['Ensemble'] = soft_predicted_data['Ensemble'].astype(int)
            soft_score_s={}
            for i in range(len(soft_predicted_data.columns)):
                soft_score_s[soft_predicted_data.columns[i]] = accuracy_score(soft_predicted_data.iloc[:, i], self.y_test)
            return soft_predicted_data, soft_score_s
        elif self.average:
            data = self.average_predict()
            self.average_predicted["Ensemble"] = data
            self.average_predicted["Ensemble"] = self.average_predicted["Ensemble"].astype(int)
            score_average = {}
            for i in range(len(self.average_predicted.columns)):
                score_average[self.average_predicted.columns[i]] = accuracy_score(self.average_predicted.iloc[:, i], self.y_test)
            return self.average_predicted, score_average
            #raise NotImplementedError("Weighted and Average errors are still IN implementation")
        elif self.weight:
            value = self.weighted_predict()
            #print(value)
            self.weighted_predicted["Ensemble"] = value
            self.weighted_predicted["Ensemble"] = self.weighted_predicted["Ensemble"].astype(int)
            self.weighted_predicted.drop("weight", inplace=True, axis=1)
            #print(self.weighted_predicted)
            score_weighted = {}
            for i in range(len(self.weighted_predicted.columns)):
                score_weighted[self.weighted_predicted.columns[i]] = accuracy_score(self.weighted_predicted.iloc[:, i], 
                                                                                    self.y_test)
            return self.weighted_predicted, score_weighted
            
        else:
            
            raise BaseException("No algorithm were selected ? select from \
                                \n 1.weight \n 2.hard \n 3.soft \n 4.average")
            

algos = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier()]
obj = EnsembleLearning(weight=True)
data = obj.data_prep(file_name="iris.csv")
fitting = obj.fit(algos=algos)
predicted, score = obj.Ensemble_select()

print(f"score :- {score}")
print(f"Predicted data :- {predicted}")