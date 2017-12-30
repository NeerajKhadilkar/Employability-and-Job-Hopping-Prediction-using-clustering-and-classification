
from rank import final_list

import csv
import numpy as np
import sklearn.metrics as sm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import preprocessing, cross_validation , neighbors
from sklearn import linear_model
import pandas as pd
from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
#from imp import final_li
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

start = timer()

df=pd.DataFrame()
df= (pd.read_csv('C:/Users/neeraj/Desktop/MOCK_DATA.csv',delimiter=','))


#df.drop(['id'], 1, inplace=True)


X=np.array(df.drop(['Performance'], 1))  #all features except class output

y=np.array(df['Performance'])  # class output
seed=100
X_train ,X_test ,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.5,random_state=seed)

li=[DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(),neighbors.KNeighborsClassifier()]



for ele in li:

    acc=ele.fit(X_train,y_train)
    accuracy=acc.score(X_test,y_test)
    print("Accuracy is :",accuracy*100)        

    example_measures=np.array([final_list])    # new data for prediction    
    example_measures=example_measures.reshape(1,-1)
    prediction=ele.predict(example_measures)
    print("Output belongs to class :",prediction)

    if prediction ==1:
    	print("Accomplish")

    if prediction== 2:
    	print("Exceed")

    if prediction==3:
    	print("Far Exceed")

    print()




end = timer()
print("Time required to execute : ",end-start)



conf=sm.confusion_matrix(X_test,y_test)
print("conf is", conf)

