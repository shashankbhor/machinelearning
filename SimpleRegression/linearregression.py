#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:55:06 2019

@author: Shashank
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get the data
dataset=pd.read_csv('Salary_Data.csv')

# Split Dependand and InDependant variable 
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1:2].values


# Spilt dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Fitting Simple Linear regression model 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# predicting the test-set
y_pred=regressor.predict(X_test)

# Visualizing Training set
plt.scatter(X_train,y_train,c='red')
plt.plot(X_train,regressor.predict(X_train),c='blue')
plt.title("Simple Linear Regression Model")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# Visualizing Test data
plt.scatter(X_test,y_test,c='green')
plt.plot(X_train,regressor.predict(X_train),c='blue')
plt.title("Simple Linear Regression Model")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# Saving Model on Disk for future use 
import pickle
fileName="finalized_linearRegressionModel.sav"
pickle.dump(regressor,open(fileName,'wb'))

# loading the model from disk
loaded_model=pickle.load(open(fileName,'rb'))
result=loaded_model.score(X_test,y_test)
print(result)
## Another way to save the model using joblob 
''' Joblib is part of the SciPy ecosystem and provides utilities for pipelining 
python jobs. It provides utilities for saving and loading objects that make use 
of NumPy datastructures efficiently ''' 

from sklearn.externals import joblib
fileNameJob='finalized_job_linearRegression.sav'
joblib.dump(regressor,fileNameJob)

#loading from joblib
loaded_model=joblib.load(fileNameJob)
result=loaded_model.score(X_test,y_test)
print(result)