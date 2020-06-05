# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing data
dataset = pd.read_csv('Salary_data.csv')
print (dataset)

#Get features and Labels seperate
X = dataset.iloc[:,0:1].values
y = dataset.iloc[:,-1].values

#Taking Care of missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
imputer.fit(X[:,:])
X = imputer.fit_transform(X)

#Splitting data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


#Modeling 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Plotting
# Training set plot
plt.scatter(X_train,y_train,color='red')
# Test set plot
plt.scatter(X_test,y_test,color='blue')
# Regressor line (model) 
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.show


