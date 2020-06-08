#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Shashank
- Data should follow some linearity in data 
This model demostrates how should we check the p-Value 
and keep only co-related features for model building
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

# dataset collection
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

X = DataFrame(X,columns=['R&D Spend','Administration','Marketing Spend','State']) 
X  = np.array(X)
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lableencoder_X=LabelEncoder()
X[:, 3] = lableencoder_X.fit_transform(X[:, 3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

# Removing dummy varible trap by removing first dummy variable column
# below line drops '0th' column from dataset i.e. taking columns from 1st
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.score(X_test,y_test))

# Predicting the Test set results
y_pred = regressor.predict(X_test)

''' ------------------------------------------------------'''
'''    Optimizing the model using backward elimination    '''
''' Though sklearn classes will automatically do this     '''  
''' ------------------------------------------------------'''

import statsmodels.api as sm
# Adding first column as '1' to ensure b0*X0 =b0 (of b0X0 + b1X1 + b2X2 ..)
X=np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)

X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
# Above execution gave 99% P-value for X2 (i.e. 3th column) so removing it for next iteration
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
# removing 2
X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#Removing 4
X_opt=X[:,[0,3,5]]  
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#Removing 4
X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


''' As only one variable(R&D spend) is impacting Profit, 
I again used linearregression using only one variable, 
it increased the accuracy on model compare to multiple vars'''

