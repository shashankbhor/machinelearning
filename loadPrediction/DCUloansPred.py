#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:59:28 2019
The model is build for a bank to validate the Loan eligibility of Customer
"""

import pandas as pd
import numpy as np

df=pd.read_csv('trainset.csv')
df.dropna()

df['Gender'].fillna('Male', inplace=True)
df['Self_Employed'].fillna('No', inplace=True)
df['Credit_History'].fillna(0, inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Dependents'].replace('3+','3')

df = df.sample(frac=1)

from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.2,random_state=2)

x_train=train.drop('Loan_Status',axis=1)
y_train=train['Loan_Status']
x_test=test.drop('Loan_Status',axis=1)
y_test=test['Loan_Status']

#create dummies - This is for converting catogorical variable into dummy variables
x_train=pd.get_dummies(x_train)
x_test=pd.get_dummies(x_test)

x_train,x_test = x_train.align(x_test, join='outer', axis=1, fill_value=0)


from sklearn.ensemble import BaggingClassifier
from sklearn import tree

model=BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(x_train,y_train)
model.score(x_test, y_test)

# Lets check for RandomForest
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(random_state=1)
model.fit(x_train, y_train)
model.score(x_test, y_test)

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)
model.fit(x_train, y_train)
model.score(x_test, y_test)

from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(learning_rate=0.01,random_state=1)
model.fit(x_train, y_train)
model.score(x_test, y_test)

import xgboost as xgb
model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model.fit(x_train, y_train)
model.score(x_test, y_test)
