#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:59:28 2019
The model is build for a bank to validate the Loan eligibility of Customer
"""

import pandas as pd
import numpy as np

df=pd.read_csv('trainset.csv')
df['Gender'].fillna('Male', inplace=True)

from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.3,random_state=1)

x_train=train.drop('Loan_Status',axis=1)
y_train=train['Loan_Status']
x_test=test.drop('Loan_Status',axis=1)
y_test=test['Loan_Status']

#create dummies - This is for converting catogorical variable into dummy variables
x_train=pd.get_dummies(x_train)
x_test=pd.get_dummies(x_test)

from sklearn.ensemble import BaggingClassifier
from sklearn import tree

model=BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(x_train,y_train)
