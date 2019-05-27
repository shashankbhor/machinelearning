# -*- coding: utf-8 -*-
"""

@author: Shashank

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('insurance.csv')
X=dataset.iloc[:,0:7].values
y=dataset.iloc[:,7:8].values

#Encoding categorical variables 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_geo=LabelEncoder()
X[:,1]=labelEncoder_geo.fit_transform(X[:,1])
labelEncoder_sex=LabelEncoder()
X[:,2]=labelEncoder_geo.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

# Spliting Training & Tesing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

## ANN model setting - It needs Sequencial and Dense models
import keras
from keras.models import Sequential
from keras.layers import Dense

### > Initializing the ANN
classifier=Sequential()

### . Adding layers input layer and first hidden layer
# Activation fucntion is  rectifier fuction 
classifier.add(Dense(input_dim=7,output_dim=4,kernel_initializer="uniform",activation="relu"))

### Adding second layer 
classifier.add(Dense(output_dim=4,kernel_initializer="uniform",activation="relu"))

## Adding output layer > This is going to be classifier with only binary output 
# Activation function is Sigmoid 
classifier.add(Dense(output_dim=1,kernel_initializer="uniform",activation="sigmoid"))

## Compiling ANN - its process of applying schocastic gradient decent 
# optimier - applying optimizer to find optimum weights  
# loss - loss function ( e.g. sun of squared errors )
# metrics - criteria for verifying observation 
classifier.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])


## Fitting Training set to ANN and training the model 
classifier.fit(X_train,y_train,batch_size=10,epochs=300)

## Prediction 
y_pred=classifier.predict(X_test)

### last function is sigmoid so it will have probability so need to 
## check it with probability ...lets set ot for 0.5 (50%)
y_pred=(y_pred > 0.5)


## confusion metrics 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

