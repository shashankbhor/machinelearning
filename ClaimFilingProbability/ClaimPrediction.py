# -*- coding: utf-8 -*-
"""
@author: Shashank
This model provides claim prediction to help business if there 
is probability of policy holder to raise claim

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Get the data for analysis
ins_df=pd.read_csv('insurance.csv')
ins_df_x=ins_df.iloc[:,:-1].values
ins_df_y=ins_df.iloc[:,-1].values
ins_df.info()

#Check corrlation 
import seaborn as sns
cor=ins_df.corr()
sns.heatmap(cor,xticklabels=cor.columns,yticklabels=cor.columns,linecolor='green')

#separate out data for validations, Kept 1001 onward data for validation 
test_ins=ins_df.iloc[1200:]
ins_df_x=ins_df.iloc[:,:-1].values
ins_df_y=ins_df.iloc[:,-1].values

train_ins=ins_df.iloc[:1201,:]

# Define train & test set 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test=train_test_split(ins_df_x,ins_df_y,test_size=.3,random_state=0)

#Build Model 
from sklearn.linear_model import LogisticRegression
insuranceCheck = LogisticRegression(max_iter=100000)
insuranceCheck.fit(X_train, y_train)

# Lets test the model accuracy with test data using test set 
accuracy=insuranceCheck.score(X_test,y_test)
print(accuracy) # 78.60%

coeff = list(insuranceCheck.coef_[0])
labels = list(ins_df.iloc[:,:-1].columns)
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')

#Lets see if ANN does better 
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
classifier.fit(X_train,y_train,batch_size=10,epochs=200)

y_pred=(y_pred > 0.5)


## confusion metrics 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)



