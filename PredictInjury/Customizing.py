#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:22:54 2019

@author: m_159834
"""

import pandas as pd
import numpy as np

df=pd.read_csv('./data/Causeofinjurymodified.csv',encoding='latin1')
df=df.iloc[:,:2]

docs = df['Cause_of_Injury']

# Preprocessing 
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from spellchecker import SpellChecker
ps=PorterStemmer()
spell = SpellChecker()
stop = stopwords.words('english')
samples= len(docs)
temp=[]
for txt in docs.values:
    txt=' '.join([ps.stem(word) for word in re.sub('[^A-Za-z]',' ',txt).lower().split() if word not in set(stop) and word != 'ee'])
    temp.append(txt)
df['Cause_of_Injury']=pd.Series(temp)
# complete pre-processing for cause of injury text


labels = df['category']
from sklearn.preprocessing import LabelEncoder
lb_en=LabelEncoder()
lb_tr=lb_en.fit_transform(labels)
from keras.utils import np_utils
labels = np_utils.to_categorical(lb_en.fit_transform(labels))


shuffle_indices = np.random.permutation(np.arange(len(labels)))
docs = docs[shuffle_indices]
labels = labels[shuffle_indices]


max_length=400#max([len(x) for x in df['Cause_of_Injury'].values])

c=[]
s= [x for x in df['Cause_of_Injury'].values]
s= [d.split() for d in s]
for wl in s:
    for w in wl:
        c.append(w)
c=len(set(c))
        
'''

s=[len(x) for x in df['Cause_of_Injury'].values]
import matplotlib.pyplot as plt
plt.hist(s, bins = 10)
'''

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(docs, labels , test_size = 0.20)


from keras.preprocessing.text import one_hot

vocab_size = c#5000

X_train = [one_hot(d, vocab_size,split=' ') for d in X_train]
X_test = [one_hot(d, vocab_size,split=' ') for d in X_test]
                  
from keras.preprocessing.sequence import pad_sequences
X_train = pad_sequences(X_train, maxlen=max_length, padding='pre')
X_test = pad_sequences(X_test, maxlen=max_length, padding='pre')

from keras.models import Sequential
model=Sequential()

from keras.layers.embeddings import Embedding
model.add(Embedding(vocab_size,50,input_length=max_length))

from keras.layers import Flatten 
model.add(Flatten())

from keras.layers import GRU, Dense
#model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
#model.add(Dense(16, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=20)

y_pred=model.predict(X_test)
#y_pred = (y_pred > 0.7).astype(float)
#y_pred=lb_en.inverse_transform(y_pred)
y_pred=np.argmax(y_pred,1)
y_test_=np.argmax(y_test,1)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_, y_pred)


loss, accuracy = model.evaluate(X_train, y_train, verbose=1)
print('Training Accuracy is {}'.format(accuracy*100))

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print('Training Accuracy is {}'.format(accuracy*100))


from keras.preprocessing.sequence import pad_sequences
injurydetails=['strong hit on head','flood in river']
c=[]
for txt in injurydetails:
    txt=one_hot(' '.join([ps.stem(word) for word in re.sub('[^A-Za-z]',' ',txt).lower().split() if word not in set(stop) and word != 'ee']),vocab_size)
    c.append(txt)
injurydetails = pad_sequences(c, maxlen=max_length)
res=model.predict(injurydetails)
res=np.argmax(res,1)
res=lb_en.inverse_transform(res)
print(res)

def predictBatch(testData):
    injurydetails=testData
    c=[]
    for txt in injurydetails:
        txt=one_hot(' '.join([ps.stem(word) for word in re.sub('[^A-Za-z]',' ',txt).lower().split() if word not in set(stop) and word != 'ee']),vocab_size)
        c.append(txt)
    injurydetails = pad_sequences(c, maxlen=max_length)
    res=model.predict(injurydetails)
    res=np.argmax(res,1)


    