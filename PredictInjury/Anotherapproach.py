#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:17:54 2019

@author: m_159834
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from spellchecker import SpellChecker
from pandas import DataFrame

def preprocess(docs=DataFrame()):
    ps=PorterStemmer()
    spell = SpellChecker()
    stop = stopwords.words('english')
    samples= len(docs)
    temp=[]
    for txt in docs.values:
        txt=' '.join([ps.stem(word) for word in re.sub('[^A-Za-z]',' ',txt).lower().split() if word not in set(stop) and word != 'ee'])
        temp.append(txt.strip())
    return pd.Series(temp)


# Load
df=pd.read_csv('./data/Causeofinjurymodified.csv',encoding='latin1')
df=df.iloc[:,:2]

#Pre-process
df['Cause_of_Injury']=preprocess(df['Cause_of_Injury'])

# just shuffle your data - my data is order by first column so need to shuffle 
shuffle_indices = np.random.permutation(np.arange(len(df['Cause_of_Injury'])))
docs = df['Cause_of_Injury'][shuffle_indices]
oh_doc=docs
labels = df['category'][shuffle_indices]
maxlen=max([len(txt.split()) for txt in docs])


#Label encoding
from sklearn.preprocessing import LabelEncoder
lb_en=LabelEncoder()
lb_tr=lb_en.fit_transform(labels)
from keras.utils import np_utils
labels = np_utils.to_categorical(lb_en.fit_transform(labels))

# Getting ready for word embedding
# calculate the needed length of vectors ( numeric representation of each row in dataset)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizerObj=Tokenizer()
tokenizerObj.fit_on_texts(docs)
docs=tokenizerObj.texts_to_sequences(docs)
docs=pad_sequences(docs,maxlen=maxlen,padding='post')
vocab_size=len(tokenizerObj.word_index)+1


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(docs, labels , test_size = 0.20)

from keras.preprocessing.text import one_hot
oh_docs = [one_hot(d, vocab_size,split=' ') for d in oh_doc.values]
oh_docs = pad_sequences(oh_docs, maxlen=maxlen, padding='post')



# Its ready for Embedding layer - Embedding(vocabulary size,size of the real-valued vector space EMBEDDING_DIM , maximum length of input documents 
# Lets consider building EMBEDDING_DIM = 100


# Start building model 
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Flatten 
from keras.layers import GRU, Dense

model=Sequential()
model.add(Embedding(vocab_size,50,input_length=maxlen))
model.add(Flatten())

#model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=20,verbose=2)

testTxt=['right ankl burn']
res=model.predict(pad_sequences(tokenizerObj.texts_to_sequences(testTxt),maxlen=maxlen))
res=lb_en.inverse_transform(np.argmax(res,1))
print(res)

y_pred=model.predict(X_test)
y_pred=np.argmax(y_pred,1)
y_test=np.argmax(y_test,1)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
