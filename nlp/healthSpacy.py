# -*- coding: utf-8 -*-
"""
@ Shashank
"""

import spacy
import pandas as pd
from spacy import displacy

df=pd.read_excel('Email parsing output.xlsx')

doc_sub=df['Body'].values

docLst=[]
for doc in doc_sub:
    docLst.append(nlp(doc))
    
docObj=docLst[0]

for num,sentence in enumerate(docObj.sents):
    print(f'{num} : {sentence}')
    
lst_sen=[sentenses.text for sentenses in docObj.sents]
lst_tok=[tok.text for tok in docObj]



