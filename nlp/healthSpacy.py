# -*- coding: utf-8 -*-
"""
@ Shashank
"""

import spacy
import pandas as pd
import numpy as np
from spacy import displacy
from spacy.matcher import Matcher

df=pd.read_csv('instruction_data.csv')
df=df.iloc[:,0:1].values


    
nlp=spacy.load('en')
matcher=Matcher(nlp.vocab)

doc = nlp(u"A complex-example,!")
print([token.text for token in doc])
pateren=[{"LOWER":"complex"},{"IS_PUNCT":True},{"STARTSWITH":"exa"},{"ENDSWITH":","}]
matcher.add("patName", None, pateren)

matches=matcher(doc)
for matcher_id,start,end in matches:
    matched_span=doc[start:end].text
    print(f'{matcher_id} --- {nlp.vocab.strings[matcher_id]} -- {start} ---- {end} --- {matched_span}')


TRAIN_DATA = [('what is the price of polo?', {'entities': [(21, 25, 'PrdName')]}), ('what is the price of ball?', {'entities': [(21, 25, 'PrdName')]}), ('what is the price of jegging?', {'entities': [(21, 28, 'PrdName')]}), ('what is the price of t-shirt?', {'entities': [(21, 28, 'PrdName')]}), ('what is the price of jeans?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of bat?', {'entities': [(21, 24, 'PrdName')]}), ('what is the price of shirt?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of bag?', {'entities': [(21, 24, 'PrdName')]}), ('what is the price of cup?', {'entities': [(21, 24, 'PrdName')]}), ('what is the price of jug?', {'entities': [(21, 24, 'PrdName')]}), ('what is the price of plate?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of glass?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of moniter?', {'entities': [(21, 28, 'PrdName')]}), ('what is the price of desktop?', {'entities': [(21, 28, 'PrdName')]}), ('what is the price of bottle?', {'entities': [(21, 27, 'PrdName')]}), ('what is the price of mouse?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of keyboad?', {'entities': [(21, 28, 'PrdName')]}), ('what is the price of chair?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of table?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of watch?', {'entities': [(21, 26, 'PrdName')]})]
for text, annotations in TRAIN_DATA:
    print(annotations)


doc=nlp('Check team Pls VOID Check # 3018677 amount $36,812.96  ;  ;  ; Check #/ EFT #:000003018677   Issue Date:11/01/2017      Payable To:Southeast Toyota Finance   Amount:36,812.96 USD;   Thirty six thousand eight hundred twelve and 96/100     Mail To Name:   Southeast Toyota Finance   Bank:   USD Bank Of America     Address:Lockbox #8881 PO Box: 8500 Philadelphia, PA 19178-8500 US')

def getAllTokens(doc):
    return [token for token in doc]

for t in getAllTokens(doc):
    print(t.pos_)
    
for token in doc:
    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(
        token.text,
        token.idx,
        token.lemma_,
        token.is_punct, 
        token.is_space,
        token.shape_,
        token.pos_,
        token.tag_,
        token.lefts,
        token.rights
    ))              
        
for word in doc:
    if word.is_stop==True:
        print(word)
    
displacy.serve(doc, style='dep')


fileNotes= pd.read_csv('instruction_data.csv')

for note in fileNotes.iterrows():
    for str1 in note[1].values:
        for entity in nlp(str1).ents:
            if(entity.label_=='DATE'):
                print(f' {entity.text}  - {entity.label_}')  
                
    
lst_ent=[ent.text for ent in doc.ents]

