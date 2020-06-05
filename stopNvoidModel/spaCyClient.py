#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 21:46:40 2019

@author: Shashank
Client for Modeling 
"""

import spacy
import random
from spacy import displacy
import modelling
from modelling.ModelConfiguration import ModelConfiguration 
from modelling.TrainModel import Training
from modelling.BuildModel import BuildModel
import pandas as pd
from pandas import DataFrame

training_data = [
        ('Check team Pls VOID Check #3018677?', [{'entities': [(27, 34, 'check_number')]},{'entities': [(15, 19, 'check_instruction')]}]), 
               ('Check Team Pls VOID Check #2983294  - Amount $24,7 ?', [{'entities': [(27, 34, 'check_number')]},{'entities': [(15, 19, 'check_instruction')]}]), 
                ('Pls VOID Check #2993435 amount $7101.00?', [{'entities': [(16, 23, 'check_number')]},{'entities': [(4, 8, 'check_instruction')]}]),
                 ('please STOP PAY on check #2962623 Check #/ EFT?', [{'entities': [(26, 33, 'check_number')]},{'entities': [(7, 11, 'check_instruction')]}]), 
                  ('Check #2990276 Stopped / Reissued', [{'entities': [(7, 14, 'check_number')]},{'entities': [(25, 32, 'check_instruction')]}]),
                   ('Check # 3027262  has been voided?', [{'entities': [(8, 15, 'check_number')]},{'entities': [(26, 30, 'check_instruction')]}]),
                    ('A - Check #2949930 Stop Paid / ReIssued?', [{'entities': [(11, 18, 'check_number')]},{'entities': [(31, 38, 'check_instruction')]}]),
                     ('Stop check #2972221?', [{'entities': [(12, 19, 'check_number')]},{'entities': [(1, 4, 'check_instruction')]}]),
                      ('Check #2885046 stopped / reissued?', [{'entities': [(7, 14, 'check_number')]},{'entities': [(25, 32, 'check_instruction')]}]), 
                       ('Stop check#2873200', [{'entities': [(11, 18, 'check_number')]},{'entities': [(1, 4, 'check_instruction')]}]), 
                        ('Stop check#2920316', [{'entities': [(11, 18, 'check_number')]},{'entities': [(1, 4, 'check_instruction')]}]), 
                         ('Check # 2870820 Stoped / Reissued', [{'entities': [(8, 15, 'check_number')]},{'entities': [(25, 32, 'check_instruction')]}])
                          ]



#[{'entities': [(8, 15, 'check_number')]}, {'entities': [(25, 32, 'check_instruction')]}]
# New way - ('Check # 2870820 Stoped / Reissued', [{'entities': [(8, 15, 'check_number'),(25, 32, 'check_instruction')]}]  )

data=pd.read_json("./data/stopnvoid_full.json",lines=True)
trainingData=DataFrame()
train_lbls=data['labels'].values
train_set=[]
for index,txt in enumerate(train_lbls):
    entlst=data['labels'][index]
    #print(entlst)
    entlst_=[{'entities': [tuple(ent_x for ent_x in ent_) for ent_x in ent_] for ent_ in entlst}]
    #entlst_=[{'entities': [ tuple([ent_x for ent_x in ent_])] } for ent_ in entlst]
    train_set.append((data['text'][index],entlst_))

training_data=train_set
# Build Model Configuration 
modelConf=ModelConfiguration('Shanky')
modelConf.addTrainingData(training_data)
modelConf.addEpochs(30)

s=Training()
model=s.buildModel(modelConf)

doc = model('Check # 2870820 Stoped / Reissued')
 
#Check # 2870820 Stoped / Reissued
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)


#Display Entities
displacy.serve(doc,style="ent")

nlp = spacy.blank('en')  # create blank Language class
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)

for  _,tagged_set in training_data:
    print(tagged_set)
    for tagged_item in tagged_set:
        for item in tagged_item.get('entities'):
            print(item[2])
            ner.add_label(item[2])


other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(30):
        random.shuffle(training_data)
        losses = {}
        for text, annotations in training_data:
            annotations=[annotations[0]]
            print(annotations)
            nlp.update(
                [text]*len(annotations),  # batch of texts
                annotations,  # batch of annotations
                drop=0.2,  # dropout - make it harder to memorise data
                sgd=optimizer,  # callable to update weights
                losses=losses)
        print(losses) 

doc=nlp('Check#123455 pls reissue')