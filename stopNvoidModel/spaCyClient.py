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
from ModelConfiguration import ModelConfiguration 

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



# Build Model Configuration 
modelConf=ModelConfiguration('Shanky')
modelConf.addTrainingData(training_data)
modelConf.addEpochs(30)

s=Training()
model=s.buildModel(modelConf)

doc = model('#pls. stop the check# 2870822 Stoped / Reissued')
 
#Check # 2870820 Stoped / Reissued
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)


#Display Entities
#displacy.serve(doc,style="ent")

