#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:35:10 2019

@author: m_159834
"""

import spacy
import random
from spacy import displacy
import json
import pandas as pd
from pandas import DataFrame

text1='Frank works at Cognizant Technology Solutions. His insurance premium $2,870,820 is deducted at source. Last primium has been paid on 12/03/2019 by Check. The Check number is #2870820. This needs to Stoped/Reissued'
nlp_=spacy.load('en_core_web_sm')
doc_=nlp_(text1)
for ent in doc_.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

displacy.serve(doc_,style="ent")


#Set 1 - single entity
TRAIN_DATA = [('Check team Pls VOID Check #3018677?', [{'entities': [(27, 34, 'check_number')]}]), 
               ('Check Team Pls VOID Check #2983294  - Amount $24,7 ?', [{'entities': [(27, 34, 'check_number')]}]), 
                ('Pls VOID Check #2993435 amount $7101.00?', [{'entities': [(16, 23, 'check_number')]}]), 
                 ('please STOP PAY on check #2962623 Check #/ EFT?', [{'entities': [(26, 33, 'check_number')]}]), 
                  ('Check #2990276 Stopped / Reissued', [{'entities': [(7, 14, 'check_number')]}]),
                   ('Check # 3027262  has been voided?', [{'entities': [(8, 15, 'check_number')]}]),
                    ('A - Check #2949930 Stop Paid / ReIssued?', [{'entities': [(11, 18, 'check_number')]}]),
                     ('Stop check #2972221?', [{'entities': [(12, 19, 'check_number')]}]),
                      ('Check #2885046 stopped / reissued?', [{'entities': [(7, 14, 'check_number')]}]), 
                       ('Stop check#2873200', [{'entities': [(11, 18, 'check_number')]}]), 
                        ('Stop check#2920316', [{'entities': [(11, 18, 'check_number')]}]), 
                         ('Check # 2870820 Stoped / Reissued', [{'entities': [(8, 15, 'check_number')]}]
                          )]

#Set 2 - Multiple entity
training_data = [
        ('Check team Pls VOID Check #3018677?', [{'entities': [(27, 34, 'check_number'),(15, 19, 'check_instruction')]}]), 
               ('Check Team Pls VOID Check #2983294  - Amount $24,7 ?', [{'entities': [(27, 34, 'check_number'),(15, 19, 'check_instruction')]}]), 
                ('Pls VOID Check #2993435 amount $7101.00?', [{'entities': [(16, 23, 'check_number'),(4, 8, 'check_instruction')]}]),
                 ('please STOP PAY on check #2962623 Check #/ EFT?', [{'entities': [(26, 33, 'check_number'),(7, 11, 'check_instruction')]}]), 
                  ('Check #2990276 Stopped / Reissued', [{'entities': [(7, 14, 'check_number'),(25, 32, 'check_instruction')]}]),
                   ('Check # 3027262  has been voided?', [{'entities': [(8, 15, 'check_number'),(26, 30, 'check_instruction')]}]),
                    ('A - Check #2949930 Stop Paid / ReIssued?', [{'entities': [(11, 18, 'check_number'),(31, 38, 'check_instruction')]}]),
                     ('Stop check #2972221?', [{'entities': [(12, 19, 'check_number'),(1, 4, 'check_instruction')]}]),
                      ('Check #2885046 stopped / reissued?', [{'entities': [(7, 14, 'check_number'),(25, 32, 'check_instruction')]}]), 
                       ('Stop check#2873200', [{'entities': [(11, 18, 'check_number'),(1, 4, 'check_instruction')]}]), 
                        ('Stop check#2920316', [{'entities': [(11, 18, 'check_number'),(1, 4, 'check_instruction')]}]), 
                         ('Check # 2870820 Stoped / Reissued', [{'entities': [(8, 15, 'check_number')]}])
                          ]


training_data = [('Check team Pls VOID Check #3018677                 ,"  Check team Pls VOID Check # 3018677 amount $36,812.96  ;  ;  ;            Check #/ EFT #:   000003018677   Issue Date:   11/01/2017      Payable To:   Southeast Toyota Finance   Amount:   36,812.96 USD      ;   Thirty six thousand eight hundred twelve and 96/100     Mail To Name:   Southeast Toyota Finance   Bank:   USD Bank Of America     Address:   Lockbox #8881 PO Box: 8500 Philadelphia, PA 19178-8500 US                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               "',
  [{'entities': [(27, 35, 'check_number'),
     (99, 109, 'check_value'),
     (16, 20, 'instructions')]}]),
 ('"  tls, pls void and reissue for $34000.00.  Thanks.  ","  tls, pls void and reissue for $34000.00.; Thanks.  ;          Check #/ EFT #:   000002958328   Issue Date:   10/04/2017      Payable To:   Raymond W. Jr. and Kathleen F. Buehler Jr.   Amount:   34,550.26 USD      ;   Thirty four thousand five hundred fifty and 26/100     Mail To Name:   Raymond W. Jr. and Kathleen F. Buehler Jr.   Bank:   USD Bank Of America     Address:   111 Haverford Cir, Mount Lebanon, PA 15228-2381 US                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  "',
  [{'entities': [(12, 17, 'instructions'),
     (253, 262, 'check_value'),
     (139, 151, 'check_number')]}]),
 ('"  Check Team Pls VOID Check #2983294  - Amount $24,7 ","  Check Team Pls VOID Check #2983294; - Amount $24,716.00 USD   ;          Check #/ EFT #:   000002983294   Issue Date:   10/16/2017      Payable To:   Ally Financial   Amount:   24,716.00 USD      ;   Twenty four thousand seven hundred sixteen and 00/100     Mail To Name:   Ally Financial   Bank:   USD Bank Of America     Address:   PO Box 9001951, Louisville, KY 40290 US "',
  [{'entities': [(30, 38, 'check_number'),
     (105, 114, 'check_value'),
     (74, 79, 'check_number')]}]),
 ('Please Void Check                                  ,"  issued check with incorrect amount $13,654.84  please void check;   ;          Check #/ EFT #:   000002872436   Issue Date:   8/23/2017      Payable To:   Axel &amp; Noemi Gros   Amount:   13,654.84 USD      ;   Thirteen thousand six hundred fifty four and 84/100     Mail To Name:   Axel &amp; Noemi Gros   Bank:   USD Bank Of America     Address:   1100 Donegal Ln, Northbrook, IL 60062-4322 US   ;  ;  ;  please send me task when void complete, I will reissue check for correct amount $13,949.28 (revised offer ); Thank you   ;  ;  ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         "',
  [{'entities': [(7, 12, 'instructions'),
     (152, 164, 'check_number'),
     (244, 253, 'check_value')]}]),
 ('"  tls, pls stop pymt and reissue for $10025.10.  Ple ","  tls, pls stop pymt and reissue for $10025.10.; Please mail FL poa with the ck.; See;me for poa.;;Thx  ;  \'In Settlement of\' Collision total loss for 2005 Ford Expedition vin A29857 + tow reimbursement.  ;  MAIL TO  2973 RIVERLAND RD.  FT. LAUDERDALE, FL 33312  ;  PLEASE OVERNIGHT  ;          Check #/ EFT #:   000002955712   Issue Date:   10/03/2017      Payable To:   Todd H. Pitcairn   Amount:   9,747.00 USD      ;   Nine thousand seven hundred forty seven and 00/100     Mail To Name:   Todd H. Pitcairn   Bank:   USD Bank Of America     Address:   165 Township Line Rd, Jenkintown, PA 19046-3531 US"',
  [{'entities': [(12, 17, 'instructions'),
     (370, 382, 'check_number'),
     (458, 466, 'check_value')]}]),
 ('PLS VOID CHECK                                     ,"  CA SUPPORT PLS VOID CHECK 2968886 ISSUED 10/9/17 FOR $7605.00 THIS IS AN ERROR -PLS DO NOT LET IT LEAVE THE BUILDING   ;          Check #/ EFT #:   000002968886   Issue Date:   10/09/2017      Payable To:   Kathlen Olsen and Stephen Zammitti   Amount:   7,605.00 USD      ;   Seven thousand six hundred five and 00/100     Mail To Name:   Kathlen Olsen and Stephen Zammitti   Bank:   USD Bank Of America     Address:   226 Hillside, Leonia, NJ 07605 US                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          "',
  [{'entities': [(203, 215, 'check_number'),
     (309, 317, 'check_value'),
     (4, 9, 'instructions')]}]),
 ('Check team Pls VOID Check #2993435 amount $7101.00 ,"  Check team Pls VOID Check #2993435 amount $7101.00  ;  ;          Check #/ EFT #:   000002993435   Issue Date:   10/20/2017      Payable To:   Maria Noble   Amount:   7,101.00 USD      ;   Seven thousand one hundred one and 00/100     Mail To Name:   Maria Noble   Bank:   USD Bank Of America     Address:   1635 Powers Ridge Place NW, Atlanta, GA 30327 US "',
  [{'entities': [(15, 20, 'instructions'),
     (139, 151, 'check_number'),
     (222, 230, 'check_value')]}]),
 ('"  Check team, pls VOID                               ","  Check team, pls VOID Ck# 3004902 - $7,034.74  ;          Check #/ EFT #:   000003004902   Issue Date:   10/25/2017      Payable To:   Sheila M. Bernson   Amount:   7,034.74 USD      ;   Seven thousand thirty four and 74/100     Mail To Name:   Sheila Bernson   Bank:   USD Bank Of America     Address:   7 Alder Way, Armonk, NY 10504-1337 US "',
  [{'entities': [(134, 147, 'check_number'),
     (223, 231, 'check_value'),
     (19, 24, 'instructions')]}]),
 ('"  tls, pls void ck and reissue for $6774.74.  Thanks ","  tls, pls void ck and reissue for $6774.74.; Thanks.  ;  note: (deduct 4 days storage @ $65/day)  ;          Check #/ EFT #:   000003004902   Issue Date:   10/25/2017      Payable To:   Sheila M. Bernson   Amount:   7,034.74 USD      ;   Seven thousand thirty four and 74/100     Mail To Name:   Sheila Bernson   Bank:   USD Bank Of America     Address:   7 Alder Way, Armonk, NY 10504-1337 US                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  "',
  [{'entities': [(12, 17, 'instructions'),
     (185, 197, 'check_number'),
     (274, 283, 'check_value')]}]),
 ('STOP Pay And Reissue Request                       ,"    STOP PAY And Reissue Request (Please see specific instructions below - Amount is different - Thank you).  ;          Check #/ EFT #:   000002918455   Issue Date:   9/15/2017     Payable To:   Nexus Business Solutions, Llc   Amount:   6,855.15 USD     ;   Six thousand eight hundred fifty five and 15/100     Mail To Name:   Nexus Business Solutions, Llc   Bank:   USD Bank Of America     Address:   100 W Big Beaver Rd, Apt/Suite: 200 Troy, MI 48084-5283 US   Please Reissue $6,293.53 to: General Motors LLC P.O. Box 43830 Detroit, MI 48243-1199 PLEASE REFERENCE THE GM FILE NUMBER, 9393888, ON THE CHECK.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  "',
  [{'entities': [(0, 5, 'instructions'),
     (192, 204, 'check_number'),
     (291, 299, 'check_value')]}]),
 ('ACS - Pls Stop pay and Reissue                     ,"  Pls Stop Pay and Re-issue for Correct Amount of $4114.90   ;  ;  ;          Financial Type:   Payment  ;  ;  Check Amt:   5,059.29 USD    Group Status:   Posted   Issue Date:   9/15/2017  Payable To:   Brannings Princeton Auto Body     Performer:   Lionie Abshire   ;  Payment Method:   Regular       Check Number:   000002919697   Requested Date:   9/15/2017  ;  ;  ;        Financial Ctgy /  ;  Claimant/Line /  ;  Svc/Benefit   ;  Start Date   ;  Check Detail   ;  Amt   ;        LOSS PAID   ;  01-0 Matthew &amp; Alene Frankel / First Party Physical Damage / Collision - First Party Phys Damage / Collision "',
  [{'entities': [(10, 15, 'instructions'),
     (177, 185, 'check_value'),
     (372, 384, 'check_number')]}])]

    
for txt,entlst in training_data:
    lst=entlst[0].get('entities')
    for l in lst:
        a,b,c = l
        print(c,txt[a:b])
    



#Set 3 - Variable entity
data=pd.read_json("./data/stopnvoid_full.json",lines=True)
trainingData=DataFrame()
train_lbls=data['labels'].values
training_data=[]
df=DataFrame()
for index,txt in enumerate(train_lbls):
    entlst=data['labels'][index]
    #entlst_=[{'entities': [tuple(ent_x for ent_x in ent_) for ent_x in ent_] for ent_ in entlst}]
    entlst=[{'entities':[tuple(oneset) for oneset in entlst]}]
    #print(entlst)
    #entlst_=[{'entities': [tuple(ent_x for ent_x in ent_) for ent_x in ent_] for ent_ in entlst}]
    #entlst_=[{'entities': [ tuple([ent_x for ent_x in ent_])] } for ent_ in entlst]
    training_data.append((data['text'][index],entlst))
    if(index==10):
        break
    
TRAIN_DATA=training_data    

def train_spacy(data,iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    for  _,tagged_set in TRAIN_DATA:
        for tagged_item in tagged_set:
            for item in tagged_item.get('entities'):
                ner.add_label(item[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                #print(annotations)
                nlp.update(
                    #[text]*len(ner.labels),  # batch of texts
                    [text]*len(annotations),
                    annotations,  # batch of annotations
                    drop=0.3,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses) 
    return nlp

prdnlp = train_spacy(TRAIN_DATA, 10)

# Save our trained Model
#modelfile = input("Enter your Model Name: ")
#prdnlp.to_disk('modelfile')

#Test your text
#test_text = input("Enter your testing text: ")
text1='Shashank works at Cognizant Technology Solutions. His insurance premium $2,870,820 is deducted at source. Last primium has been paid on 12/03/2019 by Check. The Check number is #2870820. This needs to Stoped/Reissued'
nlp_=spacy.load('en_core_web_sm')
doc_=nlp_(text1)
doc = prdnlp('Shashank works at Cognizant. His insurance premium $2,870,820 is deducted at source. Last primium has been paid on 12/03/2019 by Check. The Check number#2870820. This needs to Stoped/Reissued')
doc = prdnlp('Check team Pls VOID Check #3018677?')
doc = prdnlp('Check team Pls VOID Check #3018677 ')
             
doc = prdnlp( 'Check team Pls VOID Check #3018677                 ,"  Check team Pls VOID Check # 3018677 amount $36,812.96  ;  ;  ;            Check #/ EFT #:   000003018677   Issue Date:   11/01/2017      Payable To:   Southeast Toyota Finance   Amount:   36,812.96 USD      ;   Thirty six thousand eight hundred twelve and 96/100     Mail To Name:   Southeast Toyota Finance   Bank:   USD Bank Of America     Address:   Lockbox #8881 PO Box: 8500 Philadelphia, PA 19178-8500 US ')

print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
           
             '''            
from spacy.lang.en import English             
nlp_a= spacy.load('en_core_web_sm')
doc=nlp_a(text1)
   '''                   
#Check # 2870820 Stoped / Reissued

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

    
displacy.serve(doc,style="ent")