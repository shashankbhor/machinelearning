# -*- coding: utf-8 -*-
"""
The program is build to build model for information extraction 
from Stop & Void instruction. 
Extraction Fields :
    Check Amount - Number fields (may appear two numbers but need corretc amt)
    Issue instruction - Stop or Void or Re-issue
    Check number - Check number to be voided 
"""

import spacy

nlp=spacy.load('en')

doc=nlp('My name is Shashank. I work at Google')
    
for entity in doc.ents:
    print(entity.text, entity.label_)
    
lst_ent=[ent.text for ent in doc.ents]

