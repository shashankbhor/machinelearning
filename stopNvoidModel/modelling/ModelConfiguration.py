#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 20:31:48 2019
@author: Shashank 
Classification Modeling using spaCy - nlp 

"""

import spacy
import random
from spacy import displacy 

class ModelConfiguration:
    def __init__(self,modelConfigName):
        self.modelName=modelConfigName
        self.epochs=20
        self.inputPath=""
        self.outputPath=""
        self.trainingSet=""
        
    def addEpochs(self,epochs):
        epochs=epochs
        
    def addTrainingData(self,data):
        self.trainingSet=data
    
    def addInputPath(self,inputPath):
        self.inputPath=inputPath
        
    def parseTrainingDataCSV(self):
        if self.inputPath != "" and self.trainingSet == "":
            print('Need to parse input file ')
        
