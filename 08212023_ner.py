#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 23:43:39 2023

@author: Tiangeng Lu
"""
import time
import pandas as pd
raw = pd.read_excel('mig_analysis.xlsx', sheet_name = 'data')
raw = raw.fillna('')
theme = pd.read_excel('mig_analysis.xlsx', sheet_name = 'txt_binary')
raw['raw_text'] = raw['Article Title'].str.cat(raw['Abstract'], sep = '. ')

import spacy
nlp = spacy.load("en_core_web_sm")

gpe = [None]*len(raw['raw_text'])
# TypeError: 'English' object does not support item assignment
# AttributeError: 'English' object has no attribute 'append'
start_time = time.time()
gpe = []   
for i,text in enumerate(raw['raw_text']):
    doc = nlp(text)
    print(f"Processing item {i}")
    # leave duplicated items as they are and then process them later
    gpe.append([token.text for token in doc if token.ent_type_ == "GPE"])
print(f"Finished processing with blank model in {round( (time.time() - start_time)/60.0, 2)} minutes")
