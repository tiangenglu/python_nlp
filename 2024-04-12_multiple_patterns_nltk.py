#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 2024

@author: Tiangeng Lu

nltk, matching one or more patterns from a list
"""
import sys
print(sys.version) # 3.9.18 (main, Sep 11 2023, 08:38:23)
import nltk
print(nltk.__version__) # 3.8.1

# stop-words
from nltk.corpus import stopwords
stop_words = stopwords.words(['english'])

# How many stopwords are in the current list? 179
print(f'How many stopwords are in the current list? {len(stop_words)}')
print(f"{stop_words}")
"""['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
"you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o',
 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
 "weren't", 'won', "won't", 'wouldn', "wouldn't"]"""

# create a few sentences
sentences = ["Updates to OMB's Race/Ethnicity Standards",
             """OMB published the results of its review of SPD 15 and issued
             updated standards for collecting and reporting race and ethnicity
             data across federal agencies""", 
             "Undercounts and Overcounts of Young Children in the 2020 Census", 
             "2023 Population Estimates by Age and Sex", 
             "In the coming months, the Census Bureau will release additional population estimates."]

# target patterns
patterns = ['race','age','sex','estimate']

# tokenize
from nltk.tokenize import word_tokenize
# lemmatize: to the base form of a word
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()

# tokenize the sentences
tokenized_sentences = []
tokenized_sentences_joined = []

for one_sentence in sentences:
    token_list = [lm.lemmatize(word) for word in word_tokenize(one_sentence) 
                  if (word.isalnum() and word not in stop_words)]
    tokenized_sentences.append(token_list)
    tokenized_sentences_joined.append(' '.join(map(str, token_list)))

print(tokenized_sentences)
"""
[['Updates', 'OMB', 'Standards'],
 ['OMB', 'published', 'result', 'review', 'SPD', '15', 'issued', 'updated', 'standard', 'collecting', 'reporting', 'race', 'ethnicity', 'data', 'across', 'federal', 'agency'],
 ['Undercounts', 'Overcounts', 'Young', 'Children', '2020', 'Census'],
 ['2023', 'Population', 'Estimates', 'Age', 'Sex'],
 ['In', 'coming', 'month', 'Census', 'Bureau', 'release', 'additional', 'population', 'estimate']]
"""
print(tokenized_sentences_joined)
"""
['Updates OMB Standards',
 'OMB published result review SPD 15 issued updated standard collecting reporting race ethnicity data across federal agency',
 'Undercounts Overcounts Young Children 2020 Census',
 '2023 Population Estimates Age Sex',
 'In coming month Census Bureau release additional population estimate']
"""

# match multiple patterns
## loop for tokenized sentences
for token_sen in tokenized_sentences:
    if any([pat for pat in token_sen if pat in patterns]):
        print(token_sen)
"""
['OMB', 'published', 'result', 'review', 'SPD', '15', 'issued', 'updated', 'standard', 'collecting', 'reporting', 'race', 'ethnicity', 'data', 'across', 'federal', 'agency']
['In', 'coming', 'month', 'Census', 'Bureau', 'release', 'additional', 'population', 'estimate']
"""

## nested list comprehensions
[(pat,token_sen) for pat in patterns for token_sen in tokenized_sentences if pat in token_sen]    
# not outputing multiple patterns in a sentence
"""
[('race',
  ['OMB',
   'published',
   'result',
   'review',
   'SPD',
   '15',
   'issued',
   'updated',
   'standard',
   'collecting',
   'reporting',
   'race',
   'ethnicity',
   'data',
   'across',
   'federal',
   'agency']),
 ('estimate',
  ['In',
   'coming',
   'month',
   'Census',
   'Bureau',
   'release',
   'additional',
   'population',
   'estimate'])]
"""

[sen for sen in tokenized_sentences for pat in patterns if pat in sen]    

# loop for joined tokens as a sentence
for pat in patterns:
    for a_sentence in tokenized_sentences_joined:
        if pat in a_sentence:
            print(f'"{pat.upper()}" was detected in: \n"{a_sentence}"\n')
"""
"RACE" was detected in: 
"OMB published result review SPD 15 issued updated standard collecting reporting race ethnicity data across federal agency"

"AGE" was detected in: 
"OMB published result review SPD 15 issued updated standard collecting reporting race ethnicity data across federal agency"

"ESTIMATE" was detected in: 
"In coming month Census Bureau release additional population estimate"
"""

# nested list comprehensions (recommended)
[(pat,sen) for sen in tokenized_sentences_joined for pat in patterns if pat in sen]    
"""
[('race',
  'OMB published result review SPD 15 issued updated standard collecting reporting race ethnicity data across federal agency'),
 ('age',
  'OMB published result review SPD 15 issued updated standard collecting reporting race ethnicity data across federal agency'),
 ('estimate',
  'In coming month Census Bureau release additional population estimate')]
"""  
    
    
    
    