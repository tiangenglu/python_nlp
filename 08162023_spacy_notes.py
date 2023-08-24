#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 22:38:09 2023

@author: Tiangeng Lu
"""
#!python -m spacy download en_core_web_sm
#!python -m spacy download en_core_web_md
import pandas as pd
raw = pd.read_excel('mig_analysis.xlsx', sheet_name = 'data')
raw = raw.fillna('')
theme = pd.read_excel('mig_analysis.xlsx', sheet_name = 'txt_binary')
raw['raw_text'] = raw['Article Title'].str.cat(raw['Abstract'], sep = '. ')
import sys
sys.path.append("/Users/tiangeng/opt/anaconda3/pkgs/spacy-3.1.3-py39ha1f3e3e_0/lib/python3.9/site-packages")
sys.path=list(set(sys.path))
import spacy
print(spacy.__version__)
print(spacy.__file__)
# nlp object converts text into a Doc object (container) to store processed text
nlp = spacy.load("en_core_web_sm")

sample_doc = nlp(raw['raw_text'][100])
# see tokenized words
print([token.text for token in sample_doc])
# see sentences
print([sent.text for sent in sample_doc.sents])
# lemma
print([(token.text, token.lemma_) for token in sample_doc])

# pos tagging
print([(token.text, token.pos_, spacy.explain(token.pos_)) for token in sample_doc])

# NER, GPE = Geo-political entity, countries, cities
# .ents for named entities, .label_ for entity label
print([(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in sample_doc.ents])

# access entity types of each token, get location names
print([(token.text, token.ent_type_) for token in sample_doc if token.ent_type_ == 'GPE'])

# POS tagging
sample_sent = 'Control of settlement is more difficult and problematic, involving the role of immigrants in European society.'
print([(token.text, token.pos_, spacy.explain(token.pos_)) for token in nlp(sample_sent)])
# Word-sense disambiguation (WSD) is the problem of deciding in which sense a word is used in a sentence

# Dependency parsing
# Explores a sentence syntax, links between two tokens, results in a tree
# Dependency label describes the type of syntactic relation between two tokens
# .dep_ attribute to access the dependency label of a token
print([(token.text, token.dep_, spacy.explain(token.dep_)) for token in nlp(sample_sent)])

# Word vectors: a pre-defined number of dimensions; considers word frequencies and the presence of other words in similar contexts.
# Word vectors not functional
nlp = spacy.load("en_core_web_md") # add this for word.vector
# Multiple approaches to produce word vectors: word2vec, Glove, fastText, transformer-based architectures
# spaCy vocabulary is a part of many spaCy models, 'en_core_web_md' has 300-dimensional vectors for 20,000 words.
print(nlp.meta['vectors'])
# nlp.vocab, to access vocabulary(vocab class)
# nlp.vocab.strings, to access word IDs in a vocabulary

immigrant_id = nlp.vocab.strings['immigrant']
print(immigrant_id)
print(nlp.vocab.vectors[immigrant_id])
migrant_id = nlp.vocab.strings['migrant']
print(migrant_id)
# KeyError: '[E058] Could not retrieve vector for key 8040526675107974556.'
print(nlp.vocab.vectors[migrant_id])

employment_id = nlp.vocab.strings['employment']
print(employment_id)
# KeyError: '[E058] Could not retrieve vector for key 10954873364127974648.'
print(nlp.vocab.vectors[employment_id])

print(nlp.vocab.vectors[nlp.vocab.strings["like"]])
print(spacy.__version__)


md_nlp = spacy.load("en_core_web_md")
print("Number of words: ", md_nlp.meta["vectors"]["vectors"], "\n")
print("Dimension of word vectors: ", md_nlp.meta["vectors"]["width"])


words = ["like", "love", "awesome", "horrible", "hate"]
nlp = spacy.load("en_core_web_md")
# IDs of all the given words
ids = [nlp.vocab.strings[w] for w in words]
nlp.vocab.vectors[ids[1]]
# Store the first ten elements of the word vectors for each word
# KeyError: '[E058] Could not retrieve vector for key 18194338103975822726.'
word_vectors = [nlp.vocab.vectors[i][:10] for i in ids]
# Print the first ten elements of the first word vector
print(word_vectors[0])

# Word vectors visualization, word vectors allow to understand how words are grouped
# PCA projects word vectors into a two-dimensional space
# word vectors visualization
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
nlp=spacy.load("en_core_web_md")
# extract word vectors for a given list of words and stck them vertically
words = ["wonderful", "horrible", "apple", "banana", "orange", "watermelon", "dog", "cat"]
# FutureWarning: arrays to stack must be passed as a "sequence" type such as list or tuple. Support for non-sequence iterables such as generators is deprecated as of NumPy 1.16 and will raise an error in the future.
word_vectors = np.vstack([nlp.vocab.vectors[nlp.vocab.strings[w]]] for w in words)
# Extract two principal components using PCA
pca = PCA(n_components = 2)
word_vectors_transformed = pca.fit_transform(word_vectors)

# visualize the scatter plot of transformed vectors
plt.figure(figsize = (10, 8))
plt.scatter(word_vectors_transformed[:, 0], word_vectors_transformed[:, 1])
for word, coord in zip(words, word_vectors_transformed):
    x, y = coord
    plt.text(x, y, word, size = 10)
plt.show()    

# analogies and vector operations
# a sematic relationship between a pair of words
# word embeddings generate analogies such as gender and tense: queen-woman+man = king

# Similar words in a vocabulary
# spacy find semantically similar terms to a given term
import numpy as np
import spacy
nlp = spacy.load("en_core_web_md")
word = "skill"
most_similar_words = nlp.vocab.vectors.most_similar(
    np.asarray([nlp.vocab.vectors[nlp.vocab.strings[word]]]), n = 5)
words = [nlp.vocab.strings[w] for w in most_similar_words[0][0]]
print(words)

# Semantic similarity method: process of analyzing texts to identify similarities, categorizes texts into predefined categories or detect relevant texts
# similarity score measures how similar two pieces of text are
# Similarity score: to measure similarity use Cosine similarity and word vectors

# Token similarity: spacy calculates similarity scores between Token objects
nlp = spacy.load("en_core_web_md")
doc1 = nlp("We eat pizza")
doc2 = nlp("We like to eat pasta")
token1 = doc1[2]
token2 = doc2[4]
print(f"Similarity between {token1} and {token2} = ", round(token1.similarity(token2), 3)) # 0.737

# Span similarity
# spacy calculates semantic similarity of two given Span objects

span1 = doc1[1:]
print(span1)
span2 = doc2[1:]
print(span2)
print(f"Similarity between \"{span1}\" and \"{span2}\" = ", round(span1.similarity(span2), 3)) # 0.862

print(doc1[1:])
print(doc2[3:])
print(f"Similarity between \"{doc1[1:]}\" and \"{doc2[3:]}\" = ", round(doc1[1:].similarity(doc2[3:]), 3)) # 0.908

# Doc similarity
# spacy calculates the similarity scores between two documents
nlp = spacy.load("en_core_web_md")
doc1 = nlp("I like to play basketball")
doc2 = nlp("I love to play basketball")
print("Similarity score :", round(doc1.similarity(doc2), 3))
# compare two abstracts, randomly compared two pairs of abstracts, both above 0.95, seems to be extremely high
abstract1 = nlp(raw['raw_text'][200])
abstract2 = nlp(raw['raw_text'][1500])
print("Similarity between two abstracts :", round(abstract1.similarity(abstract2), 3))

# Sentence dimilarity
# spacy finds relevant content to a given keyword
# finding similar customer questions to the word culture
sentences = nlp("Purpose - The purpose of this paper is to explore the capital-based benefits which arise when acculturating immigrants perform touristic practices, and how these shape their tourism and migration experiences. \
                Design/methodology/approach - Grounded in consumer culture theory, this paper draws on theories of capital consumption to inform a hermeneutic analysis of multi-modal depth interviews with Southeast Asian skilled migrants in New Zealand.\
                Findings - Domestic touristic practices offer three types of capital-based benefits, enabling consumers to index economic capital, accrue social capital and index cultural capital.")

keyword = nlp("culture")
# Similarity score with 1:  0.56068
# Similarity score with 2:  0.6036
# Similarity score with 3:  0.55892
for i, sentence in enumerate(sentences.sents):
    print(f"Similarity score with {i+1}: ", round(sentence.similarity(keyword),5))


#spaCy pipelines
# spaCy first tokenizes the text to produce a Doc object
# The Doc is processed in several different steps of processing pipeline
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(raw['raw_text'][100])
# A pipeline is a sequence of pipes, or actors on data
# A spaCy NER pipeline: tokenization, named entity identification, named entity classification
# input text--> tokenizer--> entityruler --> entitylinker --> Doc with annotated entities
print([ent.text for ent in doc.ents]) # returned named entities, EU, European Union, Spanish, Spain, non-European

# adding pipes
# sentencizer: spacy pipeline component for setence segmentation
import time
text = " ".join(["this is a test sentence"] * 10000)
en_core_sm_nlp = spacy.load("en_core_web_sm")
start_time = time.time()
doc = en_core_sm_nlp(text)
# calculate processing time
print(f"Finished processing with en_core_web_sm model in {round((time.time() - start_time)/60.0, 5)} minutes")
# Finished processing with en_core_web_sm model in 0.14898 minutes
# Reason of slowness: when calling an existing spaCy model on a text, the whole NLP pipeline will be activated
# each pipe from NER to dependency parsing will run on te text. This increases the use of computational time by 100 times.

# Create a blank model and add a sentencizer pipe
# reduce time with blank sentencizer
blank_nlp = spacy.blank("en")
blank_nlp.add_pipe("sentencizer")
start_time = time.time()
doc = blank_nlp(text)
print(f"Finished processing with blank model in {round( (time.time() - start_time)/60.0, 5)} minutes")
# Finished processing with blank model in 0.0004 minutes
# Why faster: only intended pipeline component, sentence segmentation will run on the given documents.

# Analyzing pipeline components
# nlp.analyze_pipes() analyzes a spaCy pipeline to determine
# Attributes that pipeline components set
# Scores a component produces during training
# Presence of all required attributes
nlp = spacy.load("en_core_web_sm")
analysis = nlp.analyze_pipes(pretty = True)

# add pipeline components
# Load a blank spaCy English model
nlp = spacy.blank("en")
# Add tagger and entity_linker pipeline components
nlp.add_pipe("ner")
nlp.add_pipe('sentencizer')
nlp.add_pipe("tagger")
nlp.add_pipe("entity_linker") # 'entity_linker' requirements not met: doc.ents, doc.sents,
# Analyze the pipeline
analysis = nlp.analyze_pipes(pretty=True) # ✔ No problems found.


# spaCy EntityRuler
# EntityRuler adds name-entities to a Doc container
# It can be used on its own or combined with EntityRecognizer

# Phrase entity patterns for exact string matches (string)
# {"label": "ORG", "pattern": "Microsoft"}

# Token entity patterns with one dictionary describing one token (list)
# {"label": "GPE", "pattern": [{"LOWER": "san"}, {"LOWER": "francisco"}]}

# Adding entityruler to spacy pipeline
# Using `.add_pipe()` method
nlp = spacy.blank("en")
entity_ruler = nlp.add_pipe("entity_ruler")
# define patterns
patterns = [{"label": "ORG", "pattern": "Microsoft"},\
            {"label": "GPE", "pattern": [{"LOWER": "san"}, {"LOWER": "francisco"}]}]
# add to pipeline    
entity_ruler.add_patterns(patterns)
# `.ents` store the results of an EntityLinker component
doc = nlp("Microsoft is hiring software developer in San Francisco.")
print([(ent.text, ent.label_) for ent in doc.ents])

# EntityRuler in action
# Integrates with `spacy` pipeline components
# Enhances the named-entity recognizer
# `spaCy` model without EntityRuler
nlp = spacy.load("en_core_web_sm")
doc = nlp("Manhattan associates is a company in the U.S.")
# Manhattan is recognized as a location, Manhattan associates is not recognized as an organization
print([(ent.text, ent.label_) for ent in doc.ents])

# `EntityRuler` added after existing `ner` component
# only works if they DON'T overlap
nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", after = 'ner') # "after" is the problem
patterns = [ { "label": "ORG", "pattern": [{"lower": "manhattan"}, {"lower": "associates"}] } ]
ruler.add_patterns(patterns)
doc = nlp("Manhattan associates is a company in the U.S.")
print([ (ent.text, ent.label_) for ent in doc.ents ])

# Solution is to put `entity_ruler` BEFORE
nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before = 'ner') # "after" is the problem
patterns = [ { "label": "ORG", "pattern": [{"lower": "manhattan"}, {"lower": "associates"}] } ]
ruler.add_patterns(patterns)
doc = nlp("Manhattan associates is a company in the U.S.")
print([ (ent.text, ent.label_) for ent in doc.ents ])

example_text = "This is a confection. In this case Filberts.\
    And it is cut into tiny squares.\
        This is the treat that seduces Edmund into selling out his Brother and Sisters to the Witch."
nlp = spacy.load("en_core_web_md")

# Print a list of tuples of entities text and types in the example_text
print("Before EntityRuler: ", [(ent.text, ent.label_) for ent in nlp(example_text).ents], "\n")        

# Define pattern to add a label PERSON for lower cased sisters and brother entities
patterns = [{"label": "PERSON", "pattern": [{"lower": "sisters"}]},
            {"label": "PERSON", "pattern": [{"lower": "brother"}]}]

# Add an EntityRuler component and add the patterns to the ruler
ruler = nlp.add_pipe("entity_ruler")
ruler.add_patterns(patterns)

# Print a list of tuples of entities text and types
print("After EntityRuler: ", [(ent.text, ent.label_) for ent in nlp(example_text).ents])

# What is RegEx?
# Rule-based information extraction (IR) is useful for many NLP tasks
# Regular expression (RegEx) is used with complex string matching patterns
# RegEx finds and retrieves patterns or replace matching patterns

# RegEx strengths and weaknesses
# Pros: enables writing robust rules to retrieve information; 
# can allow us to find many types of variance in strings
# runs fast, supported by many programming languages

# Cons
# Syntax is challenging for beginners
# Requires knowledge of all the ways a pattern may be mentioned in texts

# RegEx in Python
# Python comes prepackaged with a RegEx library, `re`
# The first step in using `re` package is to define a pattern
# The resulting pattern is used to find matching content
import re
# (\d) meta character 0-9, {3} how many occurrence
pattern = r"((\d){3}-(\d){3}-(\d){4})"
# pattern = r"\((\d){3}\)-(\d){3}-(\d){4}" # for (425)-123-4567
text = "Our phone number is 917-588-6010 and their phone number is 425-123-4567."
# find any matches: `.finditer()` from `re`
iter_matches = re.finditer(pattern, text)
for match in iter_matches:
    start_char = match.start()
    end_char = match.end()
    print("Start character: ", start_char,\
          "| End character: ", end_char,\
              "| Matching text: ", text[start_char:end_char])

# RegEx in spaCy
# RegEx in three pipeline components: `Matcher`, `PhraseMatcher` and `EntityRuler`
# Use `EntityRuler` to find phone number
text = "Our phone number is 917-588-6010 and their phone number is 425-123-4567."
# `EntityRuler` pattern construction with 5 small dictionary, "ORTH" = exact match of a string
patterns = [{"label": "PHONE_NUMBER",\
             "pattern": [{"SHAPE": "ddd"},\
                         {"ORTH": "-"},\
                         {"SHAPE": "ddd"},\
                         {"ORTH": "-"},\
                         {"SHAPE": "dddd"}]
                 }]
nlp = spacy.load("en_core_web_md") # re-define nlp to avoid errors: entity_ruler' already exists in pipeline. 
ruler = nlp.add_pipe("entity_ruler", before = 'ner') # I added before = "ner", otherwise, doesn't work
ruler.add_patterns(patterns)
doc = nlp(text)
print([(ent.text, ent.label_) for ent in doc.ents])

# Exercise
text = "Our phone number is 4251234567."
# Define a pattern to match phone numbers
patterns = [{"label": "PHONE_NUMBERS", "pattern": [{"TEXT": {"REGEX": "(\d){10}"}}]}]
# Load a blank model and add an EntityRuler
nlp = spacy.blank("en")
ruler = nlp.add_pipe("entity_ruler")
# Add the compiled patterns to the EntityRuler
ruler.add_patterns(patterns)
# Print the tuple of entities texts and types for the given text
doc = nlp(text)
print([(ent.text, ent.label_) for ent in doc.ents])

# spaCy Matcher and PhraseMatcher
# RegEx patterns can be complex, difficult to read and debug
# spaCy provides a readable and production-level alternative, the `Matcher` class
# The matcher class can match predefined rules to a sequence of tokens ni Doc containers.
import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")
doc = nlp("Good morning, this is our first day on campus.")
# new syntax
matcher = Matcher(nlp.vocab)

# define a pattern to match lower cased good and morning by defining a list with two key value pairs.
pattern = [{"LOWER": "good"}, {"LOWER": "morning"}]
# add this pattern with a custom name
matcher.add("morning_greeting", [pattern])
# run the matcher on the Doc container
matches = matcher(doc)
for match_id, start, end in matches:
    print("Start token: ", start, "| End token: ", end, "| Matched text: ", doc[start:end].text)

# Matcher extended syntax support
# Allows operators in defining the matching patterns
# Similar operators to Python's in, not in, and comparison operators
# Using `IN` operator to match both `good morning` and `good evening`
doc = nlp("Good morning and good evening.")
matcher = Matcher(nlp.vocab)
pattern = [{"LOWER": "good"}, {"LOWER": {"IN": ["morning", "evening"]}}] # `IN`
matcher.add("morning_greeting",[pattern])
matches = matcher(doc)
# output of matching using `IN` operator
for match_id, start, end in matches:
    print("Start token: ", start, " | End token: ", end, " | Matched text: ", doc[start:end].text)

# PhraseMatcher class matches a long list of phrases in a given text, 
# Enhance Matcher, because it is hand-crafted, need to be coded individually
# PhraseMatcher to match a list of EXACT values
from spacy.matcher import PhraseMatcher
nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab)
terms = ["Bill Gates", "John Smith"]
# PhraseMatcher outputs include start and end token indices of the matched pattern
patterns = [nlp.make_doc(term) for term in terms] # all elements in a list
matcher.add("PeopleOfInterest", patterns)

doc = nlp("Bill Gates met John Smith for an important discussion regarding importance of AI.")
matches = matcher(doc)
for match_id, start, end in matches:
    print("Start token: ", start, " | End token: ", end, " | Matched text: ", doc[start:end].text)

# use `attr` argument of the PhraseMatcher class
matcher = PhraseMatcher(nlp.vocab, attr = "LOWER")
terms = ["Government", "Investment"]
patterns = [nlp.make_doc(term) for term in terms]
matcher.add("InvestmentTerms", patterns)
doc = nlp("It was interesting to the investment division of the government.")
matchers = matcher(doc)
for match_id, start, end in matchers:
    print("Start token: ", start, " | End token: ", end, " | Matched text: ", doc[start:end].text)

# PhraseMatcher to match patterns of a given shape
matcher = PhraseMatcher(nlp.vocab, attr = "SHAPE")
terms = ["110.0.0.0", "101.243.0.0"]
patterns = [nlp.make_doc(term) for term in terms]
matcher.add("IPAddress", patterns)
doc = nlp("The tracked IP address was 234.130.0.0")
matchers = matcher(doc)
for match_id, start, end in matchers:
    print("Start token: ", start, " | End token: ", end, " | Matched text: ", doc[start:end].text)


# Why train spaCy models?
# Better results on specific domain
# Essential for domain specific text classification

# Before training, ask the following questions
# Do spaCy models perform well enough on our data?
# Does our domain include many labels that are absent in spaCy models?

# Models performance on specific data
import spacy
nlp = spacy.load("en_core_web_sm")
text = "The car was navigating to the Oxford Street."
doc = nlp(text)
print([(ent.text, ent.label_) for ent in doc.ents])
# Does our domain include many labels that are absent in spaCy models?
# Output labels in spaCy models
# Collect our domain specific data
# annotate our data
# determine to update an existing model or train a model from scratch


# AttributeError: 'spacy.tokens.token.Token' object has no attribute 'ents'
jumbo_text = "Product arrived labeled as Jumbo Salted Peanuts., Not sure if the product was labeled as Jumbo."
documents = nlp(jumbo_text)
# Append a tuple of (entities text, entities label) if Jumbo is in the entity
target_entities = []
for doc in documents:
  # AttributeError: 'spacy.tokens.token.Token' object has no attribute 'ents'
  target_entities.extend([(ent.text, ent.label_) for ent in doc.ents if "Jumbo" in ent.text])
print(target_entities)

# Append True to the correct_labels list if the entity label is `PRODUCT`
correct_labels = []
for target in target_entities:
  if target[1] == "PRODUCT":
    correct_labels.append(True)
  else:
    correct_labels.append(False)
print(correct_labels)

# Training steps
# Annotate and prepare input data
# Initialize the model weight
# Predict a few examples with the current weights
# Compare prediction with correct answers
# Use optimizer to calculate weights that improve model performance
# Update weights slightly
# Go back to step 3

# Annotating and preparing data
# First step is to prepare training data in required format
# After collecting data, annotate it
# Annotation means labeling the intent, entities, etc.
annotated_data = {
    "sentence": "An antiviral drugs used against influenza is neuraminidase inhibitors.", \
        "entities": {
            "label": "Medicine",
            "value": "neuraminidase inhibitors"}}
# another ecample of annotated data
annotated_data = {
    "sentence": "bill Gates visited the SFO Airport.",  
    "entities": [{"label": "PERSON", "value": "Gill Gates"},
        {"label": "LOC", "value": "SFO Airport"}]}

# spacy training data format
# Data annotation prepares training data for what we want the model to learn
# Annotated (training dataset) has to be stored as a dictionary
# Start and end characters are needed
# Three example pair includes a sentence as the first element
# Pair's second element is list of annotated entities and start and end characters
training_data = [
    ("I will visit you in Austin.", {"entities": [(20, 26, "GPE")]}),
    ("I'm going to Sam's house.", {"entities": [(13, 18, "PERSON"), (19, 24, "GPE")]}),
    ("I will go.", {"entities": []})
    ]

# Example object data for training
# Cannot feed the raw text directly to spacy
# need to create an example object for each training example

import spacy
from spacy.training import Example
nlp = spacy.load("en_core_web_sm")
doc = nlp("I will visit you in Austin.")
annotations = {"entities": [(20, 26, "GPE")]}
# convert doc container to an example object
example_sentence = Example.from_dict(doc, annotations)
# view attributes that are processed and stored at the example object
print(example_sentence.to_dict())

### Example
text = "A patient with chest pain had hyperthyroidism."
entity_1 = "chest pain"
entity_2 = "hyperthyroidism"

# Store annotated data information in the correct format
annotated_data = {"sentence": text, 
                  "entities": [{"label": "SYMPTOM", "value": entity_1},
                               {"label": "DISEASE", "value": entity_2}]}

# Extract start and end characters of each entity
# How to find start and end character
entity_1_start_char = text.find(entity_1) # .find()
entity_1_end_char = entity_1_start_char + len(entity_1)
entity_2_start_char = text.find(entity_2)
entity_2_end_char = entity_2_start_char + len(entity_2)

# Store the same input information in the proper format for training
training_data = [(text, {"entities": [(entity_1_start_char, entity_1_end_char, "SYMPTOM"), 
                                      (entity_2_start_char, entity_2_end_char, "DISEASE")]})]
print(training_data)

all_examples = []
# Iterate through text and annotations and convert text to a Doc container
for text, annotations in training_data:
  doc = nlp(text)
  
  # Create an Example object from the doc contianer and annotations
  example_sentence = Example.from_dict(doc, annotations)
  print(example_sentence.to_dict(), "\n")
  
  # Append the Example object to the list of all examples
  all_examples.append(example_sentence)
  
print("Number of formatted training data: ", len(all_examples))

# Training steps
# Annotate and prepare input data
# Disable other pipeline components
# Train a model for a few epochs
nlp = spacy.load("en_core_web_sm")
# Disable all pipeline components except NER
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
# ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer']
print(other_pipes)
nlp.disable_pipes(*other_pipes)
print(f"After disabe other pipes, the remaining pipe is/are: {nlp.pipe_names}")
# Model training procedure
# Go over the training set several times: one iteration is called an `epoch`
# In each epoch, update the weights of the model with a small number
# Optimizers are functions to update the model weights
optimizer = nlp.create_optimizer()
# optimizer = nlp.begin_training() # also used
# loss is a number of how bad a prediction is
import random
losses = {}
epochs = 2
for i in range(epochs):
    random.shuffle(training_data)
    for text, annotation in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotation)
        # optimizer object was defined from nlp.create_optimizer()
        nlp.update([example], sgd = optimizer, losses = losses)

# Save a trained NER model
ner = nlp.get_pipe("ner")
ner.to_disk("ner_model_name")

# Load the saved spacy model
ner = nlp.create_pipe("ner")
ner.from_disk("ner_model_name")
nlp.add_pipe(ner, "ner_model_name")

# Model for inference
# Use a saved model at inference
# Apply NER model and store tuples of (entity text, entity label)
doc = nlp(text)
entities = [(ent.text, ent.label_) for ent in doc.ents]

