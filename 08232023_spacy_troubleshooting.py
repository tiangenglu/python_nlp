#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:29:25 2023

@author: Tiangeng Lu

I've experienced errors when installing & importing spaCy 3.1.3 (the version used for datacamp).
The following works.
"""

import sys
print(sys.path)
sys.path.append("/Users/tiangeng/opt/anaconda3/pkgs/spacy-3.1.3-py39ha1f3e3e_0/lib/python3.9/site-packages")
sys.path=list(set(sys.path))
print(sys.path)
import spacy
print(spacy.__version__)
print(spacy.__file__)
nlp = spacy.load("en_core_web_sm")
