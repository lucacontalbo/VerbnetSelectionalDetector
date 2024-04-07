import os
import pandas as pd
import json
import string

import nltk
nltk.download('verbnet3')
nltk.download('wordnet')

from nltk.corpus import verbnet
from nltk.corpus import wordnet as wn

from nltk.stem import WordNetLemmatizer
 
lemmatizer = WordNetLemmatizer()

"""print(lemmatizer.lemmatize("tends", pos="v"))

print(verbnet.vnclass("75"))

print(verbnet.classids(lemma=lemmatizer.lemmatize("tends", pos="v")))
print(verbnet.themroles(verbnet.classids(lemma="attack")[0]))
"""

print(wn.synsets("Hitler", pos="n")[0].hypernym_paths())
