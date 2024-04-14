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

import networkx as nx
from selrestr import dg
 
lemmatizer = WordNetLemmatizer()

"""print(lemmatizer.lemmatize("tends", pos="v"))

print(verbnet.vnclass("75"))

print(verbnet.classids(lemma=lemmatizer.lemmatize("tends", pos="v")))
print(verbnet.themroles(verbnet.classids(lemma="attack")[0]))
"""

def get_descendants(synset):
    descendants = set()
    for hyponym in synset.hyponyms():
        descendants.add(hyponym)
        descendants |= get_descendants(hyponym)
    
    root = wn.synset("entity.n.01")
    descendants_extended = set()
    for descendant in descendants:
        descendants_extended.add(descendant)
        if descendant not in dg:
            continue
        for path in nx.all_simple_paths(dg, source=descendant, target=root):
            for syn in path:
                descendants_extended.add(syn)
    return descendants


typ = wn.synset("living_thing.n.01")
for t in typ:
    print(t)
    print(t.definition())
    for path in t.hypernym_paths():
        print(path)
    print("------")

"""
typ2 = wn.synsets("natural_event", pos="n")[0]
print(typ2)
desc = get_descendants(typ2)
print(desc)
print(typ[0] in desc)"""
"""typ2 = wn.synsets("natural_phenomenon", pos="n")[0]
desc = get_descendants(typ2)

print(desc)"""
