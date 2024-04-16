import os
import urllib
import string
import spacy
import json
import networkx as nx
import nltk
import pickle
nltk.download('verbnet')
nltk.download('wordnet')

from tqdm import tqdm
from nltk.corpus import verbnet
from nltk.corpus import wordnet as wn

import pandas as pd
from selrestr import dg
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report

from rules.saved_vuaverb_train.rules import rules


def get_rules(selrestrs):
    rules = {}
    for verb, value in selrestrs.items():
        if verb not in rules.keys():
            rules[verb] = {"metaphor": {}, "literal":{}}
        
        for label, restrictions in value.items():
            for el in restrictions:
                if el[0] not in rules[verb][label].keys():
                    rules[verb][label][el[0]] = {}
                
                if el[1] not in rules[verb][label][el[0]].keys():
                    rules[verb][label][el[0]][el[1]] = {"frequency": 0}
                
                rules[verb][label][el[0]][el[1]]["frequency"] += 1
    for verb, value in rules.items():
        for label, restrictions in value.items():
            for arg_role, arg_role_value in restrictions.items():
                total_value = 0
                for synset, synset_value in arg_role_value.items():
                    total_value += synset_value["frequency"]

                for synset, synset_value in arg_role_value.items():
                    synset_value["conf"] = synset_value["frequency"] / total_value
    for verb, value in rules.items():
        for arg_role, arg_role_value in value["metaphor"].items():
            for synset, synset_value in arg_role_value.items():
                if arg_role in value["literal"].keys() and synset in value["literal"][arg_role]:
                    literal_freq = value["literal"][arg_role][synset]["frequency"]
                else:
                    literal_freq = 0
                synset_value["score"] = synset_value["frequency"] / (synset_value["frequency"] + literal_freq)

    return rules


vnselrestr_to_wn = {
    "animate": "living_thing",
    "vehicle": "vehicle",
    "machine": "machine",
    "person": "person",
    "animal": "animal",
    "natural": "natural_event",
    "body_part": "body_part",
    "int_control": "int_control",
    "plant": "plant",
    "food": "food",
    "comestible": "food",
    "garment": "clothing",
    "tool": "tool",
    "artifact": "artifact",
    "phys-obj": "physical_object",
    "concrete": "physical_entity",
    "time": "time",
    "state": "state",
    "abstract": "abstraction",
    "idea": "idea",
    "sound": "sound",
    "communication": "communication",
    "scalar": "scalar",
    "currency": "currency",
    "location": "location",
    "region": "region",
    "place": "place",
    "organization": "organization",
    "solid": "physical_object",
    "rigid": "rigidness",
    "pointy": "rigidness",
    "elongated": "rigidness",
    "nonrigid": "rigidness",
    "substance": "substance",
    "force": "natural_phenomenon",
    "human": "person",
    "plural": "entity",
}

map_to_semantic_type = {
    "Initial Location": "Location",
}

pronouns_to_living_thing = ["I", "You", "He", "She", "We", "They", "Who", "It", "Them", "Me", "Him", "Her"]

semantic_type_list = [
    "Actor",
    "Agent",
    "Asset",
    "Attribute",
    "Beneficiary",
    "Cause",
    "Destination",
    "Location",
    "Initial location"
    "Experiencer",
    "Instrument",
    "Material",
    "Product",
    "Patient",
    "Predicate",
    "Recipient",
    "Stimulus",
    "Theme",
    "Time",
    "Topic"
]

nlp = spacy.load("en_core_web_sm")

lemmatizer = WordNetLemmatizer()

path = "./data/vuaverb/"
counter_fails = 0
for num in range(10):
    if "rules" not in globals():
        selrestrs = {}

        file = f"train.csv" #{num}.csv"
        if file.split(".")[-1] != "csv" or "train" not in file.split(".")[0]:
            continue
        print(f"reading file {file}")
        df = pd.read_csv(os.path.join(path,file))

        for i,row in tqdm(df.iterrows()):
            url = urllib.parse.quote(row["sentence"])
            command = f"curl -s localhost:8080/predict/semantics?utterance={url} | python -m json.tool"
            target_word = lemmatizer.lemmatize(row["target_word"].lower().translate(str.maketrans('', '', string.punctuation)), pos="v")

            try:
                result = json.loads(os.popen(command).read())
            except:
                counter_fails += 1
                continue

            if ("props" in result.keys() and len(result["props"]) == 0) or ("props" not in result.keys()):
                counter_fails += 1
                continue

            prop_index = -1

            for k, prop in enumerate(result["props"]):
                for span in prop["spans"]:
                    if span["predicate"] == True and lemmatizer.lemmatize(span["text"].lower().translate(str.maketrans('', '', string.punctuation)), pos="v") == target_word:
                        prop_index = k
                        break
                if prop_index != -1:
                    break

            if prop_index == -1:
                counter_fails += 1
                continue

            arguments = []

            if result["props"][prop_index]["mainEvent"] is not None:
                for arg in result["props"][prop_index]["mainEvent"]["predicates"][0]["args"]:
                    if arg["type"] in map_to_semantic_type.keys():
                        arg["type"] = map_to_semantic_type[arg["type"]]
                    if arg["type"] in semantic_type_list:
                        arguments.append((arg["type"], arg["value"]))

            for event_nbr in range(len(result["props"][prop_index]["events"])):
                for arg in result["props"][prop_index]["events"][event_nbr]["predicates"][0]["args"]:
                    if arg["type"] in map_to_semantic_type.keys():
                        arg["type"] = map_to_semantic_type[arg["type"]]
                    if arg["type"] in semantic_type_list:
                        arguments.append((arg["type"], arg["value"]))
                
            arguments = list(set(arguments))

            doc = nlp(row["sentence"])
            pos_in_doc = -1
            descendants = []
            for j,tok in enumerate(doc):
                if lemmatizer.lemmatize(tok.text.lower().translate(str.maketrans('', '', string.punctuation)), pos="v") == target_word:
                    pos_in_doc = j
                    for k,t in enumerate(doc):
                        if k == j:
                            continue
                        if t.head.i == pos_in_doc:
                            descendants.append(t.text)
                    break
            
            if pos_in_doc == -1:
                counter_fails += 1
                continue

            if target_word not in selrestrs.keys():
                selrestrs[target_word] = {"metaphor": [], "literal": []}

            for arg in arguments:
                for word in arg[1].split():
                    if word in descendants:
                        wordnet_synsets = []
                        for ent in doc.ents:
                            if ent.text in arg[1] and word in ent.text:
                                if ent.label_ == "PERSON":
                                    wordnet_synsets.append(wn.synset("person.n.01"))
                                if ent.label_ == "ORG":
                                    wordnet_synsets.append(wn.synset("organization.n.01"))
                                
                        if len(wordnet_synsets) == 0:
                            word = word.translate(str.maketrans('', '', string.punctuation)).strip()

                            if word.lower() in [p.lower() for p in pronouns_to_living_thing]:
                                wordnet_synsets = wn.synsets("person", pos="n")
                            else:
                                wordnet_synsets = wn.synsets(word, pos="n")
                        
                        if len(wordnet_synsets) == 0:
                            continue

                        sem_type_to_add = None
                        type_to_add = None

                        if wordnet_synsets[0].lemma_names()[0] in vnselrestr_to_wn.values():
                            type_to_add = wordnet_synsets[0]
                            sem_type_to_add = arg[0]
                        else:
                            hp_path = wordnet_synsets[0].hypernym_paths()[0]
                            for el in reversed(hp_path):
                                
                                if el.lemma_names()[0] in vnselrestr_to_wn.values():
                                    type_to_add = el
                                    sem_type_to_add = arg[0]
                                    break
                        
                        if row["label"] == 1:
                            selrestrs[target_word]["metaphor"].append((sem_type_to_add, type_to_add))
                        else:
                            selrestrs[target_word]["literal"].append((sem_type_to_add, type_to_add))

        rules = get_rules(selrestrs)
        with open(f"rules/saved_vuaverb_train/rules{num}.txt", "w") as writer:
            print(rules, file=writer)
    else:
        file = f"test.csv" #{num}.csv"
        if file.split(".")[-1] != "csv" or "test" not in file.split(".")[0]:
            continue
        print(f"reading file {file}")
        df = pd.read_csv(os.path.join(path,file))
        nothing_found_counter = 0

        predictions = []
        labels = []
        texts = []
        total_arguments = []

        for i,row in tqdm(df.iterrows()):

            url = urllib.parse.quote(row["sentence"])
            command = f"curl -s localhost:8080/predict/semantics?utterance={url} | python -m json.tool"
            target_word = lemmatizer.lemmatize(row["target_word"].lower().translate(str.maketrans('', '', string.punctuation)), pos="v")

            try:
                result = json.loads(os.popen(command).read())
            except:
                counter_fails += 1
                predictions.append(0)
                labels.append(row["label"])
                texts.append(row["sentence"])
                total_arguments.append([])
                continue

            if ("props" in result.keys() and len(result["props"]) == 0) or ("props" not in result.keys()):
                predictions.append(0)
                labels.append(row["label"])
                texts.append(row["sentence"])
                total_arguments.append([])
                counter_fails += 1
                continue

            prop_index = -1

            for k, prop in enumerate(result["props"]):
                for span in prop["spans"]:
                    if span["predicate"] == True and lemmatizer.lemmatize(span["text"].lower().translate(str.maketrans('', '', string.punctuation)), pos="v") == target_word:
                        prop_index = k
                        break
                if prop_index != -1:
                    break

            if prop_index == -1:
                counter_fails += 1
                predictions.append(0)
                labels.append(row["label"])
                texts.append(row["sentence"])
                total_arguments.append([])
                continue

            arguments = []

            if result["props"][prop_index]["mainEvent"] is not None:
                for arg in result["props"][prop_index]["mainEvent"]["predicates"][0]["args"]:
                    if arg["type"] in map_to_semantic_type.keys():
                        arg["type"] = map_to_semantic_type[arg["type"]]
                    if arg["type"] in semantic_type_list:
                        arguments.append((arg["type"], arg["value"]))

            for event_nbr in range(len(result["props"][prop_index]["events"])):
                for arg in result["props"][prop_index]["events"][event_nbr]["predicates"][0]["args"]:
                    if arg["type"] in map_to_semantic_type.keys():
                        arg["type"] = map_to_semantic_type[arg["type"]]
                    if arg["type"] in semantic_type_list:
                        arguments.append((arg["type"], arg["value"]))
                
            arguments = list(set(arguments))

            doc = nlp(row["sentence"])
            pos_in_doc = -1
            descendants = []
            for j,tok in enumerate(doc):
                if lemmatizer.lemmatize(tok.text.lower().translate(str.maketrans('', '', string.punctuation)), pos="v") == target_word:
                    pos_in_doc = j
                    for k,t in enumerate(doc):
                        if k == j:
                            continue
                        if t.head.i == pos_in_doc:
                            descendants.append(t.text)
                    break
            
            if pos_in_doc == -1:
                counter_fails += 1
                predictions.append(0)
                labels.append(row["label"])
                texts.append(row["sentence"])
                total_arguments.append([])
                continue

            """if target_word not in selrestrs.keys():
                selrestrs[target_word] = {"metaphor": [], "literal": []}"""

            found = False
            for arg in arguments:
                for word in arg[1].split():
                    if word in descendants:
                        wordnet_synsets = []
                        for ent in doc.ents:
                            if ent.text in arg[1] and word in ent.text:
                                if ent.label_ == "PERSON":
                                    wordnet_synsets.append(wn.synset("person.n.01"))
                                if ent.label_ == "ORG":
                                    wordnet_synsets.append(wn.synset("organization.n.01"))
                                
                        if len(wordnet_synsets) == 0:
                            word = word.translate(str.maketrans('', '', string.punctuation)).strip()

                            if word.lower() in [p.lower() for p in pronouns_to_living_thing]:
                                wordnet_synsets = wn.synsets("person", pos="n")
                            else:
                                wordnet_synsets = wn.synsets(word, pos="n")
                        
                        if len(wordnet_synsets) == 0:
                            continue

                        sem_type_to_add = None
                        type_to_add = None

                        if wordnet_synsets[0].lemma_names()[0] in vnselrestr_to_wn.values():
                            type_to_add = wordnet_synsets[0]
                            sem_type_to_add = arg[0]
                        else:
                            hp_path = wordnet_synsets[0].hypernym_paths()[0]
                            for el in reversed(hp_path):
                                
                                if el.lemma_names()[0] in vnselrestr_to_wn.values():
                                    type_to_add = el
                                    sem_type_to_add = arg[0]
                                    break
                        
                        if target_word in rules.keys() and sem_type_to_add in rules[target_word]["metaphor"].keys() and type_to_add in rules[target_word]["metaphor"][sem_type_to_add].keys():
                            if rules[target_word]["metaphor"][sem_type_to_add][type_to_add]["score"] > 0.7 and rules[target_word]["metaphor"][sem_type_to_add][type_to_add]["frequency"] >= 5:
                                predictions.append(1)
                            else:
                                predictions.append(0)
                            labels.append(row["label"])
                            texts.append(row["sentence"])
                            total_arguments.append(arguments)
                            found = True
                            break
                        else:
                            nothing_found_counter += 1
            
                if found:
                    break
            if not found:
                predictions.append(0)
                labels.append(row["label"])
                texts.append(row["sentence"])
                total_arguments.append([])

        for lab, pred, sent, arg in zip(labels, predictions, texts, total_arguments):
            #if lab != pred:
            if pred == 1 and lab == 0:
                print(f"Sentence: {sent} --- Prediction: {pred} --- Label: {lab} --- Arguments: {arg}")
                print()

        with open(f"predictions/vuaverb/{num}.pkl", "wb") as writer:
            pickle.dump(predictions, writer)

        print(classification_report(labels, predictions))
        print()
    
print(a)     

