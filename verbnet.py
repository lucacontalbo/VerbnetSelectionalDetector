import os
import pandas as pd
import json
import string

import networkx as nx
import nltk
nltk.download('verbnet')
nltk.download('wordnet')

from nltk.corpus import verbnet
from nltk.corpus import wordnet as wn
from sklearn.metrics import classification_report

from selrestr import dg

from nltk.stem import WordNetLemmatizer
 
lemmatizer = WordNetLemmatizer()

vnselrestr_to_wn = {
    "animate": "living_thing",
    "vehicle": "vehicle",
    "machine": "machine",
    "person": "person",
    "animal": "animal",
    "body-part": "body_part",
    "int_control": "int_control",
    "natural": "natural",
    "plant": "plant",
    "food": "food",
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
    "force": "force",
    "human": "person"
}

def get_selectional_restrictions(verb, sense_id, lemma="", selrestrs = {}):
    """
    Get selectional restrictions for a given VerbNet sense.
    Args:
        verb: The VerbNet verb.
        sense_id: The sense ID of the verb.
    Returns:
        A dictionary containing selectional restrictions for each argument.
    """

    try:
        print(sense_id)
        vnclass = verbnet.vnclass(sense_id)
    except:
        try:
            s = sense_id.split(".")[0]
            vnclass = verbnet.vnclass(s)
        except:
            try:
                vnclass = verbnet.vnclass(verbnet.classids(lemma=lemmatizer.lemmatize(lemma, pos="v"))[0])
            except:
                return {}
    print()
    themroles = verbnet.themroles(vnclass)
    for role in themroles:
        if role["type"] not in selrestrs.keys():
            selrestrs[role["type"]] = []

        for modifier in role["modifiers"]:
            if modifier["type"] == "refl": continue
            t = vnselrestr_to_wn[modifier["type"]]
            if t == "int_control" or t == "natural":
                #modifier["type"] = t
                modifier["type"] = wn.synset("organism.n.01")
            else:
                modifier["type"] = wn.synset(vnselrestr_to_wn[modifier["type"]]+".n.01")
            selrestrs[role["type"]].append(modifier)
    
    more_general_ids = sense_id.split("-")[:-1]
    if len(more_general_ids) > 0:
        for i in reversed(range(0,len(more_general_ids))):
            new_sense_id = '-'.join(more_general_ids[:i+1])
            selrestrs = get_selectional_restrictions(verb, new_sense_id, lemma, selrestrs)
    return selrestrs

map_to_semantic_type = {
    "Initial Location": "Location",
}

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

counter_success = 0
counter_fails = 0
counter_wn_fails = 0

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

for i in range(10):
    test_path = f"./data/test{i}.csv"
    test_df = pd.read_csv(test_path)

    predictions = []
    labels = []
    sentences = []
    total_arguments = []
    total_selrestrs = []

    for j,row in test_df.iterrows():
        row["sentence"] = row["sentence"].replace("'","%27")
        row["sentence"] = row["sentence"].replace(";", ",").strip()
        arguments = []
        splitted_sentence = '%20'.join(row["sentence"].split())

        command = f"curl -s localhost:8080/predict/semantics?utterance={splitted_sentence} | python -m json.tool"

        result = json.loads(os.popen(command).read())

        if len(result["props"]) == 0:
            counter_fails += 1
            continue

        prop_index = -1

        for k, prop in enumerate(result["props"]):
            for span in prop["spans"]:
                if span["predicate"] == True and span["text"].lower().translate(str.maketrans('', '', string.punctuation)) == row["target_word"].lower().translate(str.maketrans('', '', string.punctuation)):
                    prop_index = k
                    break
            if prop_index != -1:
                break
        
        if prop_index == -1:
            counter_fails += 1
            continue
        
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

        sense = result["props"][prop_index]["sense"]
        sense_splitted = sense.split("-")
        verb = sense_splitted[0]
        id = '-'.join(sense_splitted[1:])
        """if len(id.split('-')) > 2:
            id = '.'.join(id.split('.')[:-1])"""

        selrestrs = get_selectional_restrictions(verb, id, row["target_word"].lower().translate(str.maketrans('', '', string.punctuation)), {})
        counter_success += 1

        clash_detected = False
        for arg in arguments:
            clash_detected = False
            if arg[0] not in selrestrs.keys():
                continue

            wordnet_synset = None
            for word in arg[1].split():
                try:
                    wordnet_synset = wn.synsets(word, pos="n")[0]
                except:
                    pass

            if wordnet_synset is None:
                counter_wn_fails += 1
                continue

            for restr in selrestrs[arg[0]]:
                descendants = get_descendants(restr["type"])
                if restr["value"] == "+" and wordnet_synset not in descendants:
                    predictions.append(1)
                    clash_detected = True
                    break
                elif restr["value"] == "-" and wordnet_synset in descendants:
                    predictions.append(1)
                    clash_detected = True
                    break

            if clash_detected:
                break
            
        if not clash_detected:
            predictions.append(0)
        labels.append(row["label"])
        sentences.append(row["sentence"])
        total_arguments.append(arguments)
        total_selrestrs.append(selrestrs)

    for lab, pred, sent, arg, selr in zip(labels, predictions, sentences, total_arguments, total_selrestrs):
        if lab != pred:
            print(f"Sentence: {sent} --- Prediction: {pred} --- Label: {lab} --- Arguments: {arg} --- Selrestrs: {selr}")
            print()
    print(classification_report(labels, predictions))
    print(a)


print("Fails",counter_fails)
print("Success",counter_success)
