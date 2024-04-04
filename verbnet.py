import os
import pandas as pd
import json
import string

import nltk
nltk.download('verbnet')
nltk.download('wordnet')

from nltk.corpus import verbnet
from nltk.corpus import wordnet as wn

from selrestr import dg

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
    "solid": "solid",
    "rigid": "rigidness",
    "pointy": "rigidness",
    "elongated": "rigidness",
    "nonrigid": "rigidness",
    "substance": "substance",
    "force": "force",
    "human": "person"
}

def get_selectional_restrictions(verb, sense_id):
    """
    Get selectional restrictions for a given VerbNet sense.
    Args:
        verb: The VerbNet verb.
        sense_id: The sense ID of the verb.
    Returns:
        A dictionary containing selectional restrictions for each argument.
    """
    selrestrs = {}

    try:
        vnclass = verbnet.vnclass(sense_id)
    except:
        try:
            vnclass = verbnet.vnclass(verb)
        except:
            return {}

    themroles = verbnet.themroles(vnclass)
    for role in themroles:
        selrestrs[role["type"]] = []
        for modifier in role["modifiers"]:
            t = vnselrestr_to_wn[modifier["type"]]
            if t == "int_control" or t == "natural":
                modifier["type"] = t
            else:
                modifier["type"] = wn.synset(vnselrestr_to_wn[modifier["type"]]+".n.01")
            selrestrs[role["type"]].append(modifier)
    
    return selrestrs

semantic_type_list = [
    "actor",
    "agent",
    "asset",
    "attribute",
    "beneficiary",
    "cause",
    "destination",
    "location",
    "experiencer",
    "instrument",
    "material",
    "product",
    "patient",
    "predicate",
    "recipient",
    "stimulus",
    "theme",
    "time",
    "topic"
]

counter_success = 0
counter_fails = 0

for i in range(10):
    test_path = f"./data/test{i}.csv"
    test_df = pd.read_csv(test_path)

    predictions = []

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
            print("Error with the following row")
            print(row["sentence"])
            print(result)
            continue
        
        if result["props"][prop_index]["mainEvent"] is not None:
            for arg in result["props"][prop_index]["mainEvent"]["predicates"][0]["args"]:
                if arg["type"].lower() in semantic_type_list:
                    arguments.append((arg["type"], arg["value"]))
        else:
            for arg in result["props"][prop_index]["events"][0]["predicates"][0]["args"]:
                if arg["type"].lower() in semantic_type_list:
                    arguments.append((arg["type"], arg["value"]))

        sense = result["props"][prop_index]["sense"]
        sense_splitted = sense.split("-")
        verb = sense_splitted[0]
        id = sense_splitted[1]
        if len(id.split('.')) > 2:
            id = '.'.join(id.split('.')[:-1])

        selrestrs = get_selectional_restrictions(verb, id)
        counter_success += 1
        """print(selrestrs)
        print(result)
        print()"""


print("Fails",counter_fails)
print("Success",counter_success)
