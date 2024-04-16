import os
import pandas as pd
import json
import string
import pickle as pkl
import urllib.parse

import spacy
import networkx as nx
import nltk
nltk.download('verbnet')
nltk.download('wordnet')

from tqdm import tqdm
from nltk.corpus import verbnet
from nltk.corpus import wordnet as wn
from sklearn.metrics import classification_report

from selrestr import dg

from nltk.stem import WordNetLemmatizer

nlp = spacy.load("en_core_web_sm")

lemmatizer = WordNetLemmatizer()

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

    themroles = verbnet.themroles(vnclass)
    for role in themroles:
        if role["type"] not in selrestrs.keys():
            selrestrs[role["type"]] = []

        for modifier in role["modifiers"]:
            if modifier["type"] == "refl": continue
            t = vnselrestr_to_wn[modifier["type"]]
            if t == "int_control":
                modifier["type"] = wn.synset("natural_phenomenon.n.01")
                selrestrs[role["type"]].append(modifier.copy())
                modifier["type"] = wn.synset("natural_event.n.01")
                selrestrs[role["type"]].append(modifier.copy())
                modifier["type"] = wn.synset("living_thing.n.01")
                selrestrs[role["type"]].append(modifier.copy())
                """modifier["type"] = wn.synset("tool.n.01")
                selrestrs[role["type"]].append(modifier.copy())"""
                """modifier["type"] = wn.synset("vehicle.n.01")
                selrestrs[role["type"]].append(modifier.copy())"""
                """modifier["type"] = wn.synset("machine.n.01")
                selrestrs[role["type"]].append(modifier.copy())"""
            elif modifier["type"] == "communication":
                """modifier["type"] = wn.synset("communication.n.01")
                selrestrs[role["type"]].append(modifier.copy())
                modifier["type"] = wn.synset("communication.n.02")
                selrestrs[role["type"]].append(modifier.copy())"""
                modifier["type"] = wn.synset("entity.n.01")
                selrestrs[role["type"]].append(modifier.copy())
                modifier["type"] = wn.synset("entity.n.02")
                selrestrs[role["type"]].append(modifier.copy())

            else:
                modifier["type"] = wn.synset(vnselrestr_to_wn[modifier["type"]]+".n.01")
                selrestrs[role["type"]].append(modifier)
    
    more_general_ids = sense_id.split("-")[:-1]
    if len(more_general_ids) > 0:
        for i in reversed(range(0,len(more_general_ids))):
            new_sense_id = '-'.join(more_general_ids[:i+1])
            selrestrs = get_selectional_restrictions(verb, new_sense_id, lemma, selrestrs)
    return selrestrs

def save_results(predictions, number):
    with open(f"./predictions/{number}.pkl", "wb") as writer:
        pkl.dump(predictions, writer)

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

pronouns_to_living_thing = ["I", "You", "He", "She", "We", "They", "Who", "It", "Them", "Me", "Him", "Her"]

counter_success = 0
counter_fails = 0
counter_wn_fails = 0

def get_descendants(synset, verbose=False):
    descendants = set([synset])
    for hyponym in synset.hyponyms():
        descendants.add(hyponym)
        descendants |= get_descendants(hyponym)
    
    root = wn.synset("entity.n.01")
    """descendants_extended = set()
    for descendant in descendants:
        descendants_extended.add(descendant)
        if descendant not in dg:
            continue
        for path in nx.all_simple_paths(dg, source=descendant, target=root):
            if verbose:
                print(path)
            for i, syn in enumerate(path):
                descendants_extended.add(syn)
                if i == 1:
                    break"""
    
    if synset in dg:
        res = dg.edges(data=True)
        additional_synsets = set()
        for u,v,e in res:
            """print(f"u: {u} \t v: {v} \t e: {e}")"""
            if isinstance(u,str) or isinstance(synset, str):
                continue
            if u == synset and e["type"] == "cooccurrence":
                additional_synsets.add(v)
        for syn in additional_synsets:
            descendants.add(syn)

    return descendants

CV = True

if CV:
    for i in range(10):
        test_path = f"./data/trofi/test{i}.csv"
        test_df = pd.read_csv(test_path)

        predictions = []
        labels = []
        sentences = []
        total_arguments = []
        total_selrestrs = []

        for j,row in tqdm(test_df.iterrows()):
            doc = nlp(row["sentence"])
            row["sentence"] = row["sentence"].replace("'","%27")
            row["sentence"] = row["sentence"].replace(";", ",").strip()
            arguments = []
            splitted_sentence = '%20'.join(row["sentence"].split())
            """urllib.parse.quote('%20'.join(row["sentence"].split()))"""

            command = f"curl -s localhost:8080/predict/semantics?utterance={splitted_sentence} | python -m json.tool"

            try:
                result = json.loads(os.popen(command).read())
            except:
                """print("Error made by the Verbnet parser")"""
                counter_fails += 1
                continue

            if "props" in result.keys() and len(result["props"]) == 0:
                counter_fails += 1
                predictions.append(0)
                labels.append(row["label"])
                sentences.append(row["sentence"])
                total_arguments.append(arguments)
                total_selrestrs.append({})
                """print("Props length is zero")"""
                continue
            elif "props" not in result.keys():
                counter_fails += 1
                predictions.append(0)
                labels.append(row["label"])
                sentences.append(row["sentence"])
                total_arguments.append(arguments)
                total_selrestrs.append({})
                """print("Props is not inside the result")"""
                continue

            prop_index = -1

            for k, prop in enumerate(result["props"]):
                for span in prop["spans"]:
                    if span["predicate"] == True and lemmatizer.lemmatize(span["text"].lower().translate(str.maketrans('', '', string.punctuation)), pos="v") == lemmatizer.lemmatize(row["target_word"].lower().translate(str.maketrans('', '', string.punctuation)), pos="v"):
                        prop_index = k
                        break
                if prop_index != -1:
                    break
            
            if prop_index == -1:
                counter_fails += 1
                predictions.append(0)
                labels.append(row["label"])
                sentences.append(row["sentence"])
                total_arguments.append(arguments)
                total_selrestrs.append({})
                """print("prop_index not found")
                print(row["sentence"])
                print(lemmatizer.lemmatize(row["target_word"].lower().translate(str.maketrans('', '', string.punctuation)), pos="v"))
                spans = []
                for prop in result["props"]:
                    for span in prop["spans"]:
                        spans.append(lemmatizer.lemmatize(span["text"].lower().translate(str.maketrans('', '', string.punctuation)), pos="v"))
                print(spans)"""
                """print(a)"""
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

            selrestrs = get_selectional_restrictions(verb, id, row["target_word"].lower().translate(str.maketrans('', '', string.punctuation)), {})
            counter_success += 1

            clash_detected_plus = True
            clash_detected_minus = False
            clash_detected = False
            for arg in arguments:
                clash_detected_plus = True
                clash_detected_minus = False
                if arg[0] not in selrestrs.keys():
                    continue

                if arg[1] == '' or selrestrs[arg[0]] == []:
                    continue

                if arg[0].lower() == "location" or arg[0].lower() == "destination":
                    continue

                wordnet_synsets = []
                hypernyms = set()
                hypernyms_minus = set()
                for word in arg[1].split():
                    try:
                        word = word.translate(str.maketrans('', '', string.punctuation)).strip()
                        if word.lower() in [p.lower() for p in pronouns_to_living_thing]:
                            wordnet_synsets = wn.synsets("person", pos="n")
                        else:
                            wordnet_synsets = wn.synsets(word, pos="n")
                        for ent in doc.ents:
                            if ent.text in arg[1]:
                                if ent.label_ == "PERSON":
                                    wordnet_synsets.append(wn.synset("person.n.01"))
                                """if ent.label_ == "ORG":
                                    wordnet_synsets.append(wn.synset("organization.n.01"))"""
                                pass
                        if len(wordnet_synsets) == 0:
                            continue
                        for z, wordnet_synset in enumerate(wordnet_synsets):
                            for path in wordnet_synset.hypernym_paths():
                                for syn in path:
                                    if z == 0:
                                        hypernyms_minus.add(syn)
                                    hypernyms.add(syn)
                            if z == 0:
                                hypernyms_minus.add(wordnet_synset)
                            hypernyms.add(wordnet_synset)
                    except:
                        pass

                if len(wordnet_synsets) == 0:
                    predictions.append(0)
                    labels.append(row["label"])
                    sentences.append(row["sentence"])
                    total_arguments.append(arguments)
                    total_selrestrs.append(selrestrs)
                    continue

                in_plus = 0
                in_minus = 0
                for restr in selrestrs[arg[0]]:
                    if restr["type"] == wn.synset("rigidity.n.01"):
                        continue
                    descendants = get_descendants(restr["type"], verbose=True)
                    condition = any([hyp in descendants for hyp in hypernyms])
                    condition_minus = any([hyp in descendants for hyp in hypernyms_minus])
                    if restr["value"] == "+":
                        in_plus = 1
                    if restr["value"] == "-":
                        in_minus = 1
                    if restr["value"] == "+" and condition: #wordnet_synset in descendants:
                        clash_detected_plus = False
                    elif restr["value"] == "-" and condition_minus:
                        clash_detected_minus = True

                if not in_plus:
                    clash_detected_plus = False
                if not in_minus:
                    clash_detected_minus = False
                if clash_detected_plus or clash_detected_minus:
                    clash_detected = True
                    break
            if clash_detected:
                predictions.append(1)
            else:
                predictions.append(0)
            labels.append(row["label"])
            sentences.append(row["sentence"])
            total_arguments.append(arguments)
            total_selrestrs.append(selrestrs)

        for lab, pred, sent, arg, selr in zip(labels, predictions, sentences, total_arguments, total_selrestrs):
            #if lab != pred:
            if pred == 1 and lab == 1:
                print(f"Sentence: {sent} --- Prediction: {pred} --- Label: {lab} --- Arguments: {arg} --- Selrestrs: {selr}")
                print()
            """if pred == 1 and lab == 1:
                print(f"Sentence: {sent} --- Prediction: {pred} --- Label: {lab} --- Arguments: {arg} --- Selrestrs: {selr}")
                print()"""

        print(classification_report(labels, predictions))
        print()
        #print(a)
        new_p = []
        for j, sent in enumerate(sentences):
            new_p.append([sent, predictions[j]])
        save_results(new_p, i)
        print(a)

else:
    test_path = f"./data/vuaverb/test.csv"
    test_df = pd.read_csv(test_path)

    predictions = []
    labels = []
    sentences = []
    total_arguments = []
    total_selrestrs = []

    for j,row in tqdm(test_df.iterrows()):
        doc = nlp(row["sentence"])
        """row["sentence"] = row["sentence"].replace("'","%27")
        row["sentence"] = row["sentence"].replace(";", ",").strip()
        arguments = []
        splitted_sentence = '%20'.join(row["sentence"].split())"""
        arguments = []
        splitted_sentence = urllib.parse.quote(row["sentence"].replace(";", ",").strip(), safe='/', encoding=None, errors=None)

        command = f"curl -s localhost:8080/predict/semantics?utterance={splitted_sentence} | python -m json.tool"

        try:
            result = json.loads(os.popen(command).read())
        except:
            print("Error made by the Verbnet parser")
            print(command)
            print(a)
            counter_fails += 1
            continue

        if "props" in result.keys() and len(result["props"]) == 0:
            counter_fails += 1
            predictions.append(0)
            labels.append(row["label"])
            sentences.append(row["sentence"])
            total_arguments.append(arguments)
            total_selrestrs.append({})
            print("Props length is zero")
            continue
        elif "props" not in result.keys():
            counter_fails += 1
            predictions.append(0)
            labels.append(row["label"])
            sentences.append(row["sentence"])
            total_arguments.append(arguments)
            total_selrestrs.append({})
            print("Props is not inside the result")
            continue

        prop_index = -1

        for k, prop in enumerate(result["props"]):
            for span in prop["spans"]:
                if span["predicate"] == True and lemmatizer.lemmatize(span["text"].split()[0].lower().translate(str.maketrans('', '', string.punctuation)), pos="v") == lemmatizer.lemmatize(row["target_word"].lower().translate(str.maketrans('', '', string.punctuation)), pos="v"):
                    prop_index = k
                    break
                """elif span["predicate"]:
                    print(result)
                    print(f"Span: {span['text'].lower().translate(str.maketrans('', '', string.punctuation))}")
                    print(f"Span lemmatized: {lemmatizer.lemmatize(span['text'].lower().translate(str.maketrans('', '', string.punctuation)), pos='v')}")
                    print(f"Row target: {row['target_word'].lower().translate(str.maketrans('', '', string.punctuation))}")
                    print(f"Row target lemmatized: {lemmatizer.lemmatize(row['target_word'].lower().translate(str.maketrans('', '', string.punctuation)), pos='v')}")"""
            if prop_index != -1:
                break
        
        if prop_index == -1:
            counter_fails += 1
            predictions.append(0)
            labels.append(row["label"])
            sentences.append(row["sentence"])
            total_arguments.append(arguments)
            total_selrestrs.append({})
            print("prop_index not found")
            """print(row["sentence"])
            print(lemmatizer.lemmatize(row["target_word"].lower().translate(str.maketrans('', '', string.punctuation)), pos="v"))
            spans = []
            for prop in result["props"]:
                for span in prop["spans"]:
                    spans.append(lemmatizer.lemmatize(span["text"].lower().translate(str.maketrans('', '', string.punctuation)), pos="v"))
            print(spans)"""
            """print(a)"""
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

        selrestrs = get_selectional_restrictions(verb, id, row["target_word"].lower().translate(str.maketrans('', '', string.punctuation)), {})
        counter_success += 1

        clash_detected_plus = True
        clash_detected_minus = False
        clash_detected = False
        for arg in arguments:
            clash_detected_plus = True
            clash_detected_minus = False
            if arg[0] not in selrestrs.keys():
                continue

            if arg[1] == '' or selrestrs[arg[0]] == []:
                continue

            if arg[0].lower() == "location" or arg[0].lower() == "destination":
                continue

            wordnet_synsets = []
            hypernyms = set()
            hypernyms_minus = set()
            for word in arg[1].split():
                try:
                    word = word.translate(str.maketrans('', '', string.punctuation)).strip()
                    if word.lower() in [p.lower() for p in pronouns_to_living_thing]:
                        wordnet_synsets = wn.synsets("person", pos="n")
                    else:
                        wordnet_synsets = wn.synsets(word, pos="n")
                    for ent in doc.ents:
                        if ent.text in arg[1]:
                            if ent.label_ == "PERSON":
                                wordnet_synsets.append(wn.synset("person.n.01"))
                            if ent.label_ == "ORG":
                                wordnet_synsets.append(wn.synset("organization.n.01"))
                    if len(wordnet_synsets) == 0:
                        continue
                    for z, wordnet_synset in enumerate(wordnet_synsets):
                        for path in wordnet_synset.hypernym_paths():
                            for syn in path:
                                if z == 0:
                                    hypernyms_minus.add(syn)
                                hypernyms.add(syn)
                        if z == 0:
                            hypernyms_minus.add(wordnet_synset)
                        hypernyms.add(wordnet_synset)
                except:
                    pass

            if len(wordnet_synsets) == 0:
                predictions.append(0)
                labels.append(row["label"])
                sentences.append(row["sentence"])
                total_arguments.append(arguments)
                total_selrestrs.append(selrestrs)
                continue

            in_plus = 0
            in_minus = 0
            for restr in selrestrs[arg[0]]:
                if restr["type"] == wn.synset("rigidity.n.01"):
                    continue
                descendants = get_descendants(restr["type"], verbose=True)
                condition = any([hyp in descendants for hyp in hypernyms])
                condition_minus = any([hyp in descendants for hyp in hypernyms_minus])
                if restr["value"] == "+":
                    in_plus = 1
                if restr["value"] == "-":
                    in_minus = 1
                if restr["value"] == "+" and condition: #wordnet_synset in descendants:
                    clash_detected_plus = False
                elif restr["value"] == "-" and condition_minus:
                    clash_detected_minus = True

            if not in_plus:
                clash_detected_plus = False
            if not in_minus:
                clash_detected_minus = False
            if clash_detected_plus or clash_detected_minus:
                clash_detected = True
                break
        if clash_detected:
           predictions.append(1)
        else:
           predictions.append(0)
        labels.append(row["label"])
        sentences.append(row["sentence"])
        total_arguments.append(arguments)
        total_selrestrs.append(selrestrs)

    for lab, pred, sent, arg, selr in zip(labels, predictions, sentences, total_arguments, total_selrestrs):
        #if lab != pred:
        if pred == 1 and lab == 0:
            print(f"Sentence: {sent} --- Prediction: {pred} --- Label: {lab} --- Arguments: {arg} --- Selrestrs: {selr}")
            print()
        """if pred == 1 and lab == 1:
            print(f"Sentence: {sent} --- Prediction: {pred} --- Label: {lab} --- Arguments: {arg} --- Selrestrs: {selr}")
            print()"""

    print(classification_report(labels, predictions))
    print()
    #print(a)
    new_p = []
    for j, sent in enumerate(sentences):
        new_p.append([sent, predictions[j]])
    save_results(new_p, 0)


print("Fails",counter_fails)
print("Success",counter_success)
