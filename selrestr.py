import networkx as nx
import nltk
import matplotlib.pyplot as plt

if __name__ == "__main__":
    nltk.download('wordnet')

from nltk.corpus import wordnet as wn

dg = nx.DiGraph()

dg.add_edge(wn.synset("vehicle.n.01"), wn.synset("machine.n.01"))
dg.add_edge(wn.synset("person.n.01"), wn.synset("living_thing.n.01"))
dg.add_edge(wn.synset("animal.n.01"), wn.synset("living_thing.n.01"))
dg.add_edge(wn.synset("body_part.n.01"), wn.synset("living_thing.n.01"))
dg.add_edge(wn.synset("living_thing.n.01"), "int_control")
dg.add_edge(wn.synset("machine.n.01"), "int_control")
dg.add_edge(wn.synset("living_thing.n.01"), "natural")
dg.add_edge(wn.synset("plant.n.01"), "natural")
dg.add_edge(wn.synset("machine.n.01"), wn.synset("artifact.n.01"))
dg.add_edge(wn.synset("clothing.n.01"), wn.synset("artifact.n.01"))
dg.add_edge(wn.synset("tool.n.01"), wn.synset("artifact.n.01"))
dg.add_edge(wn.synset("artifact.n.01"), wn.synset("physical_object.n.01"))
dg.add_edge(wn.synset("food.n.01"), wn.synset("physical_object.n.01"))
dg.add_edge(wn.synset("physical_object.n.01"), wn.synset("physical_entity.n.01"))
dg.add_edge(wn.synset("social_group.n.01"), wn.synset("living_thing.n.01"))
dg.add_edge("natural", wn.synset("physical_entity.n.01"))
dg.add_edge("int_control", wn.synset("physical_entity.n.01"))
dg.add_edge(wn.synset("physical_entity.n.01"), wn.synset("entity.n.01"))
dg.add_edge(wn.synset("time.n.01"), wn.synset("entity.n.01"))
dg.add_edge(wn.synset("state.n.01"), wn.synset("entity.n.01"))
dg.add_edge(wn.synset("abstraction.n.01"), wn.synset("entity.n.01"))
dg.add_edge(wn.synset("scalar.n.01"), wn.synset("entity.n.01"))
dg.add_edge(wn.synset("currency.n.01"), wn.synset("entity.n.01"))
dg.add_edge(wn.synset("location.n.01"), wn.synset("entity.n.01"))
dg.add_edge(wn.synset("organization.n.01"), wn.synset("entity.n.01"))
dg.add_edge(wn.synset("region.n.01"), wn.synset("location.n.01"))
dg.add_edge(wn.synset("place.n.01"), wn.synset("location.n.01"))
dg.add_edge(wn.synset("communication.n.01"), wn.synset("abstraction.n.01"))
dg.add_edge(wn.synset("sound.n.01"), wn.synset("abstraction.n.01"))
dg.add_edge(wn.synset("idea.n.01"), wn.synset("abstraction.n.01"))

if __name__ == "__main__":
    pos = nx.spring_layout(dg)
    nx.draw(dg, pos, with_labels=True, node_color='lightblue', edge_color='gray', width=4)
    plt.show()