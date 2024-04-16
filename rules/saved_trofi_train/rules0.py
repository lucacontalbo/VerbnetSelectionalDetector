import nltk
nltk.download('verbnet')
nltk.download('wordnet')

from tqdm import tqdm
from nltk.corpus import verbnet
from nltk.corpus import wordnet as wn

rules = {
    "miss": {"metaphor": {}, "literal": {}},
    "examine": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 4,
                    "conf": 0.5714285714285714,
                    "score": 0.15384615384615385,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.14285714285714285,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.16666666666666666,
                },
            },
            "Location": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.6363636363636364,
                    "score": 0.1794871794871795,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 0.3333333333333333,
                },
                wn.synset("time.n.05"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 0.08333333333333333,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            },
        },
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 22, "conf": 0.6111111111111112},
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.1388888888888889,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 6,
                    "conf": 0.16666666666666666,
                },
                wn.synset("region.n.03"): {"frequency": 2, "conf": 0.05555555555555555},
                wn.synset("substance.n.07"): {
                    "frequency": 1,
                    "conf": 0.027777777777777776,
                },
            },
            "Location": {
                wn.synset("person.n.01"): {"frequency": 9, "conf": 0.1125},
                wn.synset("abstraction.n.06"): {"frequency": 32, "conf": 0.4},
                wn.synset("communication.n.02"): {"frequency": 11, "conf": 0.1375},
                wn.synset("substance.n.01"): {"frequency": 3, "conf": 0.0375},
                wn.synset("region.n.03"): {"frequency": 2, "conf": 0.025},
                wn.synset("idea.n.01"): {"frequency": 7, "conf": 0.0875},
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.025},
                wn.synset("body_part.n.01"): {"frequency": 2, "conf": 0.025},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.0125},
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.0125},
                wn.synset("artifact.n.01"): {"frequency": 6, "conf": 0.075},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.0125},
                wn.synset("plant.n.01"): {"frequency": 1, "conf": 0.0125},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.0125},
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.0125},
            },
        },
    },
    "drown": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.6666666666666666,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {"Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "dance": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.2,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.07692307692307693,
                },
            }
        },
        "literal": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.15},
                wn.synset("organization.n.01"): {"frequency": 4, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 12, "conf": 0.6},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.05},
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("food.n.01"): {"frequency": 2, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.25},
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.125},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.125},
            },
        },
    },
    "ride": {
        "metaphor": {
            "Location": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            },
            "Theme": {
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.16666666666666666,
                },
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 0.25}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 5, "conf": 0.3125},
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.0625},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.0625},
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.25},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.0625},
                wn.synset("machine.n.04"): {"frequency": 1, "conf": 0.0625},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.0625},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.0625},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.0625},
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.2727272727272727,
                },
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.5454545454545454},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.09090909090909091},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                },
            },
            "Location": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "play": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 9,
                    "conf": 0.8181818181818182,
                    "score": 0.9,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 7,
                    "conf": 0.7777777777777778,
                    "score": 0.875,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 1.0,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "touch": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 6,
                    "conf": 0.6,
                    "score": 0.6666666666666666,
                },
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.3, "score": 0.6},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.1, "score": 1.0},
            }
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.3},
                wn.synset("physical_entity.n.01"): {"frequency": 2, "conf": 0.2},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.2},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.1},
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.1},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.1},
            },
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "pump": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 20,
                    "conf": 0.6666666666666666,
                    "score": 0.9523809523809523,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.03333333333333333,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.03333333333333333,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.03333333333333333,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.03333333333333333,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 3,
                    "conf": 0.1,
                    "score": 0.42857142857142855,
                },
                wn.synset("currency.n.01"): {
                    "frequency": 1,
                    "conf": 0.03333333333333333,
                    "score": 0.5,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.03333333333333333,
                    "score": 0.5,
                },
                wn.synset("state.n.01"): {
                    "frequency": 1,
                    "conf": 0.03333333333333333,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 5,
                    "conf": 0.35714285714285715,
                    "score": 0.5555555555555556,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 3,
                    "conf": 0.21428571428571427,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 0.5,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 4,
                    "conf": 0.2857142857142857,
                    "score": 1.0,
                },
            },
            "Destination": {
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.5,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.5714285714285714,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("currency.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("substance.n.01"): {"frequency": 4, "conf": 0.5714285714285714},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
            },
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.8},
            },
            "Destination": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "die": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.3333333333333333,
                    "score": 0.2,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 6,
                    "conf": 0.6666666666666666,
                    "score": 0.46153846153846156,
                },
            }
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 12, "conf": 0.5454545454545454},
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.3181818181818182,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.045454545454545456,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.045454545454545456,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.045454545454545456},
            }
        },
    },
    "plow": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.058823529411764705,
                }
            },
            "Destination": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.07142857142857142,
                }
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 16, "conf": 0.8},
                wn.synset("organization.n.01"): {"frequency": 3, "conf": 0.15},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.05},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 13, "conf": 0.65},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.05},
                wn.synset("artifact.n.01"): {"frequency": 4, "conf": 0.2},
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.05},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.05},
            },
            "Destination": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "cool": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 14,
                    "conf": 0.45161290322580644,
                    "score": 0.8235294117647058,
                },
                wn.synset("state.n.02"): {
                    "frequency": 7,
                    "conf": 0.22580645161290322,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 2,
                    "conf": 0.06451612903225806,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.03225806451612903,
                    "score": 1.0,
                },
                wn.synset("sound.n.04"): {
                    "frequency": 1,
                    "conf": 0.03225806451612903,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.06451612903225806,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.03225806451612903,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.03225806451612903,
                    "score": 0.5,
                },
                wn.synset("region.n.01"): {
                    "frequency": 1,
                    "conf": 0.03225806451612903,
                    "score": 1.0,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.03225806451612903,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("sound.n.04"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25, "score": 0.5},
            },
        },
        "literal": {
            "Patient": {
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.75},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "lend": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 6,
                    "conf": 0.8571428571428571,
                    "score": 0.15789473684210525,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.25}
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 20, "conf": 0.5714285714285714},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.02857142857142857},
                wn.synset("organization.n.01"): {
                    "frequency": 8,
                    "conf": 0.22857142857142856,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.02857142857142857},
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.08571428571428572,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.02857142857142857,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.02857142857142857,
                },
            },
            "Recipient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.26666666666666666,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 4,
                    "conf": 0.26666666666666666,
                },
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.4},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.06666666666666667},
            },
            "Theme": {
                wn.synset("location.n.01"): {"frequency": 2, "conf": 0.04878048780487805},
                wn.synset("abstraction.n.06"): {
                    "frequency": 32,
                    "conf": 0.7804878048780488,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.024390243902439025,
                },
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.07317073170731707},
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.04878048780487805,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.024390243902439025,
                },
            },
        },
    },
    "dissolve": {
        "metaphor": {
            "Patient": {
                wn.synset("organization.n.01"): {
                    "frequency": 11,
                    "conf": 0.4230769230769231,
                    "score": 0.6875,
                },
                wn.synset("communication.n.01"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                    "score": 0.25,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.2692307692307692,
                    "score": 0.3888888888888889,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.11538461538461539,
                    "score": 0.6,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                    "score": 0.2,
                },
                wn.synset("machine.n.01"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                    "score": 1.0,
                },
                wn.synset("state.n.01"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 4,
                    "conf": 0.8,
                    "score": 0.6666666666666666,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 0.3333333333333333,
                },
            },
        },
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 11,
                    "conf": 0.3548387096774194,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 4,
                    "conf": 0.12903225806451613,
                },
                wn.synset("state.n.02"): {"frequency": 3, "conf": 0.0967741935483871},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.0967741935483871},
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.06451612903225806,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 5,
                    "conf": 0.16129032258064516,
                },
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.03225806451612903},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.03225806451612903},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.03225806451612903},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.2857142857142857},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.14285714285714285},
            },
        },
    },
    "target": {
        "metaphor": {
            "Theme": {
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.0625,
                    "score": 0.2,
                },
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.125, "score": 1.0},
                wn.synset("living_thing.n.01"): {
                    "frequency": 1,
                    "conf": 0.0625,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.1875,
                    "score": 0.21428571428571427,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.0625, "score": 0.1},
                wn.synset("state.n.04"): {"frequency": 1, "conf": 0.0625, "score": 1.0},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.0625,
                    "score": 0.3333333333333333,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.25,
                    "score": 0.23529411764705882,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.0625, "score": 0.5},
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.0625,
                    "score": 0.3333333333333333,
                },
            },
            "Agent": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.375,
                    "score": 0.75,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 4,
                    "conf": 0.5,
                    "score": 0.4444444444444444,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 13,
                    "conf": 0.28888888888888886,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 11,
                    "conf": 0.24444444444444444,
                },
                wn.synset("person.n.01"): {"frequency": 9, "conf": 0.2},
                wn.synset("state.n.01"): {"frequency": 2, "conf": 0.044444444444444446},
                wn.synset("substance.n.01"): {"frequency": 4, "conf": 0.08888888888888889},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.044444444444444446,
                },
                wn.synset("region.n.03"): {"frequency": 2, "conf": 0.044444444444444446},
                wn.synset("place.n.10"): {"frequency": 1, "conf": 0.022222222222222223},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.022222222222222223},
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 5, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 15, "conf": 0.6},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.04},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.04},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.04},
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.04},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.04},
            },
            "Instrument": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5},
            },
        },
    },
    "absorb": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 12,
                    "conf": 0.5454545454545454,
                    "score": 0.3,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.045454545454545456,
                    "score": 0.3333333333333333,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.09090909090909091,
                    "score": 0.5,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.045454545454545456,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.13636363636363635,
                    "score": 0.42857142857142855,
                },
                wn.synset("food.n.01"): {
                    "frequency": 1,
                    "conf": 0.045454545454545456,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.045454545454545456,
                    "score": 0.3333333333333333,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.045454545454545456,
                    "score": 1.0,
                },
            }
        },
        "literal": {
            "Theme": {
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.043478260869565216,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 28,
                    "conf": 0.6086956521739131,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.043478260869565216,
                },
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.08695652173913043},
                wn.synset("substance.n.01"): {
                    "frequency": 2,
                    "conf": 0.043478260869565216,
                },
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.043478260869565216},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.021739130434782608},
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.021739130434782608},
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 4,
                    "conf": 0.08695652173913043,
                },
            }
        },
    },
    "escape": {
        "metaphor": {
            "Theme": {
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.25,
                    "score": 0.5,
                },
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.5, "score": 0.25},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 0.2,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 0.5,
                },
            }
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 12, "conf": 0.5217391304347826},
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.17391304347826086,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                },
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.08695652173913043},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.08695652173913043,
                },
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.043478260869565216},
            }
        },
    },
    "fill": {
        "metaphor": {
            "Destination": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 6,
                    "conf": 0.23076923076923078,
                    "score": 0.3157894736842105,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 8,
                    "conf": 0.3076923076923077,
                    "score": 0.47058823529411764,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 5,
                    "conf": 0.19230769230769232,
                    "score": 0.45454545454545453,
                },
                wn.synset("region.n.03"): {
                    "frequency": 3,
                    "conf": 0.11538461538461539,
                    "score": 1.0,
                },
                wn.synset("location.n.01"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                    "score": 0.125,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.42857142857142855,
                    "score": 0.75,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                    "score": 0.5,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.5,
                },
                wn.synset("state.n.04"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("animal.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                    "score": 0.6666666666666666,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
            },
            "Destination": {
                wn.synset("location.n.01"): {"frequency": 7, "conf": 0.19444444444444445},
                wn.synset("abstraction.n.06"): {"frequency": 9, "conf": 0.25},
                wn.synset("state.n.02"): {"frequency": 13, "conf": 0.3611111111111111},
                wn.synset("artifact.n.01"): {"frequency": 6, "conf": 0.16666666666666666},
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.027777777777777776,
                },
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "melt": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            }
        },
        "literal": {},
    },
    "flow": {
        "metaphor": {
            "Theme": {
                wn.synset("currency.n.01"): {
                    "frequency": 2,
                    "conf": 0.13333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 6,
                    "conf": 0.4,
                    "score": 0.2857142857142857,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                    "score": 0.3333333333333333,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.13333333333333333,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {"frequency": 3, "conf": 0.2, "score": 1.0},
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                    "score": 1.0,
                },
            }
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 15,
                    "conf": 0.5769230769230769,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 3,
                    "conf": 0.11538461538461539,
                },
                wn.synset("artifact.n.01"): {"frequency": 4, "conf": 0.15384615384615385},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.07692307692307693},
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                },
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.038461538461538464},
            }
        },
    },
    "rest": {
        "metaphor": {
            "Theme": {
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.10526315789473684,
                    "score": 0.5,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 6,
                    "conf": 0.3157894736842105,
                    "score": 0.6666666666666666,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.05263157894736842,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 3,
                    "conf": 0.15789473684210525,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.10526315789473684,
                    "score": 0.6666666666666666,
                },
                wn.synset("time.n.05"): {
                    "frequency": 1,
                    "conf": 0.05263157894736842,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.05263157894736842,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.10526315789473684,
                    "score": 0.3333333333333333,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.05263157894736842,
                    "score": 0.25,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.23076923076923078,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.15384615384615385,
                },
                wn.synset("body_part.n.01"): {"frequency": 3, "conf": 0.23076923076923078},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                },
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.3076923076923077},
            }
        },
    },
    "sleep": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 14,
                    "conf": 0.875,
                    "score": 0.6666666666666666,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.125,
                    "score": 0.5,
                },
            }
        },
        "literal": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 7, "conf": 0.7},
                wn.synset("time.n.05"): {"frequency": 1, "conf": 0.1},
            }
        },
    },
    "eat": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 8,
                    "conf": 0.6153846153846154,
                    "score": 0.4444444444444444,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.23076923076923078,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
            },
            "Patient": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.11764705882352941,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.17647058823529413,
                    "score": 0.42857142857142855,
                },
                wn.synset("person.n.01"): {
                    "frequency": 4,
                    "conf": 0.23529411764705882,
                    "score": 0.5714285714285714,
                },
                wn.synset("food.n.02"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                    "score": 0.3333333333333333,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                    "score": 1.0,
                },
                wn.synset("food.n.01"): {
                    "frequency": 3,
                    "conf": 0.17647058823529413,
                    "score": 0.3333333333333333,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                    "score": 0.2,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.11764705882352941,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 10, "conf": 1.0}},
            "Patient": {
                wn.synset("food.n.01"): {"frequency": 6, "conf": 0.24},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.12},
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.16},
                wn.synset("animal.n.01"): {"frequency": 4, "conf": 0.16},
                wn.synset("artifact.n.01"): {"frequency": 4, "conf": 0.16},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.04},
                wn.synset("food.n.02"): {"frequency": 2, "conf": 0.08},
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.04},
            },
        },
    },
    "roll": {
        "metaphor": {
            "Theme": {
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.03225806451612903,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 9,
                    "conf": 0.2903225806451613,
                    "score": 0.75,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.03225806451612903,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 8,
                    "conf": 0.25806451612903225,
                    "score": 0.5333333333333333,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 12,
                    "conf": 0.3870967741935484,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 0.5}
            },
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 7, "conf": 0.5833333333333334},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.25},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                },
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.08333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Patient": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "step": {
        "metaphor": {
            "Theme": {
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.21428571428571427,
                    "score": 0.75,
                },
                wn.synset("person.n.01"): {
                    "frequency": 10,
                    "conf": 0.7142857142857143,
                    "score": 0.7692307692307693,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 0.5,
                },
            },
            "Destination": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
        },
        "literal": {
            "Location": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.375},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.125},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.125},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.125},
            },
        },
    },
    "grasp": {
        "metaphor": {
            "Stimulus": {
                wn.synset("substance.n.01"): {
                    "frequency": 2,
                    "conf": 0.15384615384615385,
                    "score": 0.6666666666666666,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 0.5,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.23076923076923078,
                    "score": 0.75,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.38461538461538464,
                    "score": 0.8333333333333334,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 0.5,
                },
            }
        },
        "literal": {
            "Stimulus": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
            }
        },
    },
    "rid": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.5, "score": 0.25},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.2,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.5714285714285714},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
            },
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 6, "conf": 0.4},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.06666666666666667},
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.26666666666666666,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                },
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.06666666666666667},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.06666666666666667},
            },
        },
    },
    "kill": {
        "metaphor": {
            "Patient": {
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.07894736842105263,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 2,
                    "conf": 0.05263157894736842,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 18,
                    "conf": 0.47368421052631576,
                    "score": 0.5142857142857142,
                },
                wn.synset("person.n.01"): {
                    "frequency": 12,
                    "conf": 0.3157894736842105,
                    "score": 0.3076923076923077,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.02631578947368421,
                    "score": 0.5,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.02631578947368421,
                    "score": 0.5,
                },
                wn.synset("animal.n.01"): {
                    "frequency": 1,
                    "conf": 0.02631578947368421,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 6,
                    "conf": 0.5,
                    "score": 0.5454545454545454,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 0.5,
                },
                wn.synset("sound.n.04"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 0.5,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.16666666666666666,
                    "score": 0.2857142857142857,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 0.5,
                },
            },
        },
        "literal": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 27, "conf": 0.574468085106383},
                wn.synset("abstraction.n.06"): {
                    "frequency": 17,
                    "conf": 0.3617021276595745,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.02127659574468085},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.02127659574468085,
                },
                wn.synset("living_thing.n.01"): {
                    "frequency": 1,
                    "conf": 0.02127659574468085,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 5, "conf": 0.2777777777777778},
                wn.synset("sound.n.04"): {"frequency": 1, "conf": 0.05555555555555555},
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.16666666666666666},
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.2777777777777778,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.1111111111111111,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                },
            },
        },
    },
    "fly": {
        "metaphor": {
            "Theme": {
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 6,
                    "conf": 0.3333333333333333,
                    "score": 0.46153846153846156,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.2777777777777778,
                    "score": 0.17857142857142858,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.1111111111111111,
                    "score": 0.6666666666666666,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                    "score": 0.25,
                },
                wn.synset("location.n.01"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                    "score": 0.14285714285714285,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5, "score": 0.25},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 0.5,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.06666666666666667},
                wn.synset("abstraction.n.06"): {
                    "frequency": 23,
                    "conf": 0.5111111111111111,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.022222222222222223,
                },
                wn.synset("animal.n.01"): {"frequency": 2, "conf": 0.044444444444444446},
                wn.synset("person.n.01"): {"frequency": 7, "conf": 0.15555555555555556},
                wn.synset("organization.n.01"): {
                    "frequency": 6,
                    "conf": 0.13333333333333333,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.022222222222222223,
                },
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.022222222222222223},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.022222222222222223,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.8571428571428571},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
            },
        },
    },
    "vaporize": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
            }
        },
    },
    "pour": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 19,
                    "conf": 0.5757575757575758,
                    "score": 0.76,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 4,
                    "conf": 0.12121212121212122,
                    "score": 1.0,
                },
                wn.synset("food.n.01"): {
                    "frequency": 1,
                    "conf": 0.030303030303030304,
                    "score": 0.16666666666666666,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.030303030303030304,
                    "score": 1.0,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 4,
                    "conf": 0.12121212121212122,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.030303030303030304,
                    "score": 0.5,
                },
            },
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 4,
                    "conf": 0.4444444444444444,
                    "score": 0.8,
                },
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.3333333333333333,
                    "score": 0.23076923076923078,
                },
                wn.synset("region.n.03"): {
                    "frequency": 2,
                    "conf": 0.2222222222222222,
                    "score": 1.0,
                },
            },
            "Destination": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.7142857142857143,
                    "score": 0.7142857142857143,
                },
                wn.synset("state.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 10, "conf": 0.8333333333333334},
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.08333333333333333},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                },
            },
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.05263157894736842},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 3,
                    "conf": 0.15789473684210525,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 6,
                    "conf": 0.3157894736842105,
                },
                wn.synset("food.n.01"): {"frequency": 5, "conf": 0.2631578947368421},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.05263157894736842},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.05263157894736842},
                wn.synset("substance.n.07"): {"frequency": 2, "conf": 0.10526315789473684},
            },
            "Destination": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.4},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4},
            },
        },
    },
    "strike": {
        "metaphor": {
            "Patient": {
                wn.synset("substance.n.01"): {"frequency": 3, "conf": 0.15, "score": 0.75},
                wn.synset("person.n.01"): {
                    "frequency": 4,
                    "conf": 0.2,
                    "score": 0.36363636363636365,
                },
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.05, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.05,
                    "score": 0.3333333333333333,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.05,
                    "score": 0.3333333333333333,
                },
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.05, "score": 1.0},
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.25,
                    "score": 0.45454545454545453,
                },
                wn.synset("state.n.02"): {"frequency": 4, "conf": 0.2, "score": 0.8},
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 5,
                    "conf": 0.7142857142857143,
                    "score": 0.38461538461538464,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.25,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.5,
                },
            },
            "Theme": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Instrument": {
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.25, "score": 1.0},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.125, "score": 1.0},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 2,
                    "conf": 0.25,
                    "score": 0.6666666666666666,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
            },
            "Location": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 8, "conf": 0.6153846153846154},
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.23076923076923078,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.07692307692307693},
            },
            "Patient": {
                wn.synset("plant.n.01"): {"frequency": 3, "conf": 0.08571428571428572},
                wn.synset("person.n.01"): {"frequency": 7, "conf": 0.2},
                wn.synset("currency.n.01"): {"frequency": 1, "conf": 0.02857142857142857},
                wn.synset("region.n.03"): {"frequency": 6, "conf": 0.17142857142857143},
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.05714285714285714,
                },
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.05714285714285714},
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.05714285714285714},
                wn.synset("abstraction.n.06"): {
                    "frequency": 6,
                    "conf": 0.17142857142857143,
                },
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.02857142857142857},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.02857142857142857,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.02857142857142857,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.02857142857142857},
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.02857142857142857},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.02857142857142857},
            },
            "Instrument": {
                wn.synset("natural_phenomenon.n.01"): {"frequency": 2, "conf": 0.4},
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.2},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.2},
            },
            "Location": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.4},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4},
                wn.synset("natural_phenomenon.n.01"): {"frequency": 1, "conf": 0.2},
            },
        },
    },
    "attack": {
        "metaphor": {
            "Theme": {
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.037037037037037035,
                    "score": 0.2,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 4,
                    "conf": 0.14814814814814814,
                    "score": 0.6666666666666666,
                },
                wn.synset("food.n.01"): {
                    "frequency": 1,
                    "conf": 0.037037037037037035,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 10,
                    "conf": 0.37037037037037035,
                    "score": 0.37037037037037035,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 2,
                    "conf": 0.07407407407407407,
                    "score": 0.6666666666666666,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.037037037037037035,
                    "score": 0.5,
                },
                wn.synset("plant.n.02"): {
                    "frequency": 1,
                    "conf": 0.037037037037037035,
                    "score": 0.5,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 2,
                    "conf": 0.07407407407407407,
                    "score": 0.25,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.07407407407407407,
                    "score": 0.13333333333333333,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.037037037037037035,
                    "score": 0.125,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.037037037037037035,
                    "score": 1.0,
                },
                wn.synset("animal.n.01"): {
                    "frequency": 1,
                    "conf": 0.037037037037037035,
                    "score": 0.3333333333333333,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 7,
                    "conf": 0.6363636363636364,
                    "score": 0.2916666666666667,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 0.25,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 0.2,
                },
                wn.synset("animal.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
            },
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2, "score": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 0.5,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.2, "score": 1.0},
            },
            "Attribute": {
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.16666666666666666,
                }
            },
        },
        "literal": {
            "Patient": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.125},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("state.n.04"): {"frequency": 1, "conf": 0.125},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.125},
                wn.synset("organization.n.01"): {"frequency": 2, "conf": 0.25},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.125},
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.125},
                wn.synset("person.n.01"): {"frequency": 17, "conf": 0.53125},
                wn.synset("vehicle.n.01"): {"frequency": 4, "conf": 0.125},
                wn.synset("state.n.03"): {"frequency": 2, "conf": 0.0625},
                wn.synset("organization.n.01"): {"frequency": 3, "conf": 0.09375},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.03125},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.03125},
            },
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 7,
                    "conf": 0.11475409836065574,
                },
                wn.synset("vehicle.n.01"): {"frequency": 3, "conf": 0.04918032786885246},
                wn.synset("person.n.01"): {"frequency": 13, "conf": 0.21311475409836064},
                wn.synset("abstraction.n.06"): {
                    "frequency": 17,
                    "conf": 0.2786885245901639,
                },
                wn.synset("substance.n.01"): {"frequency": 3, "conf": 0.04918032786885246},
                wn.synset("artifact.n.01"): {"frequency": 6, "conf": 0.09836065573770492},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.01639344262295082},
                wn.synset("animal.n.01"): {"frequency": 2, "conf": 0.03278688524590164},
                wn.synset("state.n.02"): {"frequency": 4, "conf": 0.06557377049180328},
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.01639344262295082},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.01639344262295082},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.01639344262295082},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.03278688524590164,
                },
            },
            "Attribute": {
                wn.synset("substance.n.01"): {"frequency": 5, "conf": 0.8333333333333334},
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
        },
    },
    "destroy": {
        "metaphor": {
            "Agent": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 0.125,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.4, "score": 0.2},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 12,
                    "conf": 0.5714285714285714,
                    "score": 0.4444444444444444,
                },
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.14285714285714285,
                    "score": 0.2727272727272727,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.047619047619047616,
                    "score": 0.3333333333333333,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.047619047619047616,
                    "score": 0.5,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.14285714285714285,
                    "score": 0.3333333333333333,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.047619047619047616,
                    "score": 0.25,
                },
            },
        },
        "literal": {
            "Agent": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.05263157894736842,
                },
                wn.synset("person.n.01"): {"frequency": 8, "conf": 0.42105263157894735},
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.3684210526315789,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.05263157894736842},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.05263157894736842},
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.05263157894736842},
            },
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 14, "conf": 0.208955223880597},
                wn.synset("organization.n.01"): {
                    "frequency": 6,
                    "conf": 0.08955223880597014,
                },
                wn.synset("location.n.01"): {"frequency": 2, "conf": 0.029850746268656716},
                wn.synset("abstraction.n.06"): {
                    "frequency": 15,
                    "conf": 0.22388059701492538,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 3,
                    "conf": 0.04477611940298507,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.014925373134328358,
                },
                wn.synset("vehicle.n.01"): {"frequency": 3, "conf": 0.04477611940298507},
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.029850746268656716},
                wn.synset("person.n.01"): {"frequency": 8, "conf": 0.11940298507462686},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.014925373134328358},
                wn.synset("communication.n.02"): {
                    "frequency": 7,
                    "conf": 0.1044776119402985,
                },
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.029850746268656716},
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.014925373134328358},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.014925373134328358},
                wn.synset("plant.n.01"): {"frequency": 1, "conf": 0.014925373134328358},
            },
        },
    },
    "flood": {
        "metaphor": {
            "Patient": {
                wn.synset("region.n.03"): {
                    "frequency": 2,
                    "conf": 0.15384615384615385,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 4,
                    "conf": 0.3076923076923077,
                    "score": 0.5714285714285714,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.23076923076923078,
                    "score": 0.25,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.15384615384615385,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.25,
                    "score": 0.6666666666666666,
                },
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.5, "score": 1.0},
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 0.3333333333333333,
                },
            },
            "Theme": {
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.1, "score": 0.5},
                wn.synset("person.n.01"): {
                    "frequency": 4,
                    "conf": 0.4,
                    "score": 0.6666666666666666,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.1, "score": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.3, "score": 0.6},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.1,
                    "score": 0.2,
                },
            },
        },
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 9,
                    "conf": 0.6923076923076923,
                },
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.23076923076923078},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                },
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.18181818181818182},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.18181818181818182,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 4,
                    "conf": 0.36363636363636365,
                },
                wn.synset("state.n.04"): {"frequency": 1, "conf": 0.09090909090909091},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.09090909090909091},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.09090909090909091},
            },
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
            },
        },
    },
    "grab": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.2,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.26666666666666666,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 7,
                    "conf": 0.4666666666666667,
                    "score": 0.5384615384615384,
                },
                wn.synset("vehicle.n.01"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 17,
                    "conf": 0.5862068965517241,
                    "score": 0.8947368421052632,
                },
                wn.synset("person.n.01"): {
                    "frequency": 5,
                    "conf": 0.1724137931034483,
                    "score": 0.625,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 3,
                    "conf": 0.10344827586206896,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.034482758620689655,
                    "score": 0.5,
                },
                wn.synset("sound.n.01"): {
                    "frequency": 1,
                    "conf": 0.034482758620689655,
                    "score": 1.0,
                },
                wn.synset("clothing.n.01"): {
                    "frequency": 1,
                    "conf": 0.034482758620689655,
                    "score": 0.5,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.034482758620689655,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("tool.n.01"): {"frequency": 1, "conf": 0.08333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.16666666666666666,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.25},
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.08333333333333333},
                wn.synset("body_part.n.01"): {"frequency": 2, "conf": 0.16666666666666666},
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.08333333333333333},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.75},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.125},
            },
        },
    },
    "plant": {
        "metaphor": {
            "Theme": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.16666666666666666,
                    "score": 0.3333333333333333,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 6,
                    "conf": 0.5,
                    "score": 0.5454545454545454,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.16666666666666666,
                    "score": 0.4,
                },
                wn.synset("plant.n.02"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 0.1111111111111111,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
            },
            "Destination": {
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.2, "score": 1.0},
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.6,
                    "score": 0.2727272727272727,
                },
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.2, "score": 1.0},
            },
            "Agent": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.5,
                }
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 11,
                    "conf": 0.3142857142857143,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 4,
                    "conf": 0.11428571428571428,
                },
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.08571428571428572},
                wn.synset("plant.n.02"): {"frequency": 8, "conf": 0.22857142857142856},
                wn.synset("artifact.n.01"): {"frequency": 5, "conf": 0.14285714285714285},
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.05714285714285714,
                },
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.05714285714285714},
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.05263157894736842,
                },
                wn.synset("region.n.03"): {"frequency": 2, "conf": 0.10526315789473684},
                wn.synset("person.n.01"): {"frequency": 13, "conf": 0.6842105263157895},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.10526315789473684,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.05263157894736842,
                },
            },
            "Destination": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.1},
                wn.synset("abstraction.n.06"): {"frequency": 8, "conf": 0.8},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.1},
            },
        },
    },
    "knock": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 9,
                    "conf": 0.42857142857142855,
                    "score": 0.8181818181818182,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 4,
                    "conf": 0.19047619047619047,
                    "score": 0.8,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.047619047619047616,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 6,
                    "conf": 0.2857142857142857,
                    "score": 0.6666666666666666,
                },
                wn.synset("animal.n.01"): {
                    "frequency": 1,
                    "conf": 0.047619047619047616,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 4,
                    "conf": 0.5714285714285714,
                    "score": 0.6666666666666666,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.5,
                },
            },
            "Instrument": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Agent": {
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.25},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.25},
            },
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.375},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.25},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.125},
            },
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "stumble": {
        "metaphor": {
            "Theme": {
                wn.synset("organization.n.01"): {
                    "frequency": 5,
                    "conf": 0.3125,
                    "score": 0.4166666666666667,
                },
                wn.synset("person.n.01"): {
                    "frequency": 7,
                    "conf": 0.4375,
                    "score": 0.3888888888888889,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.125,
                    "score": 0.4,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.0625, "score": 1.0},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.0625, "score": 1.0},
            }
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 11, "conf": 0.4782608695652174},
                wn.synset("organization.n.01"): {
                    "frequency": 7,
                    "conf": 0.30434782608695654,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.13043478260869565,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.043478260869565216},
            },
            "Destination": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "assault": {
        "metaphor": {
            "Patient": {
                wn.synset("region.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.16666666666666666,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            }
        },
        "literal": {
            "Patient": {wn.synset("person.n.01"): {"frequency": 5, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "drink": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 0.16666666666666666,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.2, "score": 1.0},
                wn.synset("food.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 0.05555555555555555,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 2,
                    "conf": 0.4,
                    "score": 0.6666666666666666,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 1.0,
                    "score": 0.15789473684210525,
                }
            },
        },
        "literal": {
            "Patient": {
                wn.synset("food.n.01"): {"frequency": 17, "conf": 0.5151515151515151},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.06060606060606061,
                },
                wn.synset("person.n.01"): {"frequency": 5, "conf": 0.15151515151515152},
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.030303030303030304,
                },
                wn.synset("food.n.02"): {"frequency": 2, "conf": 0.06060606060606061},
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.15151515151515152,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.030303030303030304,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 16, "conf": 0.6956521739130435},
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.17391304347826086,
                },
                wn.synset("time.n.05"): {"frequency": 1, "conf": 0.043478260869565216},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                },
            },
        },
    },
    "kick": {
        "metaphor": {
            "Patient": {
                wn.synset("animal.n.01"): {
                    "frequency": 2,
                    "conf": 0.06896551724137931,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 12,
                    "conf": 0.41379310344827586,
                    "score": 0.9230769230769231,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 4,
                    "conf": 0.13793103448275862,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.034482758620689655,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.10344827586206896,
                    "score": 0.5,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.034482758620689655,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.034482758620689655,
                    "score": 0.5,
                },
                wn.synset("region.n.01"): {
                    "frequency": 1,
                    "conf": 0.034482758620689655,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.034482758620689655,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.034482758620689655,
                    "score": 0.25,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.034482758620689655,
                    "score": 1.0,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.034482758620689655,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 10,
                    "conf": 0.625,
                    "score": 0.8333333333333334,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.1875,
                    "score": 0.6,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.0625,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.0625, "score": 1.0},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.0625, "score": 1.0},
            },
            "Location": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.6666666666666666,
                    "score": 0.6666666666666666,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Patient": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.375},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.375},
            },
            "Location": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}},
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5},
            },
            "Instrument": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "fix": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.4, "score": 0.4},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 9,
                    "conf": 0.5625,
                    "score": 0.42857142857142855,
                },
                wn.synset("state.n.02"): {"frequency": 3, "conf": 0.1875, "score": 0.5},
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.0625,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.125,
                    "score": 1.0,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.0625,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 12,
                    "conf": 0.7058823529411765,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                },
                wn.synset("state.n.02"): {"frequency": 3, "conf": 0.17647058823529413},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.058823529411764705},
            },
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.5},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.16666666666666666},
            },
        },
    },
    "stick": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 18,
                    "conf": 0.45,
                    "score": 0.8181818181818182,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 8,
                    "conf": 0.2,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.025, "score": 0.5},
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.175,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.025,
                    "score": 0.5,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.075,
                    "score": 1.0,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 2,
                    "conf": 0.05,
                    "score": 0.6666666666666666,
                },
            },
            "Destination": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 10,
                    "conf": 0.9090909090909091,
                    "score": 0.7142857142857143,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.5714285714285714},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
            },
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.8},
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.2},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "flourish": {
        "metaphor": {
            "Theme": {
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.3333333333333333,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.16666666666666666,
                },
            }
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 10,
                    "conf": 0.35714285714285715,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 5,
                    "conf": 0.17857142857142858,
                },
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.14285714285714285},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.03571428571428571,
                },
                wn.synset("living_thing.n.01"): {
                    "frequency": 1,
                    "conf": 0.03571428571428571,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.03571428571428571},
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.10714285714285714,
                },
                wn.synset("region.n.03"): {"frequency": 2, "conf": 0.07142857142857142},
                wn.synset("plant.n.01"): {"frequency": 1, "conf": 0.03571428571428571},
            }
        },
    },
    "drag": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.07142857142857142,
                    "score": 0.25,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.023809523809523808,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 16,
                    "conf": 0.38095238095238093,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.047619047619047616,
                    "score": 1.0,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 9,
                    "conf": 0.21428571428571427,
                    "score": 0.9,
                },
                wn.synset("plant.n.02"): {
                    "frequency": 1,
                    "conf": 0.023809523809523808,
                    "score": 1.0,
                },
                wn.synset("region.n.01"): {
                    "frequency": 3,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.023809523809523808,
                    "score": 0.3333333333333333,
                },
                wn.synset("time.n.05"): {
                    "frequency": 1,
                    "conf": 0.023809523809523808,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.023809523809523808,
                    "score": 0.3333333333333333,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.023809523809523808,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 8, "conf": 0.4, "score": 0.8},
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.35,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.15,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.05, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.05,
                    "score": 1.0,
                },
            },
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 9, "conf": 0.6428571428571429},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.07142857142857142},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.14285714285714285,
                },
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.14285714285714285},
            },
            "Agent": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
            },
            "Destination": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Location": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "pass": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "wither": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.07142857142857142,
                }
            }
        },
        "literal": {
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.10526315789473684},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.05263157894736842},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.05263157894736842},
                wn.synset("sound.n.04"): {"frequency": 1, "conf": 0.05263157894736842},
                wn.synset("abstraction.n.06"): {
                    "frequency": 13,
                    "conf": 0.6842105263157895,
                },
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.05263157894736842},
            }
        },
    },
    "evaporate": {
        "metaphor": {
            "Patient": {
                wn.synset("state.n.02"): {
                    "frequency": 5,
                    "conf": 0.29411764705882354,
                    "score": 0.5555555555555556,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 9,
                    "conf": 0.5294117647058824,
                    "score": 0.4090909090909091,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                    "score": 0.5,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                    "score": 0.25,
                },
                wn.synset("communication.n.01"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                    "score": 1.0,
                },
            }
        },
        "literal": {
            "Patient": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 3,
                    "conf": 0.13043478260869565,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 13,
                    "conf": 0.5652173913043478,
                },
                wn.synset("state.n.02"): {"frequency": 4, "conf": 0.17391304347826086},
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.043478260869565216},
            }
        },
    },
    "experience": {
        "metaphor": {},
        "literal": {
            "Stimulus": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "rain": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.5, "score": 1.0},
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
            }
        },
        "literal": {},
    },
    "smooth": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.7777777777777778,
                    "score": 0.875,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 1.0,
                },
                wn.synset("currency.n.01"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.6666666666666666,
                },
                wn.synset("location.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Patient": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "bring": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "drop": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("sound.n.04"): {"frequency": 1, "conf": 1.0}}},
    },
    "worry": {"metaphor": {}, "literal": {}},
    "appropriate": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "drive": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
}
