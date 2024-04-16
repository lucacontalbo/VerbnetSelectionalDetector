import nltk
import pickle
nltk.download('verbnet')
nltk.download('wordnet')

from tqdm import tqdm
from nltk.corpus import verbnet
from nltk.corpus import wordnet as wn

rules = {
    "fail": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.6, "score": 0.5},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 0.5,
                },
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.16666666666666666,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.7142857142857143,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.6},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.2},
            },
        },
    },
    "win": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.5,
                    "score": 0.4,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 0.08333333333333333,
                },
            },
            "Beneficiary": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.25,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5, "score": 0.4},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 0.5,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.6},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.2},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 11, "conf": 0.7333333333333333},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.06666666666666667},
                wn.synset("organization.n.01"): {"frequency": 3, "conf": 0.2},
            },
            "Beneficiary": {
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.5},
            },
        },
    },
    "go": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 35,
                    "conf": 0.5932203389830508,
                    "score": 0.23333333333333334,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.01694915254237288,
                    "score": 0.5,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 2,
                    "conf": 0.03389830508474576,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 15,
                    "conf": 0.2542372881355932,
                    "score": 0.5769230769230769,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.01694915254237288,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 3,
                    "conf": 0.05084745762711865,
                    "score": 0.6,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.01694915254237288,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.01694915254237288,
                    "score": 0.5,
                },
            },
            "Location": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
            },
            "Destination": {
                wn.synset("substance.n.01"): {
                    "frequency": 3,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
                wn.synset("region.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.07692307692307693,
                },
                wn.synset("location.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.037037037037037035,
                },
            },
            "Patient": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.6666666666666666,
                }
            },
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 115, "conf": 0.8712121212121212},
                wn.synset("abstraction.n.06"): {
                    "frequency": 11,
                    "conf": 0.08333333333333333,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.007575757575757576},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.007575757575757576,
                },
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.015151515151515152},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.007575757575757576},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.007575757575757576},
            },
            "Destination": {
                wn.synset("location.n.01"): {"frequency": 26, "conf": 0.41935483870967744},
                wn.synset("abstraction.n.06"): {
                    "frequency": 12,
                    "conf": 0.1935483870967742,
                },
                wn.synset("artifact.n.01"): {"frequency": 6, "conf": 0.0967741935483871},
                wn.synset("region.n.03"): {"frequency": 4, "conf": 0.06451612903225806},
                wn.synset("body_part.n.01"): {"frequency": 7, "conf": 0.11290322580645161},
                wn.synset("substance.n.01"): {"frequency": 6, "conf": 0.0967741935483871},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.016129032258064516,
                },
            },
            "Location": {
                wn.synset("body_part.n.01"): {"frequency": 8, "conf": 0.6153846153846154},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.07692307692307693},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.15384615384615385,
                },
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.15384615384615385},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 12, "conf": 0.9230769230769231},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                },
            },
            "Patient": {
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
            },
        },
    },
    "show": {
        "metaphor": {
            "Topic": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 8,
                    "conf": 0.4,
                    "score": 0.5333333333333333,
                },
                wn.synset("state.n.02"): {"frequency": 4, "conf": 0.2, "score": 0.8},
                wn.synset("state.n.01"): {"frequency": 2, "conf": 0.1, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.05,
                    "score": 0.3333333333333333,
                },
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.05, "score": 0.5},
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.1,
                    "score": 0.3333333333333333,
                },
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.1, "score": 1.0},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
            "Recipient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.2}
            },
        },
        "literal": {
            "Topic": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.16666666666666666},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.041666666666666664},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.041666666666666664},
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.2916666666666667,
                },
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.041666666666666664},
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.041666666666666664},
                wn.synset("substance.n.01"): {"frequency": 3, "conf": 0.125},
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.08333333333333333,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.041666666666666664},
                wn.synset("organization.n.01"): {"frequency": 3, "conf": 0.125},
            },
            "Recipient": {
                wn.synset("region.n.03"): {"frequency": 2, "conf": 0.3333333333333333},
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.6666666666666666},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "climb": {
        "metaphor": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "pour": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("food.n.01"): {"frequency": 3, "conf": 0.75},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.25},
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "demand": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            }
        },
    },
    "dump": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
            "Destination": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "say": {
        "metaphor": {
            "Topic": {
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                    "score": 0.07142857142857142,
                },
                wn.synset("state.n.01"): {
                    "frequency": 2,
                    "conf": 0.1111111111111111,
                    "score": 0.07692307692307693,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 8,
                    "conf": 0.4444444444444444,
                    "score": 0.053691275167785234,
                },
                wn.synset("person.n.01"): {
                    "frequency": 5,
                    "conf": 0.2777777777777778,
                    "score": 0.078125,
                },
                wn.synset("location.n.01"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                    "score": 0.5,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                    "score": 0.1111111111111111,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 10,
                    "conf": 0.7692307692307693,
                    "score": 0.025252525252525252,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.23076923076923078,
                    "score": 0.12,
                },
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 386, "conf": 0.9278846153846154},
                wn.synset("organization.n.01"): {
                    "frequency": 22,
                    "conf": 0.052884615384615384,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.009615384615384616,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.002403846153846154,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.002403846153846154,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.002403846153846154,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.002403846153846154},
            },
            "Topic": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 141,
                    "conf": 0.4684385382059801,
                },
                wn.synset("person.n.01"): {"frequency": 59, "conf": 0.19601328903654486},
                wn.synset("region.n.03"): {"frequency": 3, "conf": 0.009966777408637873},
                wn.synset("state.n.01"): {"frequency": 24, "conf": 0.07973421926910298},
                wn.synset("state.n.02"): {"frequency": 13, "conf": 0.04318936877076412},
                wn.synset("substance.n.01"): {
                    "frequency": 22,
                    "conf": 0.07308970099667775,
                },
                wn.synset("artifact.n.01"): {"frequency": 8, "conf": 0.026578073089700997},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 4,
                    "conf": 0.013289036544850499,
                },
                wn.synset("region.n.01"): {"frequency": 2, "conf": 0.006644518272425249},
                wn.synset("communication.n.02"): {
                    "frequency": 14,
                    "conf": 0.046511627906976744,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.006644518272425249,
                },
                wn.synset("location.n.01"): {
                    "frequency": 1,
                    "conf": 0.0033222591362126247,
                },
                wn.synset("plant.n.01"): {"frequency": 1, "conf": 0.0033222591362126247},
                wn.synset("idea.n.01"): {"frequency": 6, "conf": 0.019933554817275746},
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.0033222591362126247,
                },
            },
            "Recipient": {wn.synset("state.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "bear": {"metaphor": {}, "literal": {}},
    "get": {
        "metaphor": {
            "Patient": {
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.11538461538461539,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.19230769230769232,
                    "score": 0.45454545454545453,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 2,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 11,
                    "conf": 0.4230769230769231,
                    "score": 0.22916666666666666,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                    "score": 1.0,
                },
                wn.synset("vehicle.n.01"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.01"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                    "score": 1.0,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                    "score": 0.5,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 62,
                    "conf": 0.9393939393939394,
                    "score": 0.31958762886597936,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.015151515151515152,
                    "score": 1.0,
                },
                wn.synset("region.n.03"): {
                    "frequency": 2,
                    "conf": 0.030303030303030304,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.015151515151515152,
                    "score": 0.16666666666666666,
                },
            },
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 19,
                    "conf": 0.16964285714285715,
                    "score": 0.48717948717948717,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.017857142857142856,
                    "score": 0.6666666666666666,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 4,
                    "conf": 0.03571428571428571,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 3,
                    "conf": 0.026785714285714284,
                    "score": 0.3,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 5,
                    "conf": 0.044642857142857144,
                    "score": 0.5555555555555556,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 46,
                    "conf": 0.4107142857142857,
                    "score": 0.41818181818181815,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 2,
                    "conf": 0.017857142857142856,
                    "score": 0.06896551724137931,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 14,
                    "conf": 0.125,
                    "score": 0.7368421052631579,
                },
                wn.synset("location.n.01"): {
                    "frequency": 1,
                    "conf": 0.008928571428571428,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.01"): {
                    "frequency": 3,
                    "conf": 0.026785714285714284,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 5,
                    "conf": 0.044642857142857144,
                    "score": 0.8333333333333334,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 3,
                    "conf": 0.026785714285714284,
                    "score": 0.75,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.008928571428571428,
                    "score": 1.0,
                },
                wn.synset("sound.n.01"): {
                    "frequency": 1,
                    "conf": 0.008928571428571428,
                    "score": 1.0,
                },
                wn.synset("food.n.01"): {
                    "frequency": 3,
                    "conf": 0.026785714285714284,
                    "score": 0.3,
                },
            },
            "Beneficiary": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.125}
            },
            "Asset": {
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
            },
        },
        "literal": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 5,
                    "conf": 0.032679738562091505,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 64,
                    "conf": 0.41830065359477125,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.006535947712418301},
                wn.synset("artifact.n.01"): {"frequency": 27, "conf": 0.17647058823529413},
                wn.synset("substance.n.01"): {"frequency": 7, "conf": 0.0457516339869281},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 4,
                    "conf": 0.026143790849673203,
                },
                wn.synset("person.n.01"): {"frequency": 20, "conf": 0.13071895424836602},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.006535947712418301},
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.006535947712418301},
                wn.synset("food.n.01"): {"frequency": 7, "conf": 0.0457516339869281},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.006535947712418301},
                wn.synset("food.n.02"): {"frequency": 3, "conf": 0.0196078431372549},
                wn.synset("clothing.n.01"): {"frequency": 8, "conf": 0.05228758169934641},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.006535947712418301},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.006535947712418301},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.006535947712418301,
                },
                wn.synset("sound.n.04"): {"frequency": 1, "conf": 0.006535947712418301},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 132, "conf": 0.9496402877697842},
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.03597122302158273,
                },
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.007194244604316547},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.007194244604316547},
            },
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 37, "conf": 0.74},
                wn.synset("abstraction.n.06"): {"frequency": 6, "conf": 0.12},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.02},
                wn.synset("idea.n.01"): {"frequency": 3, "conf": 0.06},
                wn.synset("physical_entity.n.01"): {"frequency": 2, "conf": 0.04},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.02},
            },
            "Beneficiary": {wn.synset("person.n.01"): {"frequency": 7, "conf": 1.0}},
        },
    },
    "take": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 34,
                    "conf": 0.918918918918919,
                    "score": 0.5964912280701754,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.05405405405405406,
                    "score": 0.6666666666666666,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.02702702702702703,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 37,
                    "conf": 0.3627450980392157,
                    "score": 0.8043478260869565,
                },
                wn.synset("state.n.02"): {
                    "frequency": 6,
                    "conf": 0.058823529411764705,
                    "score": 0.75,
                },
                wn.synset("location.n.01"): {
                    "frequency": 2,
                    "conf": 0.0196078431372549,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 5,
                    "conf": 0.049019607843137254,
                    "score": 1.0,
                },
                wn.synset("time.n.03"): {
                    "frequency": 2,
                    "conf": 0.0196078431372549,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 8,
                    "conf": 0.0784313725490196,
                    "score": 0.8888888888888888,
                },
                wn.synset("person.n.01"): {
                    "frequency": 16,
                    "conf": 0.1568627450980392,
                    "score": 0.3902439024390244,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 5,
                    "conf": 0.049019607843137254,
                    "score": 0.625,
                },
                wn.synset("time.n.01"): {
                    "frequency": 4,
                    "conf": 0.0392156862745098,
                    "score": 0.8,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 3,
                    "conf": 0.029411764705882353,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 9,
                    "conf": 0.08823529411764706,
                    "score": 0.75,
                },
                wn.synset("communication.n.01"): {
                    "frequency": 1,
                    "conf": 0.00980392156862745,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 3,
                    "conf": 0.029411764705882353,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.00980392156862745,
                    "score": 0.5,
                },
            },
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 9, "conf": 1.0, "score": 0.9}
            },
            "Attribute": {
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.5, "score": 1.0},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5, "score": 1.0},
            },
            "Instrument": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.6, "score": 0.6},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4, "score": 1.0},
            },
            "Asset": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Destination": {
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 0.5},
            },
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 25, "conf": 0.5102040816326531},
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.04081632653061224},
                wn.synset("abstraction.n.06"): {
                    "frequency": 9,
                    "conf": 0.1836734693877551,
                },
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.061224489795918366},
                wn.synset("time.n.01"): {"frequency": 1, "conf": 0.02040816326530612},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.02040816326530612,
                },
                wn.synset("region.n.01"): {"frequency": 2, "conf": 0.04081632653061224},
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.061224489795918366,
                },
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.02040816326530612},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.02040816326530612,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.02040816326530612},
            },
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.42857142857142855},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.2857142857142857},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                },
            },
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
            },
            "Instrument": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 23, "conf": 0.8846153846153846},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.07692307692307693,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                },
            },
            "Location": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Asset": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "find": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.08108108108108109,
                    "score": 0.75,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 20,
                    "conf": 0.5405405405405406,
                    "score": 0.7407407407407407,
                },
                wn.synset("person.n.01"): {
                    "frequency": 7,
                    "conf": 0.1891891891891892,
                    "score": 0.3181818181818182,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.02702702702702703,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.02702702702702703,
                    "score": 0.5,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.02702702702702703,
                    "score": 0.5,
                },
                wn.synset("state.n.01"): {
                    "frequency": 3,
                    "conf": 0.08108108108108109,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.02702702702702703,
                    "score": 0.5,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 25,
                    "conf": 0.78125,
                    "score": 0.6944444444444444,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.09375,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.03125,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.0625,
                    "score": 0.6666666666666666,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.03125,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.19444444444444445,
                },
                wn.synset("person.n.01"): {"frequency": 15, "conf": 0.4166666666666667},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.027777777777777776},
                wn.synset("artifact.n.01"): {"frequency": 5, "conf": 0.1388888888888889},
                wn.synset("location.n.01"): {"frequency": 2, "conf": 0.05555555555555555},
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.027777777777777776,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.027777777777777776,
                },
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.027777777777777776},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.027777777777777776,
                },
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.027777777777777776},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.027777777777777776},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 11, "conf": 0.9166666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                },
            },
        },
    },
    "disappear": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            }
        },
    },
    "omit": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "buy": {
        "metaphor": {
            "Theme": {
                wn.synset("time.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.1111111111111111,
                },
            },
            "Beneficiary": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.16666666666666666,
                }
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.03571428571428571,
                }
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 12,
                    "conf": 0.2857142857142857,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 2,
                    "conf": 0.047619047619047616,
                },
                wn.synset("person.n.01"): {"frequency": 8, "conf": 0.19047619047619047},
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.047619047619047616,
                },
                wn.synset("food.n.01"): {"frequency": 2, "conf": 0.047619047619047616},
                wn.synset("artifact.n.01"): {"frequency": 12, "conf": 0.2857142857142857},
                wn.synset("clothing.n.01"): {"frequency": 2, "conf": 0.047619047619047616},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.023809523809523808},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.023809523809523808,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 27, "conf": 0.9},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.03333333333333333},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.03333333333333333,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.03333333333333333,
                },
            },
            "Asset": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
            "Beneficiary": {wn.synset("person.n.01"): {"frequency": 5, "conf": 1.0}},
        },
    },
    "reflect": {"metaphor": {}, "literal": {}},
    "render": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "turn": {
        "metaphor": {
            "Patient": {
                wn.synset("machine.n.04"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 2,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.25,
                    "score": 0.42857142857142855,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0, "score": 0.5}
            },
            "Theme": {
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.08333333333333333,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 11, "conf": 0.7333333333333333},
                wn.synset("body_part.n.01"): {"frequency": 2, "conf": 0.13333333333333333},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.13333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0}},
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.6666666666666666},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
        },
    },
    "want": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.8333333333333334,
                    "score": 0.058823529411764705,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.038461538461538464,
                },
            }
        },
        "literal": {
            "Theme": {
                wn.synset("idea.n.01"): {"frequency": 8, "conf": 0.050314465408805034},
                wn.synset("person.n.01"): {"frequency": 25, "conf": 0.15723270440251572},
                wn.synset("abstraction.n.06"): {
                    "frequency": 80,
                    "conf": 0.5031446540880503,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 10,
                    "conf": 0.06289308176100629,
                },
                wn.synset("artifact.n.01"): {"frequency": 5, "conf": 0.031446540880503145},
                wn.synset("state.n.02"): {"frequency": 6, "conf": 0.03773584905660377},
                wn.synset("food.n.01"): {"frequency": 3, "conf": 0.018867924528301886},
                wn.synset("sound.n.01"): {"frequency": 1, "conf": 0.006289308176100629},
                wn.synset("time.n.03"): {"frequency": 8, "conf": 0.050314465408805034},
                wn.synset("region.n.03"): {"frequency": 4, "conf": 0.025157232704402517},
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.018867924528301886,
                },
                wn.synset("plant.n.01"): {"frequency": 1, "conf": 0.006289308176100629},
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.006289308176100629},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.012578616352201259,
                },
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.006289308176100629},
                wn.synset("currency.n.01"): {"frequency": 1, "conf": 0.006289308176100629},
            }
        },
    },
    "let": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 7,
                    "conf": 1.0,
                    "score": 0.3888888888888889,
                }
            },
            "Theme": {
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 0.3333333333333333,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 2,
                    "conf": 0.2222222222222222,
                    "score": 1.0,
                },
                wn.synset("food.n.01"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.4444444444444444,
                    "score": 0.13793103448275862,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 0.1111111111111111,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.045454545454545456},
                wn.synset("abstraction.n.06"): {
                    "frequency": 25,
                    "conf": 0.5681818181818182,
                },
                wn.synset("artifact.n.01"): {"frequency": 5, "conf": 0.11363636363636363},
                wn.synset("person.n.01"): {"frequency": 8, "conf": 0.18181818181818182},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.022727272727272728},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.022727272727272728,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.022727272727272728},
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.022727272727272728,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.14285714285714285,
                },
                wn.synset("person.n.01"): {"frequency": 11, "conf": 0.7857142857142857},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                },
            },
        },
    },
    "bother": {"metaphor": {}, "literal": {}},
    "come": {
        "metaphor": {
            "Theme": {
                wn.synset("state.n.02"): {
                    "frequency": 2,
                    "conf": 0.04081632653061224,
                    "score": 0.6666666666666666,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.04081632653061224,
                    "score": 0.6666666666666666,
                },
                wn.synset("person.n.01"): {
                    "frequency": 13,
                    "conf": 0.2653061224489796,
                    "score": 0.25,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 16,
                    "conf": 0.32653061224489793,
                    "score": 0.8,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 5,
                    "conf": 0.10204081632653061,
                    "score": 0.23809523809523808,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 2,
                    "conf": 0.04081632653061224,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.061224489795918366,
                    "score": 1.0,
                },
                wn.synset("food.n.01"): {
                    "frequency": 1,
                    "conf": 0.02040816326530612,
                    "score": 1.0,
                },
                wn.synset("sound.n.01"): {
                    "frequency": 2,
                    "conf": 0.04081632653061224,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.02040816326530612,
                    "score": 1.0,
                },
                wn.synset("sound.n.04"): {
                    "frequency": 1,
                    "conf": 0.02040816326530612,
                    "score": 0.5,
                },
                wn.synset("plant.n.01"): {
                    "frequency": 1,
                    "conf": 0.02040816326530612,
                    "score": 1.0,
                },
            },
            "Destination": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.36363636363636365,
                    "score": 0.3076923076923077,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 2,
                    "conf": 0.18181818181818182,
                    "score": 0.2,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 4,
                    "conf": 0.36363636363636365,
                    "score": 0.2222222222222222,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.21428571428571427,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 6,
                    "conf": 0.42857142857142855,
                    "score": 0.2,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
                wn.synset("time.n.01"): {
                    "frequency": 2,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
            },
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 24, "conf": 0.8},
                wn.synset("natural_phenomenon.n.01"): {"frequency": 6, "conf": 0.2},
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 39, "conf": 0.6},
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.06153846153846154,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 16,
                    "conf": 0.24615384615384617,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.015384615384615385,
                },
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.015384615384615385},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.015384615384615385},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.015384615384615385},
                wn.synset("sound.n.04"): {"frequency": 1, "conf": 0.015384615384615385},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.015384615384615385},
            },
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 9, "conf": 0.1875},
                wn.synset("location.n.01"): {"frequency": 14, "conf": 0.2916666666666667},
                wn.synset("substance.n.01"): {"frequency": 14, "conf": 0.2916666666666667},
                wn.synset("body_part.n.01"): {"frequency": 8, "conf": 0.16666666666666666},
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.041666666666666664},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.020833333333333332},
            },
            "Location": {wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "inform": {
        "metaphor": {
            "Recipient": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Recipient": {
                wn.synset("person.n.01"): {"frequency": 8, "conf": 0.8888888888888888},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.1111111111111111},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Topic": {
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "preserve": {
        "metaphor": {
            "Theme": {
                wn.synset("animal.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.25,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.25,
                },
            },
            "Agent": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
        },
        "literal": {
            "Theme": {
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.375},
                wn.synset("physical_entity.n.01"): {"frequency": 3, "conf": 0.375},
                wn.synset("living_thing.n.01"): {"frequency": 1, "conf": 0.125},
            }
        },
    },
    "see": {
        "metaphor": {
            "Stimulus": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 18,
                    "conf": 0.5294117647058824,
                    "score": 0.36,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 2,
                    "conf": 0.058823529411764705,
                    "score": 0.6666666666666666,
                },
                wn.synset("person.n.01"): {
                    "frequency": 8,
                    "conf": 0.23529411764705882,
                    "score": 0.1038961038961039,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 3,
                    "conf": 0.08823529411764706,
                    "score": 0.23076923076923078,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.029411764705882353,
                    "score": 0.1111111111111111,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.029411764705882353,
                    "score": 0.3333333333333333,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.029411764705882353,
                    "score": 0.3333333333333333,
                },
            }
        },
        "literal": {
            "Stimulus": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 32,
                    "conf": 0.2191780821917808,
                },
                wn.synset("person.n.01"): {"frequency": 69, "conf": 0.4726027397260274},
                wn.synset("communication.n.02"): {
                    "frequency": 8,
                    "conf": 0.0547945205479452,
                },
                wn.synset("region.n.03"): {"frequency": 2, "conf": 0.0136986301369863},
                wn.synset("artifact.n.01"): {"frequency": 10, "conf": 0.0684931506849315},
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.00684931506849315},
                wn.synset("substance.n.01"): {"frequency": 8, "conf": 0.0547945205479452},
                wn.synset("body_part.n.01"): {"frequency": 2, "conf": 0.0136986301369863},
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.00684931506849315},
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.0136986301369863},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.00684931506849315},
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.00684931506849315,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 6,
                    "conf": 0.0410958904109589,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.00684931506849315},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.00684931506849315,
                },
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.00684931506849315},
            }
        },
    },
    "become": {
        "metaphor": {
            "Patient": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 16, "conf": 0.32},
                wn.synset("state.n.02"): {"frequency": 3, "conf": 0.06},
                wn.synset("person.n.01"): {"frequency": 20, "conf": 0.4},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.04},
                wn.synset("organization.n.01"): {"frequency": 4, "conf": 0.08},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.02},
                wn.synset("natural_phenomenon.n.01"): {"frequency": 1, "conf": 0.02},
                wn.synset("physical_entity.n.01"): {"frequency": 2, "conf": 0.04},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.02},
            }
        },
    },
    "ensure": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.14285714285714285,
                }
            }
        },
        "literal": {
            "Theme": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                },
                wn.synset("state.n.02"): {"frequency": 3, "conf": 0.23076923076923078},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.07692307692307693},
                wn.synset("abstraction.n.06"): {
                    "frequency": 6,
                    "conf": 0.46153846153846156,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.07692307692307693},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                },
            }
        },
    },
    "laugh": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 6, "conf": 1.0}},
            "Recipient": {wn.synset("substance.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "know": {
        "metaphor": {
            "Stimulus": {
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 0.07692307692307693,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.75,
                    "score": 0.04918032786885246,
                },
            }
        },
        "literal": {
            "Stimulus": {
                wn.synset("state.n.01"): {"frequency": 20, "conf": 0.11428571428571428},
                wn.synset("location.n.01"): {"frequency": 3, "conf": 0.017142857142857144},
                wn.synset("abstraction.n.06"): {
                    "frequency": 58,
                    "conf": 0.3314285714285714,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 12,
                    "conf": 0.06857142857142857,
                },
                wn.synset("person.n.01"): {"frequency": 50, "conf": 0.2857142857142857},
                wn.synset("communication.n.02"): {
                    "frequency": 5,
                    "conf": 0.02857142857142857,
                },
                wn.synset("state.n.02"): {"frequency": 5, "conf": 0.02857142857142857},
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.005714285714285714},
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.017142857142857144},
                wn.synset("idea.n.01"): {"frequency": 3, "conf": 0.017142857142857144},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 6,
                    "conf": 0.03428571428571429,
                },
                wn.synset("sound.n.01"): {"frequency": 1, "conf": 0.005714285714285714},
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.005714285714285714,
                },
                wn.synset("plant.n.01"): {"frequency": 2, "conf": 0.011428571428571429},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.005714285714285714,
                },
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.005714285714285714},
                wn.synset("region.n.03"): {"frequency": 2, "conf": 0.011428571428571429},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.005714285714285714},
            }
        },
    },
    "acquire": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.6666666666666666,
                }
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.6666666666666666,
                }
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "repair": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "forget": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.45454545454545453,
                },
                wn.synset("person.n.01"): {"frequency": 5, "conf": 0.45454545454545453},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.8571428571428571},
            },
        },
    },
    "tell": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 6,
                    "conf": 0.6,
                    "score": 0.06818181818181818,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.1,
                    "score": 0.5,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.2,
                    "score": 0.6666666666666666,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.1,
                    "score": 0.25,
                },
            },
            "Recipient": {
                wn.synset("person.n.01"): {
                    "frequency": 6,
                    "conf": 0.46153846153846156,
                    "score": 0.05454545454545454,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 0.3333333333333333,
                },
                wn.synset("region.n.03"): {
                    "frequency": 3,
                    "conf": 0.23076923076923078,
                    "score": 0.375,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.15384615384615385,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 0.25,
                },
            },
            "Topic": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.1,
                    "score": 0.16666666666666666,
                },
                wn.synset("state.n.01"): {
                    "frequency": 2,
                    "conf": 0.2,
                    "score": 0.2222222222222222,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 6,
                    "conf": 0.6,
                    "score": 0.15384615384615385,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.1, "score": 1.0},
            },
        },
        "literal": {
            "Topic": {
                wn.synset("person.n.01"): {"frequency": 21, "conf": 0.2625},
                wn.synset("communication.n.02"): {"frequency": 5, "conf": 0.0625},
                wn.synset("abstraction.n.06"): {"frequency": 33, "conf": 0.4125},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.0125},
                wn.synset("state.n.01"): {"frequency": 7, "conf": 0.0875},
                wn.synset("substance.n.01"): {"frequency": 5, "conf": 0.0625},
                wn.synset("state.n.02"): {"frequency": 6, "conf": 0.075},
                wn.synset("time.n.05"): {"frequency": 1, "conf": 0.0125},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.0125},
            },
            "Recipient": {
                wn.synset("person.n.01"): {"frequency": 104, "conf": 0.9122807017543859},
                wn.synset("region.n.03"): {"frequency": 5, "conf": 0.043859649122807015},
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.02631578947368421,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.017543859649122806,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 82, "conf": 0.9425287356321839},
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.034482758620689655,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.011494252873563218,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.011494252873563218,
                },
            },
        },
    },
    "watch": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.05},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 0.2},
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.047619047619047616,
                },
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 19, "conf": 0.7037037037037037},
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.037037037037037035,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.14814814814814814,
                },
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.037037037037037035},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.037037037037037035},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.037037037037037035},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 20, "conf": 0.9523809523809523},
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.047619047619047616},
            },
        },
    },
    "think": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.16666666666666666,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.004608294930875576,
                },
            },
            "Theme": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.018867924528301886,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.011494252873563218,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("idea.n.01"): {"frequency": 5, "conf": 0.02631578947368421},
                wn.synset("substance.n.01"): {
                    "frequency": 12,
                    "conf": 0.06315789473684211,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 5,
                    "conf": 0.02631578947368421,
                },
                wn.synset("artifact.n.01"): {"frequency": 5, "conf": 0.02631578947368421},
                wn.synset("person.n.01"): {"frequency": 52, "conf": 0.2736842105263158},
                wn.synset("abstraction.n.06"): {
                    "frequency": 86,
                    "conf": 0.45263157894736844,
                },
                wn.synset("state.n.02"): {"frequency": 3, "conf": 0.015789473684210527},
                wn.synset("state.n.01"): {"frequency": 15, "conf": 0.07894736842105263},
                wn.synset("food.n.01"): {"frequency": 3, "conf": 0.015789473684210527},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.005263157894736842},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.005263157894736842,
                },
                wn.synset("region.n.03"): {"frequency": 2, "conf": 0.010526315789473684},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 216, "conf": 0.9557522123893806},
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.022123893805309734,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 5,
                    "conf": 0.022123893805309734,
                },
            },
            "Attribute": {
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "justify": {
        "metaphor": {},
        "literal": {
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.5},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.16666666666666666},
            }
        },
    },
    "carry": {
        "metaphor": {
            "Location": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                    "score": 0.6666666666666666,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.25,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.5,
                },
            },
            "Agent": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 5, "conf": 1.0}},
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.5},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
        },
    },
    "overlap": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "occupy": {"metaphor": {}, "literal": {}},
    "chronicle": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "involve": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 16,
                    "conf": 0.8888888888888888,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.05555555555555555},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                },
            }
        },
    },
    "mean": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.5, "score": 0.5},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.5,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.3333333333333333,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 0.009174311926605505,
                },
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5, "score": 1.0},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25, "score": 1.0},
            },
            "Topic": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 108, "conf": 1.0}},
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.375},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.25},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.125},
            },
            "Topic": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "feel": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 8,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 0.5,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 12,
                    "conf": 0.9230769230769231,
                    "score": 0.8571428571428571,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
            },
            "Stimulus": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.75, "score": 0.75},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Stimulus": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "judge": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "hold": {
        "metaphor": {
            "Theme": {
                wn.synset("time.n.01"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 2,
                    "conf": 0.08695652173913043,
                    "score": 0.5,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 8,
                    "conf": 0.34782608695652173,
                    "score": 0.5333333333333333,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 2,
                    "conf": 0.08695652173913043,
                    "score": 1.0,
                },
                wn.synset("communication.n.01"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.13043478260869565,
                    "score": 0.6,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                    "score": 1.0,
                },
                wn.synset("location.n.01"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.08695652173913043,
                    "score": 0.6666666666666666,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                    "score": 0.3333333333333333,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 9, "conf": 0.75, "score": 0.5625},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 0.5,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.3181818181818182,
                },
                wn.synset("body_part.n.01"): {"frequency": 8, "conf": 0.36363636363636365},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.09090909090909091},
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.09090909090909091,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.09090909090909091},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.045454545454545456,
                },
            },
            "Agent": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                },
                wn.synset("person.n.01"): {"frequency": 7, "conf": 0.7777777777777778},
                wn.synset("sound.n.04"): {"frequency": 1, "conf": 0.1111111111111111},
            },
        },
    },
    "seek": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5, "score": 0.4},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.4, "score": 1.0},
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.6, "score": 0.6},
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.2},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.2},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.2},
            },
        },
    },
    "prepare": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Patient": {
                wn.synset("food.n.02"): {"frequency": 1, "conf": 0.2},
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.4},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.2},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.2},
            },
        },
    },
    "cease": {
        "metaphor": {
            "Agent": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "remain": {
        "metaphor": {
            "Theme": {
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                    "score": 0.6666666666666666,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.42857142857142855,
                    "score": 0.5,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.5,
                },
            },
            "Attribute": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.21428571428571427,
                },
                wn.synset("person.n.01"): {"frequency": 8, "conf": 0.5714285714285714},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.07142857142857142},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.07142857142857142},
            }
        },
    },
    "seem": {
        "metaphor": {},
        "literal": {
            "Attribute": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.027777777777777776,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 2,
                    "conf": 0.027777777777777776,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 23,
                    "conf": 0.3194444444444444,
                },
                wn.synset("person.n.01"): {"frequency": 26, "conf": 0.3611111111111111},
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.041666666666666664,
                },
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.027777777777777776},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.027777777777777776,
                },
                wn.synset("location.n.01"): {"frequency": 4, "conf": 0.05555555555555555},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.013888888888888888},
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.041666666666666664},
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.027777777777777776},
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.013888888888888888},
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.013888888888888888},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 20,
                    "conf": 0.5405405405405406,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.02702702702702703},
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.10810810810810811},
                wn.synset("substance.n.01"): {"frequency": 9, "conf": 0.24324324324324326},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.05405405405405406},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.02702702702702703,
                },
            },
        },
    },
    "uproot": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("plant.n.02"): {"frequency": 1, "conf": 1.0}}},
    },
    "matter": {"metaphor": {}, "literal": {}},
    "include": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.045454545454545456,
                    "score": 0.25,
                },
                wn.synset("location.n.01"): {
                    "frequency": 2,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 16,
                    "conf": 0.7272727272727273,
                    "score": 0.43243243243243246,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
                wn.synset("food.n.01"): {
                    "frequency": 1,
                    "conf": 0.045454545454545456,
                    "score": 1.0,
                },
            },
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.25}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 8, "conf": 0.1702127659574468},
                wn.synset("region.n.03"): {"frequency": 2, "conf": 0.0425531914893617},
                wn.synset("abstraction.n.06"): {
                    "frequency": 21,
                    "conf": 0.44680851063829785,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 5,
                    "conf": 0.10638297872340426,
                },
                wn.synset("state.n.02"): {"frequency": 4, "conf": 0.0851063829787234},
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.06382978723404255},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.0425531914893617,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.02127659574468085},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.02127659574468085},
            },
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.75},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.25},
            },
        },
    },
    "state": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Topic": {wn.synset("state.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "sit": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.10526315789473684,
                }
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 17, "conf": 0.9444444444444444},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                },
            },
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "catch": {
        "metaphor": {
            "Theme": {
                wn.synset("body_part.n.01"): {
                    "frequency": 2,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.35714285714285715,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.21428571428571427,
                    "score": 0.42857142857142855,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 2,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.75, "score": 0.75},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.125, "score": 1.0},
            },
        },
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "earn": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0}},
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 1.0}},
        },
    },
    "hear": {
        "metaphor": {
            "Stimulus": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            }
        },
        "literal": {
            "Stimulus": {
                wn.synset("person.n.01"): {"frequency": 7, "conf": 0.28},
                wn.synset("sound.n.04"): {"frequency": 2, "conf": 0.08},
                wn.synset("sound.n.01"): {"frequency": 4, "conf": 0.16},
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.16},
                wn.synset("machine.n.01"): {"frequency": 1, "conf": 0.04},
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.12},
                wn.synset("communication.n.02"): {"frequency": 2, "conf": 0.08},
                wn.synset("region.n.03"): {"frequency": 2, "conf": 0.08},
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "float": {
        "metaphor": {
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "discuss": {
        "metaphor": {
            "Agent": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.16666666666666666,
                },
            }
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 5, "conf": 0.8333333333333334},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
            }
        },
    },
    "draw": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 11,
                    "conf": 0.7857142857142857,
                    "score": 0.7857142857142857,
                },
                wn.synset("time.n.05"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.25, "score": 0.5},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.25, "score": 1.0},
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.25, "score": 1.0},
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.5,
                    "score": 0.8333333333333334,
                },
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.4, "score": 1.0},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.1,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "glance": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 4, "conf": 0.8},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.2},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "obviate": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "lead": {
        "metaphor": {
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 4,
                    "conf": 0.4444444444444444,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.3333333333333333,
                    "score": 0.6,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.2222222222222222,
                    "score": 0.6666666666666666,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.6, "score": 1.0},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2, "score": 0.5},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.2, "score": 1.0},
            },
            "Patient": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "need": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.07692307692307693,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.16666666666666666,
                },
            }
        },
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 4, "conf": 0.06153846153846154},
                wn.synset("state.n.02"): {"frequency": 3, "conf": 0.046153846153846156},
                wn.synset("person.n.01"): {"frequency": 12, "conf": 0.18461538461538463},
                wn.synset("abstraction.n.06"): {
                    "frequency": 24,
                    "conf": 0.36923076923076925,
                },
                wn.synset("abstraction.n.01"): {
                    "frequency": 1,
                    "conf": 0.015384615384615385,
                },
                wn.synset("food.n.01"): {"frequency": 5, "conf": 0.07692307692307693},
                wn.synset("substance.n.01"): {"frequency": 5, "conf": 0.07692307692307693},
                wn.synset("location.n.01"): {"frequency": 3, "conf": 0.046153846153846156},
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.03076923076923077},
                wn.synset("communication.n.02"): {
                    "frequency": 5,
                    "conf": 0.07692307692307693,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.015384615384615385,
                },
            }
        },
    },
    "make": {
        "metaphor": {
            "Product": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.029411764705882353,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.029411764705882353,
                    "score": 0.5,
                },
                wn.synset("time.n.05"): {
                    "frequency": 1,
                    "conf": 0.029411764705882353,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 4,
                    "conf": 0.11764705882352941,
                    "score": 0.8,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 3,
                    "conf": 0.08823529411764706,
                    "score": 0.6,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 3,
                    "conf": 0.08823529411764706,
                    "score": 0.75,
                },
                wn.synset("sound.n.04"): {
                    "frequency": 4,
                    "conf": 0.11764705882352941,
                    "score": 0.8,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.029411764705882353,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 11,
                    "conf": 0.3235294117647059,
                    "score": 0.6875,
                },
                wn.synset("time.n.01"): {
                    "frequency": 1,
                    "conf": 0.029411764705882353,
                    "score": 1.0,
                },
                wn.synset("plant.n.02"): {
                    "frequency": 1,
                    "conf": 0.029411764705882353,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.029411764705882353,
                    "score": 1.0,
                },
                wn.synset("substance.n.07"): {
                    "frequency": 1,
                    "conf": 0.029411764705882353,
                    "score": 1.0,
                },
                wn.synset("state.n.01"): {
                    "frequency": 1,
                    "conf": 0.029411764705882353,
                    "score": 0.5,
                },
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 17,
                    "conf": 0.5151515151515151,
                    "score": 0.85,
                },
                wn.synset("state.n.02"): {
                    "frequency": 9,
                    "conf": 0.2727272727272727,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.030303030303030304,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 13,
                    "conf": 0.7647058823529411,
                    "score": 0.5909090909090909,
                },
                wn.synset("time.n.05"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                    "score": 0.3333333333333333,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.8, "score": 1.0},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2, "score": 1.0},
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 9, "conf": 0.75},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.08333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.16666666666666666,
                },
            },
            "Theme": {
                wn.synset("physical_entity.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.5},
            },
            "Product": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.29411764705882354,
                },
                wn.synset("food.n.01"): {"frequency": 3, "conf": 0.17647058823529413},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.058823529411764705},
                wn.synset("sound.n.04"): {"frequency": 1, "conf": 0.058823529411764705},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.11764705882352941},
                wn.synset("food.n.02"): {"frequency": 1, "conf": 0.058823529411764705},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.058823529411764705},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.058823529411764705},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.058823529411764705},
            },
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 1.0}},
        },
    },
    "sound": {
        "metaphor": {},
        "literal": {
            "Stimulus": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.07692307692307693},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                },
                wn.synset("person.n.01"): {"frequency": 8, "conf": 0.6153846153846154},
                wn.synset("sound.n.01"): {"frequency": 1, "conf": 0.07692307692307693},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                },
                wn.synset("sound.n.04"): {"frequency": 1, "conf": 0.07692307692307693},
            }
        },
    },
    "drive": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.1,
                },
            },
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            },
        },
        "literal": {
            "Theme": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.1},
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.6},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.2},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.1},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 9, "conf": 1.0}},
        },
    },
    "concern": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "set": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.2727272727272727,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.36363636363636365,
                    "score": 0.6666666666666666,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 2,
                    "conf": 0.18181818181818182,
                    "score": 1.0,
                },
                wn.synset("state.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
                wn.synset("machine.n.04"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 5, "conf": 1.0, "score": 1.0}
            },
            "Destination": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.5714285714285714,
                    "score": 0.6666666666666666,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                    "score": 0.6666666666666666,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Destination": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
        },
    },
    "look": {
        "metaphor": {
            "Theme": {
                wn.synset("substance.n.01"): {
                    "frequency": 12,
                    "conf": 0.75,
                    "score": 0.23076923076923078,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.25,
                    "score": 0.4444444444444444,
                },
            },
            "Stimulus": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.75, "score": 0.125},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 0.2,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 6,
                    "conf": 0.75,
                    "score": 0.10526315789473684,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 0.3333333333333333,
                },
            },
            "Location": {
                wn.synset("state.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 51, "conf": 0.9622641509433962},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.03773584905660377,
                },
            },
            "Stimulus": {
                wn.synset("person.n.01"): {"frequency": 21, "conf": 0.75},
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.14285714285714285,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.03571428571428571},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.03571428571428571,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.03571428571428571},
            },
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 40, "conf": 0.7142857142857143},
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.03571428571428571},
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.08928571428571429,
                },
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.03571428571428571},
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.07142857142857142},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.017857142857142856},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.017857142857142856,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.017857142857142856,
                },
            },
            "Location": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "offer": {
        "metaphor": {
            "Theme": {
                wn.synset("state.n.02"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                    "score": 0.6666666666666666,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.3333333333333333,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.5,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.14285714285714285,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 0.14285714285714285,
                },
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5, "score": 1.0},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.09090909090909091},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.18181818181818182},
                wn.synset("abstraction.n.06"): {
                    "frequency": 6,
                    "conf": 0.5454545454545454,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                },
                wn.synset("abstraction.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                },
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 6, "conf": 1.0}},
        },
    },
    "promise": {
        "metaphor": {
            "Agent": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 5, "conf": 1.0}},
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5},
            },
        },
    },
    "live": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 28, "conf": 0.7},
                wn.synset("abstraction.n.06"): {"frequency": 6, "conf": 0.15},
                wn.synset("animal.n.01"): {"frequency": 4, "conf": 0.1},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.025},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.025},
            }
        },
    },
    "put": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 11,
                    "conf": 0.34375,
                    "score": 0.4230769230769231,
                },
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.03125, "score": 1.0},
                wn.synset("abstraction.n.06"): {
                    "frequency": 10,
                    "conf": 0.3125,
                    "score": 0.8333333333333334,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.03125,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.0625, "score": 0.4},
                wn.synset("state.n.02"): {
                    "frequency": 2,
                    "conf": 0.0625,
                    "score": 0.6666666666666666,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.0625,
                    "score": 1.0,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.03125, "score": 1.0},
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.03125,
                    "score": 0.5,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.03125,
                    "score": 0.125,
                },
            },
            "Destination": {
                wn.synset("abstraction.n.01"): {
                    "frequency": 2,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.3333333333333333,
                    "score": 0.3333333333333333,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 6,
                    "conf": 0.5,
                    "score": 0.6666666666666666,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 12,
                    "conf": 0.8,
                    "score": 0.5217391304347826,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                    "score": 0.5,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 8, "conf": 0.4},
                wn.synset("substance.n.01"): {"frequency": 3, "conf": 0.15},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.05},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.05},
                wn.synset("location.n.01"): {"frequency": 4, "conf": 0.2},
                wn.synset("body_part.n.01"): {"frequency": 2, "conf": 0.1},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.05},
            },
            "Theme": {
                wn.synset("body_part.n.01"): {"frequency": 7, "conf": 0.21212121212121213},
                wn.synset("person.n.01"): {"frequency": 15, "conf": 0.45454545454545453},
                wn.synset("clothing.n.01"): {"frequency": 3, "conf": 0.09090909090909091},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.06060606060606061,
                },
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.09090909090909091},
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.030303030303030304,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.030303030303030304},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.030303030303030304,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 11, "conf": 0.8461538461538461},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.07692307692307693},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                },
            },
        },
    },
    "require": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 29,
                    "conf": 0.6744186046511628,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.06976744186046512,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.023255813953488372},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.023255813953488372,
                },
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.06976744186046512},
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.046511627906976744},
                wn.synset("food.n.02"): {"frequency": 1, "conf": 0.023255813953488372},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.023255813953488372,
                },
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.023255813953488372},
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.023255813953488372,
                },
            }
        },
    },
    "break": {
        "metaphor": {
            "Patient": {
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.4166666666666667,
                    "score": 1.0,
                },
                wn.synset("machine.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4, "score": 1.0},
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.2, "score": 1.0},
                wn.synset("sound.n.01"): {"frequency": 1, "conf": 0.2, "score": 1.0},
            },
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
        },
        "literal": {
            "Patient": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "pass": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.25, "score": 0.6},
                wn.synset("time.n.05"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
            },
            "Recipient": {
                wn.synset("communication.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("clothing.n.01"): {"frequency": 2, "conf": 0.5},
            },
            "Agent": {
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "encounter": {
        "metaphor": {},
        "literal": {"Stimulus": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "give": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 52,
                    "conf": 0.5048543689320388,
                    "score": 0.6046511627906976,
                },
                wn.synset("state.n.02"): {
                    "frequency": 15,
                    "conf": 0.14563106796116504,
                    "score": 0.8823529411764706,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 10,
                    "conf": 0.0970873786407767,
                    "score": 0.7142857142857143,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.009708737864077669,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 3,
                    "conf": 0.02912621359223301,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 3,
                    "conf": 0.02912621359223301,
                    "score": 0.75,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 4,
                    "conf": 0.038834951456310676,
                    "score": 1.0,
                },
                wn.synset("food.n.01"): {
                    "frequency": 1,
                    "conf": 0.009708737864077669,
                    "score": 0.5,
                },
                wn.synset("region.n.03"): {
                    "frequency": 2,
                    "conf": 0.019417475728155338,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.01"): {
                    "frequency": 1,
                    "conf": 0.009708737864077669,
                    "score": 1.0,
                },
                wn.synset("sound.n.04"): {
                    "frequency": 1,
                    "conf": 0.009708737864077669,
                    "score": 1.0,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 4,
                    "conf": 0.038834951456310676,
                    "score": 1.0,
                },
                wn.synset("location.n.01"): {
                    "frequency": 1,
                    "conf": 0.009708737864077669,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.009708737864077669,
                    "score": 1.0,
                },
                wn.synset("region.n.01"): {
                    "frequency": 1,
                    "conf": 0.009708737864077669,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.009708737864077669,
                    "score": 0.14285714285714285,
                },
                wn.synset("currency.n.01"): {
                    "frequency": 1,
                    "conf": 0.009708737864077669,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.009708737864077669,
                    "score": 0.14285714285714285,
                },
            },
            "Recipient": {
                wn.synset("person.n.01"): {
                    "frequency": 42,
                    "conf": 0.7636363636363637,
                    "score": 0.6363636363636364,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.01818181818181818,
                    "score": 0.5,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.01818181818181818,
                    "score": 0.5,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.01818181818181818,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.12727272727272726,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.01818181818181818,
                    "score": 0.5,
                },
                wn.synset("region.n.03"): {
                    "frequency": 2,
                    "conf": 0.03636363636363636,
                    "score": 0.6666666666666666,
                },
            },
            "Agent": {
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.044444444444444446,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 22,
                    "conf": 0.4888888888888889,
                    "score": 0.5945945945945946,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 11,
                    "conf": 0.24444444444444444,
                    "score": 0.9166666666666666,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.044444444444444446,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 2,
                    "conf": 0.044444444444444446,
                    "score": 1.0,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.022222222222222223,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.06666666666666667,
                    "score": 1.0,
                },
                wn.synset("sound.n.01"): {
                    "frequency": 1,
                    "conf": 0.022222222222222223,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.022222222222222223,
                    "score": 1.0,
                },
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Recipient": {
                wn.synset("person.n.01"): {"frequency": 24, "conf": 0.8},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.03333333333333333},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.03333333333333333},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.03333333333333333,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.03333333333333333,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.03333333333333333},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.03333333333333333},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 34,
                    "conf": 0.6181818181818182,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.01818181818181818,
                },
                wn.synset("artifact.n.01"): {"frequency": 6, "conf": 0.10909090909090909},
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.03636363636363636},
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.10909090909090909},
                wn.synset("communication.n.02"): {
                    "frequency": 4,
                    "conf": 0.07272727272727272,
                },
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.01818181818181818},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.01818181818181818},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 15, "conf": 0.9375},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.0625},
            },
        },
    },
    "gain": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.75,
                    "score": 0.6,
                },
            },
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.6666666666666666,
                }
            },
        },
        "literal": {
            "Theme": {
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "ignore": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.6},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.375},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("communication.n.02"): {"frequency": 2, "conf": 0.25},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.125},
            },
        },
    },
    "walk": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.045454545454545456,
                }
            }
        },
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 21, "conf": 1.0}},
            "Destination": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "alert": {
        "metaphor": {},
        "literal": {
            "Recipient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "owe": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 0.5}
            },
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "assume": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 7, "conf": 0.875, "score": 1.0},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("state.n.01"): {"frequency": 3, "conf": 0.25, "score": 1.0},
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 6, "conf": 0.5, "score": 1.0},
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {},
    },
    "heat": {"metaphor": {}, "literal": {}},
    "leave": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 10,
                    "conf": 0.5263157894736842,
                    "score": 0.25,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 6,
                    "conf": 0.3157894736842105,
                    "score": 0.75,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.05263157894736842,
                    "score": 1.0,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.05263157894736842,
                    "score": 1.0,
                },
                wn.synset("sound.n.01"): {
                    "frequency": 1,
                    "conf": 0.05263157894736842,
                    "score": 1.0,
                },
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 30, "conf": 0.8823529411764706},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.058823529411764705,
                },
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.058823529411764705},
            },
            "Recipient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Patient": {wn.synset("artifact.n.01"): {"frequency": 2, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "divert": {"metaphor": {}, "literal": {}},
    "kick": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Theme": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "introduce": {
        "metaphor": {
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.1, "score": 1.0},
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.3, "score": 0.6},
                wn.synset("idea.n.01"): {"frequency": 3, "conf": 0.3, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.1,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.2,
                    "score": 0.6666666666666666,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5, "score": 1.0},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5, "score": 1.0},
            },
        },
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4},
                wn.synset("organization.n.01"): {"frequency": 2, "conf": 0.4},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2},
            }
        },
    },
    "spread": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.5, "score": 1.0},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}}},
    },
    "model": {"metaphor": {}, "literal": {}},
    "regard": {
        "metaphor": {
            "Theme": {
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 0.5,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 8,
                    "conf": 0.6153846153846154,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.15384615384615385,
                    "score": 1.0,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 0.5,
                },
            },
            "Attribute": {
                wn.synset("substance.n.01"): {
                    "frequency": 17,
                    "conf": 0.9444444444444444,
                    "score": 0.8947368421052632,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 5, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Attribute": {wn.synset("substance.n.01"): {"frequency": 2, "conf": 1.0}},
            "Theme": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
        },
    },
    "function": {
        "metaphor": {},
        "literal": {
            "Attribute": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Agent": {
                wn.synset("body_part.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
            },
        },
    },
    "build": {
        "metaphor": {
            "Product": {
                wn.synset("region.n.03"): {"frequency": 3, "conf": 0.375, "score": 1.0},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 0.08333333333333333,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.125, "score": 1.0},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.125, "score": 0.5},
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.13333333333333333,
                }
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 13, "conf": 1.0}},
            "Product": {
                wn.synset("artifact.n.01"): {"frequency": 11, "conf": 0.6470588235294118},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.11764705882352941,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.058823529411764705},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.058823529411764705},
                wn.synset("plant.n.01"): {"frequency": 2, "conf": 0.11764705882352941},
            },
        },
    },
    "activate": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 0.5},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            }
        },
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "reject": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.5714285714285714,
                },
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.2857142857142857},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("organization.n.01"): {"frequency": 2, "conf": 0.5},
            },
        },
    },
    "wallow": {"metaphor": {}, "literal": {}},
    "call": {
        "metaphor": {
            "Agent": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.6,
                    "score": 0.14285714285714285,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.2, "score": 1.0},
            },
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.42857142857142855,
                    "score": 0.14285714285714285,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                    "score": 0.4,
                },
                wn.synset("vehicle.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.045454545454545456,
                },
                wn.synset("person.n.01"): {"frequency": 18, "conf": 0.8181818181818182},
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.13636363636363635,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.05},
                wn.synset("person.n.01"): {"frequency": 18, "conf": 0.9},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.05},
            },
            "Recipient": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "prefer": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Theme": {
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.25},
            }
        },
    },
    "collect": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.16666666666666666,
                },
                wn.synset("clothing.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.09090909090909091,
                },
            },
            "Agent": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 10, "conf": 0.5},
                wn.synset("food.n.01"): {"frequency": 2, "conf": 0.1},
                wn.synset("person.n.01"): {"frequency": 5, "conf": 0.25},
                wn.synset("communication.n.02"): {"frequency": 2, "conf": 0.1},
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.05},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.8},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.2},
            },
        },
    },
    "stop": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5, "score": 0.5},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("sound.n.04"): {"frequency": 1, "conf": 0.25, "score": 1.0},
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.15384615384615385,
                }
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 11, "conf": 0.9166666666666666},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                },
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.25},
            },
        },
    },
    "eat": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.2}
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.05555555555555555,
                }
            },
        },
        "literal": {
            "Patient": {
                wn.synset("food.n.02"): {"frequency": 5, "conf": 0.20833333333333334},
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.08333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.16666666666666666,
                },
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.16666666666666666},
                wn.synset("food.n.01"): {"frequency": 8, "conf": 0.3333333333333333},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.041666666666666664},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 17, "conf": 0.8947368421052632},
                wn.synset("region.n.03"): {"frequency": 2, "conf": 0.10526315789473684},
            },
        },
    },
    "fall": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Patient": {
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.4,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.2, "score": 0.5},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.2, "score": 0.5},
            },
            "Attribute": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
        },
        "literal": {
            "Location": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("physical_entity.n.01"): {"frequency": 2, "conf": 1.0}},
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "term": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "organise": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "cover": {
        "metaphor": {
            "Destination": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.45454545454545453,
                    "score": 0.8333333333333334,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.18181818181818182,
                    "score": 0.6666666666666666,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 0.5,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
        },
        "literal": {
            "Destination": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.14285714285714285},
            },
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "submerge": {"metaphor": {}, "literal": {}},
    "mind": {
        "metaphor": {},
        "literal": {
            "Stimulus": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.4444444444444444},
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.3333333333333333,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.1111111111111111},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.1111111111111111},
            }
        },
    },
    "use": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.2222222222222222,
                    "score": 0.16666666666666666,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 0.0625,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.4444444444444444,
                    "score": 0.10810810810810811,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 0.05263157894736842,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 4,
                    "conf": 0.4,
                    "score": 0.12903225806451613,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.2,
                    "score": 0.5,
                },
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.1, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.1,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.1, "score": 1.0},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.1, "score": 0.2},
            },
            "Predicate": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.5,
                }
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 27, "conf": 0.7714285714285715},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.05714285714285714,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.02857142857142857},
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.11428571428571428,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.02857142857142857},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 33,
                    "conf": 0.3473684210526316,
                },
                wn.synset("artifact.n.01"): {"frequency": 18, "conf": 0.18947368421052632},
                wn.synset("person.n.01"): {"frequency": 10, "conf": 0.10526315789473684},
                wn.synset("communication.n.02"): {
                    "frequency": 15,
                    "conf": 0.15789473684210525,
                },
                wn.synset("state.n.01"): {"frequency": 2, "conf": 0.021052631578947368},
                wn.synset("idea.n.01"): {"frequency": 4, "conf": 0.042105263157894736},
                wn.synset("organization.n.01"): {
                    "frequency": 5,
                    "conf": 0.05263157894736842,
                },
                wn.synset("plant.n.01"): {"frequency": 1, "conf": 0.010526315789473684},
                wn.synset("machine.n.01"): {"frequency": 1, "conf": 0.010526315789473684},
                wn.synset("sound.n.01"): {"frequency": 2, "conf": 0.021052631578947368},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.010526315789473684},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.010526315789473684},
                wn.synset("substance.n.07"): {
                    "frequency": 1,
                    "conf": 0.010526315789473684,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.010526315789473684,
                },
            },
            "Predicate": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 14,
                    "conf": 0.5384615384615384,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.038461538461538464},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                },
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.038461538461538464},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.11538461538461539},
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                },
                wn.synset("idea.n.01"): {"frequency": 4, "conf": 0.15384615384615385},
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                },
            },
        },
    },
    "aim": {
        "metaphor": {
            "Stimulus": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.6666666666666666,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            }
        },
        "literal": {
            "Stimulus": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "evolve": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Product": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "stand": {
        "metaphor": {
            "Theme": {
                wn.synset("location.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                    "score": 0.2,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("plant.n.02"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.18181818181818182,
                },
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 9, "conf": 1.0}},
            "Theme": {wn.synset("person.n.01"): {"frequency": 8, "conf": 1.0}},
        },
    },
    "move": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.2222222222222222,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.3333333333333333,
                    "score": 0.2727272727272727,
                },
                wn.synset("vehicle.n.01"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.2222222222222222,
                    "score": 0.6666666666666666,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 0.5,
                },
            },
            "Agent": {
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.25,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.09090909090909091},
                wn.synset("person.n.01"): {"frequency": 8, "conf": 0.7272727272727273},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                },
            },
            "Patient": {
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Location": {
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Destination": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Predicate": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "ring": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.25},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.2},
            }
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.8},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.2},
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.5},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.375},
            },
        },
    },
    "dance": {"metaphor": {}, "literal": {}},
    "drink": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0}},
            "Patient": {
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "suppose": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 14, "conf": 0.5384615384615384},
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.2692307692307692,
                },
                wn.synset("state.n.02"): {"frequency": 3, "conf": 0.11538461538461539},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.038461538461538464,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 12, "conf": 0.9230769230769231},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                },
            },
            "Attribute": {
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.2},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.1},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.1},
                wn.synset("substance.n.01"): {"frequency": 3, "conf": 0.3},
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.3},
            },
        },
    },
    "evaluate": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.5714285714285714,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
            }
        },
    },
    "publish": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "wonder": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 16, "conf": 1.0}}},
    },
    "reward": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "confer": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "grasp": {
        "metaphor": {
            "Stimulus": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.5,
                }
            }
        },
        "literal": {
            "Stimulus": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "appear": {
        "metaphor": {
            "Attribute": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.3333333333333333,
                },
            }
        },
        "literal": {
            "Attribute": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.3333333333333333},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.16666666666666666},
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
            },
        },
    },
    "write": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.4},
                wn.synset("communication.n.02"): {"frequency": 2, "conf": 0.2},
                wn.synset("state.n.01"): {"frequency": 2, "conf": 0.2},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 19, "conf": 0.95},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.05},
            },
            "Topic": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "amplify": {
        "metaphor": {
            "Patient": {
                wn.synset("communication.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {},
    },
    "play": {
        "metaphor": {
            "Agent": {
                wn.synset("sound.n.01"): {"frequency": 1, "conf": 0.2, "score": 1.0},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 0.25,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.4, "score": 0.25},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 0.3333333333333333,
                },
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 0.1},
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 9, "conf": 0.75},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.08333333333333333},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.08333333333333333},
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 3, "conf": 0.2},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.06666666666666667},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                },
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.4},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.13333333333333333,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.06666666666666667},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.06666666666666667},
            },
        },
    },
    "reply": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 5, "conf": 0.8333333333333334},
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
            "Topic": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.6},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.2},
            },
            "Recipient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "summarize": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("animal.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "start": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.25},
            }
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 13, "conf": 0.8666666666666667},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                },
            },
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.14285714285714285},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                },
                wn.synset("machine.n.01"): {"frequency": 1, "conf": 0.07142857142857142},
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.35714285714285715,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.07142857142857142},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.21428571428571427},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                },
            },
        },
    },
    "decide": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.047619047619047616,
                }
            }
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 20, "conf": 0.9523809523809523},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.047619047619047616,
                },
            }
        },
    },
    "marry": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.8},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.2},
            },
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "press": {
        "metaphor": {
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
            },
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
        },
    },
    "form": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            }
        },
    },
    "bring": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.12, "score": 0.3},
                wn.synset("abstraction.n.06"): {
                    "frequency": 12,
                    "conf": 0.48,
                    "score": 0.8,
                },
                wn.synset("region.n.03"): {"frequency": 3, "conf": 0.12, "score": 1.0},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.04, "score": 1.0},
                wn.synset("artifact.n.01"): {
                    "frequency": 2,
                    "conf": 0.08,
                    "score": 0.3333333333333333,
                },
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.08, "score": 1.0},
                wn.synset("communication.n.01"): {
                    "frequency": 1,
                    "conf": 0.04,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.04,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.3, "score": 1.0},
                wn.synset("person.n.01"): {
                    "frequency": 7,
                    "conf": 0.7,
                    "score": 0.4666666666666667,
                },
            },
            "Destination": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.5714285714285714,
                    "score": 0.8,
                },
                wn.synset("location.n.01"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                    "score": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
            },
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.8, "score": 1.0},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.2, "score": 0.25},
            },
        },
        "literal": {
            "Theme": {
                wn.synset("vehicle.n.01"): {"frequency": 2, "conf": 0.1111111111111111},
                wn.synset("person.n.01"): {"frequency": 7, "conf": 0.3888888888888889},
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.16666666666666666,
                },
                wn.synset("artifact.n.01"): {"frequency": 4, "conf": 0.2222222222222222},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                },
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.05555555555555555},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 8, "conf": 1.0}},
            "Destination": {
                wn.synset("body_part.n.01"): {"frequency": 2, "conf": 0.25},
                wn.synset("location.n.01"): {"frequency": 4, "conf": 0.5},
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.125},
            },
            "Location": {wn.synset("body_part.n.01"): {"frequency": 3, "conf": 1.0}},
        },
    },
    "pledge": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "shout": {
        "metaphor": {},
        "literal": {
            "Recipient": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "cross": {
        "metaphor": {
            "Theme": {wn.synset("food.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}}
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            }
        },
    },
    "return": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 0.07142857142857142,
                },
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5, "score": 1.0},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25, "score": 1.0},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 13, "conf": 1.0}},
            "Destination": {wn.synset("location.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "improve": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 8,
                    "conf": 0.6666666666666666,
                },
                wn.synset("communication.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                },
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.08333333333333333},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.08333333333333333},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.08333333333333333},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
        },
    },
    "emerge": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 1.0,
                    "score": 0.8333333333333334,
                }
            }
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            }
        },
    },
    "contain": {
        "metaphor": {
            "Theme": {
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Location": {
                wn.synset("organization.n.01"): {
                    "frequency": 4,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "snarl": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "expect": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            }
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "undertake": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 1.0}},
            "Instrument": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "settle": {
        "metaphor": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.5, "score": 1.0},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
            },
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "pull": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 4, "conf": 0.36363636363636365},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.18181818181818182,
                },
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.09090909090909091},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                },
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.2727272727272727},
            },
            "Patient": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
            "Location": {wn.synset("substance.n.01"): {"frequency": 2, "conf": 1.0}},
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.8571428571428571},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
            },
            "Destination": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "indicate": {
        "metaphor": {
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Topic": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.125},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.25},
                wn.synset("state.n.01"): {"frequency": 3, "conf": 0.375},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.25},
            }
        },
    },
    "read": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.18181818181818182,
                }
            }
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 9, "conf": 0.9},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.1},
            },
            "Theme": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Topic": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "produce": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.6,
                    "score": 0.75,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2, "score": 0.5},
            },
        },
        "literal": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("organization.n.01"): {"frequency": 2, "conf": 0.5},
            },
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "upset": {"metaphor": {}, "literal": {}},
    "serve": {
        "metaphor": {
            "Attribute": {
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 2,
                    "conf": 0.18181818181818182,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.45454545454545453,
                    "score": 0.8333333333333334,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.18181818181818182,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.5714285714285714,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.42857142857142855,
                    "score": 0.6,
                },
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Recipient": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Attribute": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("food.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "label": {
        "metaphor": {
            "Theme": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5, "score": 1.0},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
            }
        },
        "literal": {},
    },
    "create": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
            },
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 1.0}},
        },
    },
    "stay": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 1.0,
                    "score": 0.17647058823529413,
                }
            }
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 14, "conf": 0.9333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                },
            },
            "Location": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.42857142857142855,
                },
                wn.synset("substance.n.01"): {"frequency": 4, "conf": 0.5714285714285714},
            },
        },
    },
    "help": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.42857142857142855,
                    "score": 0.6,
                },
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.42857142857142855,
                    "score": 0.2727272727272727,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            },
            "Beneficiary": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.16666666666666666,
                },
                wn.synset("region.n.03"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.23076923076923078},
                wn.synset("abstraction.n.06"): {
                    "frequency": 14,
                    "conf": 0.5384615384615384,
                },
                wn.synset("plant.n.01"): {"frequency": 3, "conf": 0.11538461538461539},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.07692307692307693},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.038461538461538464},
            },
            "Agent": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.07692307692307693},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.15384615384615385,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.15384615384615385,
                },
                wn.synset("person.n.01"): {"frequency": 8, "conf": 0.6153846153846154},
            },
            "Beneficiary": {wn.synset("person.n.01"): {"frequency": 5, "conf": 1.0}},
        },
    },
    "keep": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 9,
                    "conf": 0.375,
                    "score": 0.6428571428571429,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 8,
                    "conf": 0.3333333333333333,
                    "score": 0.6153846153846154,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 3,
                    "conf": 0.125,
                    "score": 0.75,
                },
                wn.synset("sound.n.01"): {
                    "frequency": 1,
                    "conf": 0.041666666666666664,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.041666666666666664,
                    "score": 0.3333333333333333,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 2,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 8,
                    "conf": 0.8,
                    "score": 0.7272727272727273,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.2,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.058823529411764705},
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.29411764705882354,
                },
                wn.synset("person.n.01"): {"frequency": 5, "conf": 0.29411764705882354},
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.11764705882352941,
                },
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.11764705882352941},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.058823529411764705},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.75},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
            },
        },
    },
    "like": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 3, "conf": 0.375},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.25},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.125},
            }
        },
    },
    "anticipate": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 7, "conf": 0.875},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.125},
            },
        },
    },
    "run": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.21428571428571427,
                    "score": 0.5,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.21428571428571427,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 2,
                    "conf": 0.14285714285714285,
                    "score": 0.5,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.21428571428571427,
                    "score": 0.75,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
            },
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
        },
        "literal": {
            "Destination": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.5},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.3333333333333333},
            },
        },
    },
    "strike": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
            "Instrument": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
            "Location": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
        },
        "literal": {
            "Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "stretch": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "begin": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 0.2},
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 2, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.18181818181818182,
                },
                wn.synset("person.n.01"): {"frequency": 7, "conf": 0.6363636363636364},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.18181818181818182,
                },
            },
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.1111111111111111},
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.4444444444444444,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.2222222222222222},
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                },
            },
        },
    },
    "convince": {
        "metaphor": {},
        "literal": {
            "Predicate": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "declare": {
        "metaphor": {
            "Topic": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Topic": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "incorporate": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("abstraction.n.06"): {
                    "frequency": 6,
                    "conf": 0.8571428571428571,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "exhibit": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "accord": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "weep": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "face": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.45454545454545453,
                    "score": 0.7142857142857143,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.18181818181818182,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 0.3333333333333333,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.5,
                    "score": 0.6666666666666666,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "attach": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Patient": {wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "survey": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "confuse": {
        "metaphor": {
            "Agent": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}}
        },
        "literal": {},
    },
    "plummet": {"metaphor": {}, "literal": {}},
    "allow": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 9,
                    "conf": 0.6428571428571429,
                    "score": 0.6428571428571429,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
                wn.synset("time.n.01"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.01"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.5714285714285714,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.25,
                },
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.4166666666666667,
                },
                wn.synset("person.n.01"): {"frequency": 5, "conf": 0.4166666666666667},
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.16666666666666666,
                },
            },
            "Location": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "observe": {
        "metaphor": {
            "Stimulus": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
            "Topic": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Stimulus": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.4},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.2},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Topic": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "examine": {
        "metaphor": {
            "Location": {
                wn.synset("state.n.02"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 7, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 4, "conf": 0.2857142857142857},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.07142857142857142},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.07142857142857142},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.07142857142857142},
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
            },
        },
    },
    "handle": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.5,
                },
                wn.synset("person.n.01"): {
                    "frequency": 5,
                    "conf": 0.8333333333333334,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.2,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.5714285714285714,
                },
            },
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "destroy": {
        "metaphor": {
            "Patient": {
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
            }
        },
        "literal": {},
    },
    "combine": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Patient": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.25},
            }
        },
    },
    "desire": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            }
        },
    },
    "behave": {
        "metaphor": {},
        "literal": {
            "Attribute": {
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "lose": {
        "metaphor": {
            "Agent": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 12,
                    "conf": 0.8571428571428571,
                    "score": 0.9230769230769231,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 10,
                    "conf": 0.45454545454545453,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.045454545454545456,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 2,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 7,
                    "conf": 0.3181818181818182,
                    "score": 0.875,
                },
                wn.synset("state.n.02"): {
                    "frequency": 2,
                    "conf": 0.09090909090909091,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "support": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "cost": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.3},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.1},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.2},
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.2},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.1},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.1},
            }
        },
    },
    "remind": {
        "metaphor": {
            "Recipient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.1}
            }
        },
        "literal": {
            "Recipient": {wn.synset("person.n.01"): {"frequency": 9, "conf": 1.0}},
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.6},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.2},
            },
            "Topic": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("state.n.01"): {"frequency": 3, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
        },
    },
    "associate": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
            "Patient": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "secure": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("artifact.n.01"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.6666666666666666,
                    "score": 0.8,
                },
            },
        },
        "literal": {
            "Asset": {wn.synset("state.n.01"): {"frequency": 2, "conf": 1.0}},
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
            },
        },
    },
    "sell": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.09090909090909091,
                }
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.125,
                }
            },
        },
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 4, "conf": 0.16666666666666666},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.08333333333333333,
                },
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.041666666666666664},
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.2916666666666667,
                },
                wn.synset("person.n.01"): {"frequency": 5, "conf": 0.20833333333333334},
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.041666666666666664},
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.041666666666666664},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.041666666666666664},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.041666666666666664,
                },
                wn.synset("plant.n.01"): {"frequency": 1, "conf": 0.041666666666666664},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 10, "conf": 0.9090909090909091},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                },
            },
            "Recipient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
            },
        },
    },
    "dead": {"metaphor": {}, "literal": {}},
    "consist": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "agree": {
        "metaphor": {},
        "literal": {
            "Topic": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 17, "conf": 0.7391304347826086},
                wn.synset("organization.n.01"): {
                    "frequency": 5,
                    "conf": 0.21739130434782608,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.043478260869565216},
            },
        },
    },
    "pay": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.07142857142857142,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.4,
                    "score": 0.2222222222222222,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.2, "score": 0.5},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
            },
            "Recipient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 0.5},
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("natural_phenomenon.n.01"): {"frequency": 1, "conf": 0.1},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.1},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.1},
                wn.synset("abstraction.n.06"): {"frequency": 7, "conf": 0.7},
            },
            "Asset": {wn.synset("abstraction.n.06"): {"frequency": 5, "conf": 1.0}},
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 13, "conf": 0.9285714285714286},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                },
            },
            "Recipient": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
        },
    },
    "propose": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.2}
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.8},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.2},
            },
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.125},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.125},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.125},
            },
        },
    },
    "suggest": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.25,
                },
                wn.synset("clothing.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.09090909090909091,
                },
                wn.synset("time.n.05"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
            },
            "Topic": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.5714285714285714,
                    "score": 0.36363636363636365,
                },
                wn.synset("location.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("state.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.5,
                },
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 10, "conf": 0.6666666666666666},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.13333333333333333,
                },
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.2},
            },
            "Topic": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.5833333333333334,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.08333333333333333},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.08333333333333333},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.08333333333333333},
            },
        },
    },
    "cancel": {
        "metaphor": {
            "Agent": {
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
            "Theme": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "talk": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 29, "conf": 0.9666666666666667},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.03333333333333333,
                },
            },
            "Topic": {
                wn.synset("substance.n.07"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
            },
        },
    },
    "view": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
            "Attribute": {
                wn.synset("substance.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "assist": {
        "metaphor": {
            "Agent": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Beneficiary": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.6666666666666666,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Beneficiary": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
            "Theme": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "determine": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.1,
                },
            }
        },
        "literal": {
            "Agent": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.6},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 9, "conf": 0.45},
                wn.synset("region.n.03"): {"frequency": 3, "conf": 0.15},
                wn.synset("communication.n.02"): {"frequency": 2, "conf": 0.1},
                wn.synset("state.n.01"): {"frequency": 2, "conf": 0.1},
                wn.synset("time.n.01"): {"frequency": 1, "conf": 0.05},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.05},
                wn.synset("living_thing.n.01"): {"frequency": 1, "conf": 0.05},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.05},
            },
        },
    },
    "wait": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.11764705882352941,
                }
            }
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 15, "conf": 0.8333333333333334},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.05555555555555555},
            }
        },
    },
    "cope": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            }
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Theme": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "change": {
        "metaphor": {
            "Agent": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 5,
                    "conf": 0.7142857142857143,
                    "score": 0.7142857142857143,
                },
                wn.synset("animal.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
            },
            "Patient": {
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.2,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.3, "score": 0.75},
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.4,
                    "score": 0.5714285714285714,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.1,
                    "score": 0.5,
                },
            },
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.6, "score": 1.0},
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 0.3333333333333333,
                },
            },
        },
        "literal": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
            },
            "Patient": {
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.5},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.4},
                wn.synset("clothing.n.01"): {"frequency": 3, "conf": 0.6},
            },
        },
    },
    "try": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.5,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                    "score": 0.6666666666666666,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.058823529411764705,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("region.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.25, "score": 1.0},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.125, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.25,
                    "score": 0.4,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.125, "score": 0.25},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.25,
                    "score": 0.10526315789473684,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 17,
                    "conf": 0.6071428571428571,
                },
                wn.synset("region.n.03"): {"frequency": 2, "conf": 0.07142857142857142},
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.10714285714285714,
                },
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.03571428571428571},
                wn.synset("sound.n.01"): {"frequency": 1, "conf": 0.03571428571428571},
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.10714285714285714},
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.03571428571428571},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 16, "conf": 0.8888888888888888},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.05555555555555555,
                },
            },
        },
    },
    "prosecute": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "sob": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "head": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "phone": {
        "metaphor": {
            "Recipient": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.14285714285714285,
                }
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.8571428571428571},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
            },
            "Recipient": {wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0}},
        },
    },
    "meet": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.2,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.4,
                },
            }
        },
        "literal": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.375},
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.5},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.125},
            }
        },
    },
    "contrast": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
            }
        },
    },
    "continue": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.1}
            }
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 9, "conf": 0.6},
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.13333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.13333333333333333},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.06666666666666667},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.06666666666666667},
            }
        },
    },
    "quote": {
        "metaphor": {
            "Topic": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Topic": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 6, "conf": 1.0}},
        },
    },
    "provide": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.5,
                    "score": 0.0625,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 0.125,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.25, "score": 0.5},
            },
            "Agent": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.1111111111111111,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.14285714285714285,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 30,
                    "conf": 0.5454545454545454,
                },
                wn.synset("artifact.n.01"): {"frequency": 8, "conf": 0.14545454545454545},
                wn.synset("communication.n.02"): {
                    "frequency": 7,
                    "conf": 0.12727272727272726,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.01818181818181818},
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.03636363636363636},
                wn.synset("state.n.02"): {"frequency": 3, "conf": 0.05454545454545454},
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.01818181818181818},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.01818181818181818},
                wn.synset("tool.n.01"): {"frequency": 1, "conf": 0.01818181818181818},
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.01818181818181818,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.35294117647058826},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.11764705882352941,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.058823529411764705},
                wn.synset("abstraction.n.06"): {
                    "frequency": 8,
                    "conf": 0.47058823529411764,
                },
            },
            "Recipient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("organization.n.01"): {"frequency": 3, "conf": 0.6},
            },
        },
    },
    "attend": {
        "metaphor": {
            "Theme": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.6666666666666666,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            }
        },
    },
    "sweep": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "swing": {
        "metaphor": {
            "Theme": {
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "work": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.375, "score": 0.75},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.25,
                    "score": 0.3333333333333333,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.125, "score": 1.0},
                wn.synset("body_part.n.01"): {"frequency": 2, "conf": 0.25, "score": 1.0},
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.08695652173913043,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 21, "conf": 0.9130434782608695},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.08695652173913043,
                },
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.6666666666666666,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
        },
    },
    "star": {
        "metaphor": {
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
        },
    },
    "last": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "invert": {"metaphor": {}, "literal": {}},
    "bang": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "estimate": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
        },
    },
    "love": {"metaphor": {}, "literal": {}},
    "achieve": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "please": {"metaphor": {}, "literal": {}},
    "release": {
        "metaphor": {
            "Theme": {
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("time.n.05"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "pose": {
        "metaphor": {},
        "literal": {
            "Topic": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "radio": {
        "metaphor": {},
        "literal": {
            "Topic": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "interest": {"metaphor": {}, "literal": {}},
    "light": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "ask": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 0.0625}
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 0.2},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
            "Topic": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.1111111111111111,
                }
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 30, "conf": 0.8823529411764706},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.058823529411764705,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.058823529411764705,
                },
            },
            "Recipient": {
                wn.synset("person.n.01"): {"frequency": 17, "conf": 0.85},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.05},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.05},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.05},
            },
            "Topic": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 8,
                    "conf": 0.5333333333333333,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.06666666666666667},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.06666666666666667},
                wn.synset("state.n.01"): {"frequency": 2, "conf": 0.13333333333333333},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.13333333333333333},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.06666666666666667},
            },
            "Patient": {
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.5714285714285714,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
            },
        },
    },
    "stuff": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Destination": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "resign": {"metaphor": {}, "literal": {}},
    "add": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 10,
                    "conf": 0.5882352941176471,
                    "score": 0.8333333333333334,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.4117647058823529,
                    "score": 1.0,
                },
            },
            "Topic": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 11,
                    "conf": 0.8461538461538461,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 9,
                    "conf": 0.75,
                    "score": 0.8181818181818182,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 0.5,
                },
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.4},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2},
            },
        },
    },
    "wear": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.14285714285714285,
                }
            },
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
        },
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.14285714285714285},
                wn.synset("clothing.n.01"): {"frequency": 7, "conf": 0.5},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.14285714285714285,
                },
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.21428571428571427},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.6666666666666666},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                },
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.1111111111111111},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                },
            },
        },
    },
    "happen": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 5, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.4},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.1},
            }
        },
    },
    "pick": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.25},
            },
        },
    },
    "hamper": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "present": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.4444444444444444,
                    "score": 0.5714285714285714,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 0.5,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.75,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.6},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.2},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.2},
            },
            "Agent": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "point": {
        "metaphor": {
            "Agent": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 0.5,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25, "score": 0.25},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
            }
        },
        "literal": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.75},
            },
            "Recipient": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
        },
    },
    "wish": {"metaphor": {}, "literal": {}},
    "slice": {
        "metaphor": {
            "Patient": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "display": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("food.n.02"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {
            "Theme": {
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "fit": {"metaphor": {}, "literal": {}},
    "haul": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("state.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "borrow": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "employ": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.5, "score": 1.0},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5, "score": 1.0},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Predicate": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "wet": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "retrieve": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "recall": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.8, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 6, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "solidify": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "choke": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "thank": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 41, "conf": 0.9318181818181818},
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.06818181818181818,
                },
            },
            "Attribute": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
        },
    },
    "excuse": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "develop": {
        "metaphor": {},
        "literal": {
            "Product": {
                wn.synset("communication.n.02"): {
                    "frequency": 4,
                    "conf": 0.15384615384615385,
                },
                wn.synset("idea.n.01"): {"frequency": 4, "conf": 0.15384615384615385},
                wn.synset("abstraction.n.06"): {
                    "frequency": 12,
                    "conf": 0.46153846153846156,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.11538461538461539,
                },
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.07692307692307693},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.038461538461538464},
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5},
            },
        },
    },
    "yield": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "note": {
        "metaphor": {
            "Stimulus": {
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("state.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.5,
                },
                wn.synset("location.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
            }
        },
        "literal": {
            "Stimulus": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            }
        },
    },
    "lift": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {
            "Theme": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Destination": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "outline": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "sign": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.25,
                }
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 5, "conf": 1.0}},
            "Theme": {
                wn.synset("communication.n.02"): {"frequency": 3, "conf": 0.6},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2},
            },
        },
    },
    "highlight": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.8, "score": 1.0},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.2, "score": 0.5},
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.6666666666666666,
                },
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.5},
            },
            "Patient": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.3333333333333333},
            },
        },
    },
    "predict": {
        "metaphor": {
            "Topic": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}}
        },
        "literal": {
            "Topic": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.7142857142857143,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
            }
        },
    },
    "sponsor": {
        "metaphor": {},
        "literal": {
            "Beneficiary": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.6},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.2},
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "express": {
        "metaphor": {
            "Theme": {
                wn.synset("state.n.02"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.6666666666666666,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "breathe": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("animal.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "distrust": {"metaphor": {}, "literal": {}},
    "drop": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
            "Attribute": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Attribute": {wn.synset("substance.n.01"): {"frequency": 2, "conf": 1.0}},
            "Theme": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Destination": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "teach": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 6, "conf": 1.0}},
            "Recipient": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.6666666666666666},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
            "Topic": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
        },
    },
    "fumble": {
        "metaphor": {
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "whistle": {
        "metaphor": {
            "Agent": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "open": {
        "metaphor": {
            "Patient": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 0.3333333333333333,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 0.125,
                },
                wn.synset("communication.n.01"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.125, "score": 1.0},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.125, "score": 1.0},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.1}
            },
        },
        "literal": {
            "Patient": {
                wn.synset("body_part.n.01"): {"frequency": 2, "conf": 0.18181818181818182},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.18181818181818182},
                wn.synset("artifact.n.01"): {"frequency": 7, "conf": 0.6363636363636364},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 9, "conf": 1.0}},
        },
    },
    "assess": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.2857142857142857,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
            }
        },
        "literal": {
            "Theme": {
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.1},
                wn.synset("communication.n.02"): {"frequency": 2, "conf": 0.2},
                wn.synset("abstraction.n.06"): {"frequency": 5, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.1},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.1},
            }
        },
    },
    "visit": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}}},
    },
    "infer": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}},
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "reduce": {
        "metaphor": {
            "Patient": {
                wn.synset("communication.n.01"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 3,
                    "conf": 0.13043478260869565,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 13,
                    "conf": 0.5652173913043478,
                    "score": 0.8125,
                },
                wn.synset("state.n.02"): {
                    "frequency": 3,
                    "conf": 0.13043478260869565,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.01"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                    "score": 0.5,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.043478260869565216,
                    "score": 1.0,
                },
            }
        },
        "literal": {
            "Patient": {
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.6},
                wn.synset("natural_phenomenon.n.01"): {"frequency": 1, "conf": 0.2},
            }
        },
    },
    "characterize": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Attribute": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "save": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 1.0, "score": 0.6}
            },
            "Beneficiary": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.2},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.2},
            }
        },
    },
    "dictate": {
        "metaphor": {
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Topic": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "nod": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 5, "conf": 0.8333333333333334},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.16666666666666666},
            },
            "Theme": {wn.synset("communication.n.02"): {"frequency": 3, "conf": 1.0}},
        },
    },
    "define": {
        "metaphor": {
            "Attribute": {
                wn.synset("substance.n.01"): {"frequency": 3, "conf": 1.0, "score": 0.5}
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
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                },
            },
            "Attribute": {
                wn.synset("substance.n.01"): {"frequency": 3, "conf": 0.6},
                wn.synset("communication.n.02"): {"frequency": 2, "conf": 0.4},
            },
        },
    },
    "avoid": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.6666666666666666,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
            }
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "learn": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            }
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 11, "conf": 0.7857142857142857},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.14285714285714285,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                },
            },
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.2857142857142857},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.42857142857142855},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
            },
            "Topic": {
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.2222222222222222},
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.5555555555555556,
                },
                wn.synset("animal.n.01"): {"frequency": 2, "conf": 0.2222222222222222},
            },
        },
    },
    "demonstrate": {
        "metaphor": {
            "Topic": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 0.2,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 0.5,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25, "score": 0.5},
            }
        },
        "literal": {
            "Topic": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.6666666666666666,
                },
            }
        },
    },
    "enjoy": {"metaphor": {}, "literal": {}},
    "argue": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            }
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.26666666666666666,
                },
                wn.synset("state.n.01"): {"frequency": 5, "conf": 0.3333333333333333},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.06666666666666667},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.2},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.06666666666666667},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.06666666666666667},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 13, "conf": 0.8125},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.0625},
                wn.synset("organization.n.01"): {"frequency": 2, "conf": 0.125},
            },
        },
    },
    "frighten": {"metaphor": {}, "literal": {}},
    "subdue": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "generate": {
        "metaphor": {
            "Patient": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {
            "Patient": {wn.synset("communication.n.02"): {"frequency": 2, "conf": 1.0}}
        },
    },
    "scrub": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "dawdle": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Location": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "name": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            }
        },
        "literal": {
            "Theme": {
                wn.synset("substance.n.07"): {"frequency": 1, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.4},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.2},
            }
        },
    },
    "smile": {
        "metaphor": {
            "Recipient": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.2}
            }
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 8, "conf": 1.0}},
            "Recipient": {wn.synset("substance.n.01"): {"frequency": 4, "conf": 1.0}},
        },
    },
    "locate": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "dominate": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {"Theme": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}}},
    },
    "hang": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.3333333333333333,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            }
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.25},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.25},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.125},
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.125},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "pop": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {"Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "arrive": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.14285714285714285,
                },
            }
        },
        "literal": {
            "Destination": {
                wn.synset("substance.n.01"): {"frequency": 3, "conf": 0.5},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
            "Theme": {wn.synset("person.n.01"): {"frequency": 6, "conf": 1.0}},
        },
    },
    "believe": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.25}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.2}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("state.n.01"): {"frequency": 3, "conf": 0.125},
                wn.synset("person.n.01"): {"frequency": 10, "conf": 0.4166666666666667},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.041666666666666664,
                },
                wn.synset("substance.n.01"): {"frequency": 3, "conf": 0.125},
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.16666666666666666,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.041666666666666664,
                },
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.041666666666666664},
                wn.synset("machine.n.01"): {"frequency": 1, "conf": 0.041666666666666664},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 21, "conf": 0.875},
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.125},
            },
        },
    },
    "suffer": {"metaphor": {}, "literal": {}},
    "establish": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.11764705882352941},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                },
                wn.synset("region.n.03"): {"frequency": 4, "conf": 0.23529411764705882},
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.17647058823529413,
                },
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.11764705882352941},
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.058823529411764705,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.058823529411764705},
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.11764705882352941,
                },
            },
            "Agent": {wn.synset("organization.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "link": {
        "metaphor": {
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.25, "score": 1.0},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.125, "score": 1.0},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 0.3333333333333333,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.125, "score": 1.0},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.25, "score": 1.0},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Agent": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
            "Patient": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
            },
        },
    },
    "object": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "amount": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "force": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.5, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Patient": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "obtain": {
        "metaphor": {
            "Theme": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.5,
                    "score": 0.6666666666666666,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
            },
        },
    },
    "consult": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "consider": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 11,
                    "conf": 0.7333333333333333,
                    "score": 0.8461538461538461,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                    "score": 0.3333333333333333,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 2,
                    "conf": 0.13333333333333333,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                    "score": 0.5,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 7,
                    "conf": 0.875,
                    "score": 0.5833333333333334,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 5, "conf": 1.0}},
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.125},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.25},
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.25},
            },
        },
    },
    "grow": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {
            "Product": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Patient": {
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.2857142857142857},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                },
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
            },
        },
    },
    "recommend": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "occur": {
        "metaphor": {
            "Location": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 6,
                    "conf": 0.5454545454545454,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.18181818181818182},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                },
                wn.synset("sound.n.04"): {"frequency": 1, "conf": 0.09090909090909091},
            },
            "Location": {wn.synset("abstraction.n.06"): {"frequency": 5, "conf": 1.0}},
        },
    },
    "acknowledge": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Topic": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
            "Attribute": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "refuse": {
        "metaphor": {
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.14285714285714285,
                }
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 6, "conf": 0.6},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.1},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.1},
                wn.synset("abstraction.n.01"): {"frequency": 1, "conf": 0.1},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.1},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.8571428571428571},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
            },
        },
    },
    "engage": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            }
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}}
        },
    },
    "understand": {
        "metaphor": {},
        "literal": {
            "Stimulus": {
                wn.synset("organization.n.01"): {"frequency": 3, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 6, "conf": 0.5},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.08333333333333333},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.08333333333333333},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "fee": {
        "metaphor": {
            "Agent": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Recipient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Recipient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "remember": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.05714285714285714},
                wn.synset("abstraction.n.06"): {
                    "frequency": 20,
                    "conf": 0.5714285714285714,
                },
                wn.synset("person.n.01"): {"frequency": 8, "conf": 0.22857142857142856},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.02857142857142857},
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.08571428571428572,
                },
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.02857142857142857},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 31, "conf": 0.96875},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.03125},
            },
            "Attribute": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "exist": {
        "metaphor": {
            "Theme": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.25},
            }
        },
    },
    "step": {
        "metaphor": {
            "Destination": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Destination": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "lay": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.75,
                    "score": 0.75,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 0.5,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Destination": {wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "stir": {"metaphor": {}, "literal": {}},
    "bet": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "smother": {
        "metaphor": {
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Destination": {wn.synset("plant.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "describe": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.5,
                    "score": 0.1875,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.3333333333333333,
                },
            },
            "Agent": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.07692307692307693,
                },
            },
            "Attribute": {
                wn.synset("substance.n.01"): {
                    "frequency": 3,
                    "conf": 1.0,
                    "score": 0.2727272727272727,
                }
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 12, "conf": 0.9230769230769231},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                },
            },
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.07142857142857142},
                wn.synset("abstraction.n.06"): {
                    "frequency": 13,
                    "conf": 0.4642857142857143,
                },
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.03571428571428571},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.07142857142857142,
                },
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.03571428571428571},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.07142857142857142},
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.07142857142857142},
                wn.synset("idea.n.01"): {"frequency": 4, "conf": 0.14285714285714285},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.03571428571428571,
                },
            },
            "Attribute": {
                wn.synset("substance.n.01"): {"frequency": 8, "conf": 0.8888888888888888},
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.1111111111111111},
            },
        },
    },
    "submit": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
        },
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Recipient": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "clatter": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "consume": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}}
        },
    },
    "permit": {
        "metaphor": {
            "Beneficiary": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.6666666666666666,
                },
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
            "Beneficiary": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "plant": {
        "metaphor": {},
        "literal": {
            "Destination": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.3333333333333333},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                },
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
        },
    },
    "weave": {"metaphor": {}, "literal": {}},
    "grant": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "protest": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "manifest": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 3, "conf": 0.5},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            }
        },
    },
    "progress": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "travel": {
        "metaphor": {
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "search": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.3333333333333333,
                },
            },
            "Location": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Location": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "monitor": {
        "metaphor": {},
        "literal": {
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.2},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.2},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "wander": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "influence": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.42857142857142855,
                },
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.2857142857142857},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "concentrate": {
        "metaphor": {
            "Agent": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.4,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.4,
                    "score": 0.6666666666666666,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2, "score": 0.5},
            },
            "Topic": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Topic": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "receive": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 1.0, "score": 0.25}
            },
            "Agent": {
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.1,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 9, "conf": 0.5625},
                wn.synset("communication.n.02"): {"frequency": 5, "conf": 0.3125},
                wn.synset("natural_phenomenon.n.01"): {"frequency": 1, "conf": 0.0625},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.0625},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 9, "conf": 0.8181818181818182},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                },
            },
        },
    },
    "plan": {"metaphor": {}, "literal": {}},
    "identify": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.2857142857142857,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.07142857142857142},
                wn.synset("region.n.03"): {"frequency": 2, "conf": 0.14285714285714285},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.07142857142857142},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.07142857142857142},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                },
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.07142857142857142},
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.14285714285714285},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Attribute": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "stumble": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "enable": {
        "metaphor": {
            "Patient": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "deny": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "dress": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "constitute": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.14285714285714285,
                }
            },
        },
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 6, "conf": 0.75},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.125},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
        },
    },
    "compare": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
            }
        },
    },
    "farm": {"metaphor": {}, "literal": {}},
    "cause": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 5, "conf": 1.0}}
        },
    },
    "commend": {"metaphor": {}, "literal": {}},
    "check": {
        "metaphor": {},
        "literal": {
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4},
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.6},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 7, "conf": 1.0}},
        },
    },
    "sail": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
            },
            "Theme": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "conduct": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 6,
                    "conf": 0.75,
                    "score": 0.6666666666666666,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.75,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 1.0}}
        },
    },
    "hasten": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "comfort": {"metaphor": {}, "literal": {}},
    "recognize": {
        "metaphor": {},
        "literal": {
            "Stimulus": {
                wn.synset("person.n.01"): {"frequency": 5, "conf": 0.7142857142857143},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                },
            }
        },
    },
    "march": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "treat": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.5,
                },
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
            "Theme": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.5,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
            "Attribute": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.25}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.375},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.125},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.125},
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Attribute": {wn.synset("substance.n.01"): {"frequency": 3, "conf": 1.0}},
        },
    },
    "listen": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.8},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.2},
            },
            "Theme": {wn.synset("machine.n.04"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "invest": {
        "metaphor": {},
        "literal": {
            "Recipient": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "recover": {
        "metaphor": {
            "Theme": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}}
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            }
        },
    },
    "divide": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Patient": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            }
        },
    },
    "lie": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.6666666666666666,
                },
            }
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "circulate": {"metaphor": {}, "literal": {}},
    "leak": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {},
    },
    "rescue": {"metaphor": {}, "literal": {}},
    "ascend": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}}},
    },
    "send": {
        "metaphor": {
            "Destination": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
            },
            "Theme": {
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.3333333333333333,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.42857142857142855,
                    "score": 0.75,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 0.2,
                },
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 7, "conf": 0.875},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.125},
            },
            "Destination": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.5},
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.4},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.2},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.1},
                wn.synset("communication.n.02"): {"frequency": 3, "conf": 0.3},
            },
        },
    },
    "attract": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5, "score": 1.0},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25, "score": 1.0},
            },
            "Agent": {
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Agent": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}},
            "Patient": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "prevent": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.16666666666666666,
                }
            },
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 0.5},
            },
        },
        "literal": {
            "Theme": {
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.38461538461538464,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.07692307692307693},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.23076923076923078},
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.07692307692307693},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.07692307692307693},
            },
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "clean": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.5},
            },
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "oppose": {
        "metaphor": {
            "Patient": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.5,
                }
            }
        },
        "literal": {
            "Patient": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
            }
        },
    },
    "inherit": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "overcome": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "sip": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "tuck": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Destination": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "contaminate": {
        "metaphor": {},
        "literal": {
            "Destination": {wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "fill": {
        "metaphor": {
            "Destination": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.6, "score": 0.75},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 0.3333333333333333,
                },
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "end": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
            }
        },
    },
    "invite": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.2857142857142857,
                }
            },
            "Theme": {
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("communication.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.6666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 5, "conf": 1.0}},
        },
    },
    "gouge": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}}},
    },
    "reverse": {
        "metaphor": {
            "Patient": {
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.5, "score": 1.0},
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("location.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Patient": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "accumulate": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "promote": {
        "metaphor": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.5,
                    "score": 0.6666666666666666,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.6666666666666666,
                }
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "derive": {"metaphor": {}, "literal": {}},
    "notice": {
        "metaphor": {},
        "literal": {
            "Stimulus": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 9,
                    "conf": 0.6428571428571429,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.14285714285714285},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.07142857142857142,
                },
                wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 0.07142857142857142},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.07142857142857142},
            }
        },
    },
    "waste": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "spend": {
        "metaphor": {
            "Asset": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.6666666666666666,
                }
            },
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 7,
                    "conf": 0.5833333333333334,
                    "score": 0.6363636363636364,
                },
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0}},
            "Asset": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "lurch": {
        "metaphor": {
            "Theme": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "place": {
        "metaphor": {
            "Destination": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 4,
                    "conf": 0.6666666666666666,
                    "score": 0.5,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                    "score": 0.6666666666666666,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0, "score": 0.6}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.5},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.125},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.125},
            },
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.4},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "research": {"metaphor": {}, "literal": {}},
    "miss": {"metaphor": {}, "literal": {}},
    "speak": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.05263157894736842,
                }
            },
            "Topic": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 18, "conf": 0.9473684210526315},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.05263157894736842,
                },
            }
        },
    },
    "taint": {
        "metaphor": {
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "leap": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "strip": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "disappoint": {"metaphor": {}, "literal": {}},
    "list": {
        "metaphor": {
            "Topic": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 0.5},
            }
        },
        "literal": {
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "soar": {
        "metaphor": {
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "cast": {
        "metaphor": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
            },
        },
        "literal": {"Theme": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "border": {"metaphor": {}, "literal": {}},
    "design": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "negotiate": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "exploit": {
        "metaphor": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}}
        },
    },
    "reach": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.3076923076923077,
                    "score": 0.8,
                },
                wn.synset("location.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 2,
                    "conf": 0.15384615384615385,
                    "score": 0.6666666666666666,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
                wn.synset("currency.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
                wn.synset("body_part.n.01"): {
                    "frequency": 1,
                    "conf": 0.07692307692307693,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 6, "conf": 0.6, "score": 0.6},
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.4, "score": 1.0},
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0}},
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.25},
            },
        },
    },
    "follow": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 4,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("location.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.25,
                    "score": 0.5,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 6, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 1.0}},
            "Location": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "peer": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25},
            },
        },
    },
    "manage": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 8,
                    "conf": 0.8,
                    "score": 0.7272727272727273,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.1,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.1, "score": 1.0},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.375,
                    "score": 0.6,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.125, "score": 0.5},
                wn.synset("artifact.n.01"): {
                    "frequency": 2,
                    "conf": 0.25,
                    "score": 0.6666666666666666,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.125, "score": 1.0},
                wn.synset("sound.n.01"): {"frequency": 1, "conf": 0.125, "score": 1.0},
            },
        },
        "literal": {
            "Theme": {
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.18181818181818182,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.09090909090909091},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 3,
                    "conf": 0.2727272727272727,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.18181818181818182,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.09090909090909091},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.09090909090909091},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.09090909090909091},
            },
            "Agent": {
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.75},
            },
        },
    },
    "administer": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "shed": {
        "metaphor": {
            "Theme": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {},
    },
    "satisfy": {"metaphor": {}, "literal": {}},
    "encourage": {
        "metaphor": {
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.25}
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Recipient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.2}
            },
        },
        "literal": {
            "Recipient": {wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0}},
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.75},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.25},
            },
            "Theme": {wn.synset("natural_phenomenon.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "style": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "rush": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.25}
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}}},
    },
    "decorate": {
        "metaphor": {},
        "literal": {
            "Destination": {
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            }
        },
    },
    "base": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "possess": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 0.5},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "perceive": {
        "metaphor": {
            "Stimulus": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5, "score": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {"Stimulus": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}}},
    },
    "entrust": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "advise": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Topic": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}},
            "Recipient": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
        },
    },
    "stick": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
            "Destination": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "proceed": {
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
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            }
        },
        "literal": {},
    },
    "throw": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.25, "score": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5, "score": 1.0},
            },
            "Destination": {
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "lower": {"metaphor": {}, "literal": {}},
    "impose": {
        "metaphor": {
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.25},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.75},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
            },
        },
    },
    "tackle": {
        "metaphor": {
            "Theme": {
                wn.synset("state.n.02"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
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
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "depend": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.4},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.2},
            },
            "Theme": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "harden": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "forgive": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Attribute": {
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "maintain": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.5,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
            },
            "Agent": {wn.synset("food.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}},
        },
        "literal": {"Theme": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}}},
    },
    "trip": {"metaphor": {}, "literal": {}},
    "bite": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "result": {
        "metaphor": {},
        "literal": {
            "Location": {wn.synset("abstraction.n.06"): {"frequency": 6, "conf": 1.0}},
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.5},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
            },
        },
    },
    "wade": {"metaphor": {}, "literal": {}},
    "fight": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
            }
        },
        "literal": {
            "Agent": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "clutch": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("artifact.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "succumb": {
        "metaphor": {
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "dissuade": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "bake": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "bore": {"metaphor": {}, "literal": {}},
    "deduct": {"metaphor": {}, "literal": {}},
    "arrange": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
            "Product": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "convey": {
        "metaphor": {
            "Topic": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {},
    },
    "gleam": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "illustrate": {
        "metaphor": {
            "Theme": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
            },
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.8333333333333334,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.16666666666666666},
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
        },
    },
    "separate": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Patient": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "smell": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Stimulus": {
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "preach": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Topic": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "rise": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.4166666666666667,
                    "score": 0.8333333333333334,
                },
                wn.synset("sound.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 2,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.08333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.16666666666666666,
                    "score": 0.6666666666666666,
                },
            }
        },
        "literal": {
            "Patient": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            }
        },
    },
    "enrage": {"metaphor": {}, "literal": {}},
    "realize": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 8, "conf": 1.0}},
            "Theme": {
                wn.synset("state.n.01"): {"frequency": 2, "conf": 0.18181818181818182},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.09090909090909091},
                wn.synset("communication.n.02"): {
                    "frequency": 3,
                    "conf": 0.2727272727272727,
                },
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.36363636363636365},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                },
            },
            "Stimulus": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "choose": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 5, "conf": 0.7142857142857143},
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
            },
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.125},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.5},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.125},
            },
        },
    },
    "substitute": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {
            "Location": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "bash": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Location": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "mention": {
        "metaphor": {},
        "literal": {
            "Topic": {
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.25},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.25},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
        },
    },
    "test": {
        "metaphor": {},
        "literal": {
            "Location": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            }
        },
    },
    "analyse": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.16666666666666666},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "realise": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("state.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
            },
        },
    },
    "shock": {"metaphor": {}, "literal": {}},
    "slaughter": {"metaphor": {}, "literal": {}},
    "aggravate": {"metaphor": {}, "literal": {}},
    "vanish": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "act": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
            "Attribute": {
                wn.synset("substance.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            },
        },
        "literal": {
            "Attribute": {
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.25},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "drift": {
        "metaphor": {
            "Theme": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "wave": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "rain": {"metaphor": {}, "literal": {}},
    "bark": {"metaphor": {}, "literal": {}},
    "dream": {"metaphor": {}, "literal": {}},
    "close": {
        "metaphor": {
            "Patient": {
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.3333333333333333,
                },
            }
        },
        "literal": {
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.5},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
        },
    },
    "kill": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.4},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.2},
            },
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "protect": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.8333333333333334,
                },
            }
        },
    },
    "seize": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "surprise": {"metaphor": {}, "literal": {}},
    "confirm": {
        "metaphor": {
            "Topic": {
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            }
        },
        "literal": {
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25},
            }
        },
    },
    "finance": {
        "metaphor": {
            "Recipient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Recipient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            }
        },
    },
    "deal": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 4,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
            }
        },
        "literal": {
            "Agent": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "greet": {
        "metaphor": {
            "Theme": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}}},
    },
    "slump": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "risk": {
        "metaphor": {
            "Theme": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}}
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "minimize": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "mitigate": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25},
            }
        },
    },
    "slow": {
        "metaphor": {
            "Patient": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "transform": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Patient": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "shudder": {"metaphor": {}, "literal": {}},
    "cart": {"metaphor": {}, "literal": {}},
    "blast": {"metaphor": {}, "literal": {}},
    "stress": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.8, "score": 0.8},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
            },
            "Destination": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.2, "score": 1.0},
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.4, "score": 1.0},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.2, "score": 0.2},
            },
        },
        "literal": {
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.8},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "increase": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 0.2}
            }
        },
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 8, "conf": 0.8},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.1},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.1},
            },
            "Attribute": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "roll": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "shake": {
        "metaphor": {
            "Theme": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.25}
            },
        },
        "literal": {
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.6},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.2},
            },
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Recipient": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "surge": {
        "metaphor": {
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "affect": {"metaphor": {}, "literal": {}},
    "remedy": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "surround": {
        "metaphor": {
            "Destination": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.6666666666666666,
                }
            }
        },
        "literal": {
            "Destination": {
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "push": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.2},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Destination": {
                wn.synset("location.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.6666666666666666,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
            },
        },
        "literal": {
            "Destination": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0}},
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.3333333333333333},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.5},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "participate": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "vote": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.2},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.4},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "thrill": {"metaphor": {}, "literal": {}},
    "update": {
        "metaphor": {},
        "literal": {
            "Recipient": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "oblige": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
            "Predicate": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Patient": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
            }
        },
    },
    "reveal": {
        "metaphor": {
            "Theme": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Topic": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.75,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
            },
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "prove": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.09090909090909091},
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.36363636363636365},
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.36363636363636365,
                },
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.09090909090909091},
            },
            "Topic": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
            },
        },
    },
    "book": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            }
        },
    },
    "enforce": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.3333333333333333},
            }
        },
    },
    "assure": {
        "metaphor": {},
        "literal": {
            "Recipient": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Topic": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "attain": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {wn.synset("state.n.02"): {"frequency": 3, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "address": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            }
        },
        "literal": {},
    },
    "welcome": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.6666666666666666,
                    "score": 0.8,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 0.5,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "fund": {
        "metaphor": {},
        "literal": {
            "Recipient": {
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Agent": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "trace": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "park": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "demolish": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "share": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 5,
                    "conf": 1.0,
                    "score": 0.7142857142857143,
                }
            },
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 0.3333333333333333,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.125, "score": 1.0},
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.25,
                    "score": 0.6666666666666666,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.125,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
        },
    },
    "worry": {"metaphor": {}, "literal": {}},
    "hide": {
        "metaphor": {
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Patient": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "fear": {"metaphor": {}, "literal": {}},
    "paint": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.2},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("natural_phenomenon.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "pertain": {"metaphor": {}, "literal": {}},
    "steam": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "glare": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 4, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "match": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Patient": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "reckon": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 7,
                    "conf": 1.0,
                    "score": 0.7777777777777778,
                }
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("state.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "shove": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Destination": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "apply": {
        "metaphor": {
            "Theme": {
                wn.synset("idea.n.01"): {
                    "frequency": 3,
                    "conf": 0.3333333333333333,
                    "score": 0.75,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.2222222222222222,
                    "score": 0.6666666666666666,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.2222222222222222,
                    "score": 0.6666666666666666,
                },
                wn.synset("food.n.01"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 1.0,
                },
            }
        },
        "literal": {
            "Theme": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            }
        },
    },
    "struggle": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5, "score": 1.0},
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
            }
        },
        "literal": {},
    },
    "snatch": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "lend": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Recipient": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "crash": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "afford": {
        "metaphor": {
            "Theme": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.75},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
        },
    },
    "report": {
        "metaphor": {
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.3333333333333333},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.75},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.25},
            },
        },
    },
    "investigate": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            },
            "Location": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Location": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "chew": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "adopt": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            }
        },
    },
    "insist": {
        "metaphor": {},
        "literal": {
            "Topic": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
        },
    },
    "etch": {"metaphor": {}, "literal": {}},
    "hop": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            }
        },
    },
    "shut": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.4},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.2},
            }
        },
    },
    "differ": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("animal.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "rely": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Theme": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "breast": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "implement": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                },
                wn.synset("idea.n.01"): {"frequency": 3, "conf": 0.42857142857142855},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
            },
            "Agent": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "question": {
        "metaphor": {
            "Topic": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Topic": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}},
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "sustain": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "admit": {
        "metaphor": {
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4, "score": 1.0},
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.4,
                    "score": 0.3333333333333333,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.4,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Recipient": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Topic": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.8},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.2},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
        },
    },
    "refine": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "wash": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "stimulate": {"metaphor": {}, "literal": {}},
    "suppress": {
        "metaphor": {
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {
            "Patient": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "remove": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
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
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "agitate": {"metaphor": {}, "literal": {}},
    "vary": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
            }
        },
    },
    "operate": {
        "metaphor": {
            "Theme": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.25, "score": 0.25},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 0.5,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
            }
        },
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 3, "conf": 0.75},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
            }
        },
    },
    "order": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.2}
            },
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0}},
        },
    },
    "bow": {
        "metaphor": {
            "Agent": {wn.synset("plant.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}}
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Recipient": {wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "raise": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.75, "score": 0.5},
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 11,
                    "conf": 0.7333333333333333,
                    "score": 1.0,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.13333333333333333,
                    "score": 1.0,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                    "score": 1.0,
                },
            },
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Theme": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "pause": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
            }
        },
    },
    "measure": {
        "metaphor": {
            "Theme": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.5,
                }
            }
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.25},
            }
        },
    },
    "score": {
        "metaphor": {
            "Asset": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.6666666666666666,
                }
            },
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "review": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("abstraction.n.06"): {
                    "frequency": 5,
                    "conf": 0.7142857142857143,
                },
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.75},
            },
            "Attribute": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "applaud": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "blow": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "educate": {
        "metaphor": {},
        "literal": {
            "Topic": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Recipient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "tremble": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "eliminate": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "joke": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "air": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "slip": {
        "metaphor": {
            "Patient": {
                wn.synset("time.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
            }
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Patient": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
        },
    },
    "originate": {"metaphor": {}, "literal": {}},
    "allege": {
        "metaphor": {},
        "literal": {
            "Topic": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.3333333333333333},
            }
        },
    },
    "comment": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0}}},
    },
    "hope": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.09090909090909091,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 7,
                    "conf": 0.6363636363636364,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.18181818181818182},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.09090909090909091},
            }
        },
    },
    "dip": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "control": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5, "score": 1.0},
                wn.synset("abstraction.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {"Theme": {wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0}}},
    },
    "freeze": {"metaphor": {}, "literal": {}},
    "erect": {
        "metaphor": {},
        "literal": {
            "Product": {wn.synset("artifact.n.01"): {"frequency": 2, "conf": 1.0}}
        },
    },
    "part": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "copy": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "exercise": {
        "metaphor": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("abstraction.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "long": {"metaphor": {}, "literal": {}},
    "approve": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "ionize": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("natural_phenomenon.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "construct": {
        "metaphor": {
            "Material": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "envy": {"metaphor": {}, "literal": {}},
    "fix": {
        "metaphor": {
            "Theme": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Theme": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "moan": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "induce": {"metaphor": {}, "literal": {}},
    "finish": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 14, "conf": 0.9333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.06666666666666667,
                },
            },
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "limit": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "entangle": {"metaphor": {}, "literal": {}},
    "charge": {
        "metaphor": {
            "Recipient": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
        },
        "literal": {"Recipient": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "dismiss": {
        "metaphor": {
            "Theme": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}}
        },
        "literal": {},
    },
    "survive": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "warn": {
        "metaphor": {
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Recipient": {
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Topic": {wn.synset("substance.n.01"): {"frequency": 2, "conf": 1.0}},
            "Recipient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "crush": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "escape": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.6666666666666666,
                }
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "supervise": {"metaphor": {}, "literal": {}},
    "revive": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
        },
        "literal": {},
    },
    "chase": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "injure": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "supply": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Recipient": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.25},
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "congratulate": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "smooth": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "rouse": {
        "metaphor": {
            "Patient": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "envelop": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {},
    },
    "forbid": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}}
        },
    },
    "kiss": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "blame": {
        "metaphor": {
            "Attribute": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Attribute": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Theme": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
        },
    },
    "stash": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "practise": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}}
        },
    },
    "glide": {
        "metaphor": {},
        "literal": {
            "Destination": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "die": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 11, "conf": 0.7333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.26666666666666666,
                },
            }
        },
    },
    "ai": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0}}},
    },
    "threaten": {"metaphor": {}, "literal": {}},
    "join": {
        "metaphor": {
            "Patient": {
                wn.synset("organization.n.01"): {
                    "frequency": 6,
                    "conf": 0.6666666666666666,
                    "score": 0.75,
                },
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.2222222222222222,
                    "score": 0.6666666666666666,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.1111111111111111,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("organization.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.25},
            },
        },
    },
    "bridge": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "daunt": {"metaphor": {}, "literal": {}},
    "tend": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.2}
            }
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.6666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                },
            }
        },
    },
    "blossom": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "wipe": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
            },
        },
    },
    "compile": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
            "Product": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
        },
    },
    "shiver": {"metaphor": {}, "literal": {}},
    "trudge": {
        "metaphor": {},
        "literal": {
            "Destination": {wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "exacerbate": {"metaphor": {}, "literal": {}},
    "curve": {"metaphor": {}, "literal": {}},
    "merge": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "claim": {
        "metaphor": {
            "Topic": {
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.4,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.125},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 7, "conf": 1.0}},
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.5},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
        },
    },
    "squelch": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "dash": {"metaphor": {}, "literal": {}},
    "rip": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "falter": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "blush": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "provoke": {"metaphor": {}, "literal": {}},
    "approach": {
        "metaphor": {
            "Destination": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.6666666666666666,
                },
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
        },
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Destination": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "perform": {
        "metaphor": {
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            }
        },
    },
    "frame": {
        "metaphor": {
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {},
    },
    "assign": {
        "metaphor": {
            "Theme": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}},
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}}
        },
    },
    "canvas": {"metaphor": {}, "literal": {}},
    "emphasize": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 0.5}
            },
            "Agent": {
                wn.synset("state.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "foresee": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "mutter": {
        "metaphor": {},
        "literal": {
            "Topic": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "hit": {
        "metaphor": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
            "Location": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
            "Patient": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "sort": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("sound.n.04"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("state.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
        },
        "literal": {},
    },
    "explain": {
        "metaphor": {
            "Topic": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}},
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.2},
            },
        },
        "literal": {
            "Topic": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.125},
                wn.synset("abstraction.n.06"): {"frequency": 6, "conf": 0.75},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.125},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.8},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.2},
            },
        },
    },
    "click": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "furnish": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {},
    },
    "isolate": {
        "metaphor": {
            "Patient": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "sense": {"metaphor": {}, "literal": {}},
    "count": {
        "metaphor": {
            "Attribute": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "hire": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "banish": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "worsen": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("state.n.02"): {"frequency": 2, "conf": 1.0}},
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "experience": {
        "metaphor": {
            "Stimulus": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "doubt": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0}},
            "Theme": {
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "dazzle": {"metaphor": {}, "literal": {}},
    "extend": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.3333333333333333,
                    "score": 0.6666666666666666,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                    "score": 0.5,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.2},
                wn.synset("body_part.n.01"): {"frequency": 2, "conf": 0.4},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.2},
            }
        },
    },
    "deliver": {
        "metaphor": {
            "Theme": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "discover": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {"Theme": {wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "buoy": {"metaphor": {}, "literal": {}},
    "guarantee": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("time.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "grip": {"metaphor": {}, "literal": {}},
    "snap": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.8, "score": 1.0},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
            },
            "Topic": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Recipient": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "dare": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}}},
    },
    "conclude": {
        "metaphor": {
            "Agent": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.5,
                },
            },
            "Theme": {
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.5,
                },
            },
        },
        "literal": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                },
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.2857142857142857},
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.42857142857142855,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.14285714285714285},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.8},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.2},
            },
        },
    },
    "spoil": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "study": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "devise": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "arrest": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "succeed": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Agent": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "repeat": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Topic": {
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.5},
                wn.synset("natural_phenomenon.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "enclose": {"metaphor": {}, "literal": {}},
    "knight": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "revert": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
            }
        },
    },
    "swirl": {
        "metaphor": {
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "murmur": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "retire": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            }
        },
        "literal": {},
    },
    "loathe": {"metaphor": {}, "literal": {}},
    "dispose": {"metaphor": {}, "literal": {}},
    "manufacture": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "consolidate": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.6666666666666666,
                }
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
        },
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.25},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "knock": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Patient": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "expand": {
        "metaphor": {
            "Patient": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "diminish": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "deport": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "commence": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "eye": {
        "metaphor": {},
        "literal": {
            "Stimulus": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "gush": {"metaphor": {}, "literal": {}},
    "percolate": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "answer": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "pursue": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "alter": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "replace": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {"Theme": {wn.synset("artifact.n.01"): {"frequency": 3, "conf": 1.0}}},
    },
    "confine": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.5,
                    "score": 0.6666666666666666,
                },
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.25, "score": 1.0},
            }
        },
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "befall": {"metaphor": {}, "literal": {}},
    "prod": {"metaphor": {}, "literal": {}},
    "attempt": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Theme": {
                wn.synset("idea.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
        },
    },
    "commit": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5, "score": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
            }
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.3333333333333333},
            }
        },
    },
    "organize": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "shrug": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 5, "conf": 0.8333333333333334},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
            }
        },
    },
    "gasp": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "register": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "chop": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.75},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
            },
            "Patient": {
                wn.synset("natural_phenomenon.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "couple": {
        "metaphor": {
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "presume": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "land": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("animal.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "tailor": {
        "metaphor": {},
        "literal": {
            "Beneficiary": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "accept": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.3333333333333333,
                },
            },
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.5714285714285714,
                },
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
                wn.synset("plant.n.02"): {"frequency": 1, "conf": 0.14285714285714285},
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
    "focus": {
        "metaphor": {
            "Agent": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
            }
        },
        "literal": {
            "Topic": {wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "collapse": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.25, "score": 1.0},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
            },
        },
        "literal": {},
    },
    "twiddle": {"metaphor": {}, "literal": {}},
    "formulate": {"metaphor": {}, "literal": {}},
    "weaken": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "grunt": {
        "metaphor": {},
        "literal": {
            "Topic": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "relate": {
        "metaphor": {},
        "literal": {"Topic": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "exceed": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            }
        },
    },
    "sacrifice": {"metaphor": {}, "literal": {}},
    "cook": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "simmer": {"metaphor": {}, "literal": {}},
    "categorize": {"metaphor": {}, "literal": {}},
    "burn": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "represent": {
        "metaphor": {
            "Theme": {
                wn.synset("time.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("state.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("idea.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 0.3333333333333333,
                },
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
        },
        "literal": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.2857142857142857,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 4,
                    "conf": 0.5714285714285714,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.14285714285714285},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
        },
    },
    "adapt": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "flop": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "regret": {"metaphor": {}, "literal": {}},
    "shop": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "hat": {"metaphor": {}, "literal": {}},
    "detect": {
        "metaphor": {
            "Stimulus": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            }
        },
        "literal": {
            "Stimulus": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            }
        },
    },
    "film": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "assimilate": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "tighten": {
        "metaphor": {
            "Patient": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "hand": {
        "metaphor": {
            "Destination": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "sleep": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 5, "conf": 1.0}}},
    },
    "rest": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "switch": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Patient": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "dig": {
        "metaphor": {
            "Agent": {
                wn.synset("region.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "exchange": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "expel": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "cheer": {"metaphor": {}, "literal": {}},
    "intimidate": {"metaphor": {}, "literal": {}},
    "hammer": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "record": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Theme": {
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.2},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4},
            },
        },
    },
    "smoke": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "stare": {
        "metaphor": {
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "attack": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "sprinkle": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "steal": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
        },
    },
    "announce": {
        "metaphor": {
            "Agent": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
            "Topic": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Topic": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.25},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.25},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "cut": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.6666666666666666,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "complain": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 5, "conf": 0.8333333333333334},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
            },
            "Topic": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "double": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.6666666666666666,
                }
            }
        },
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "denounce": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Attribute": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {"Theme": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "flourish": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "spew": {"metaphor": {}, "literal": {}},
    "balance": {"metaphor": {}, "literal": {}},
    "launch": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.42857142857142855,
                    "score": 1.0,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 3,
                    "conf": 0.42857142857142855,
                    "score": 1.0,
                },
                wn.synset("region.n.01"): {
                    "frequency": 1,
                    "conf": 0.14285714285714285,
                    "score": 1.0,
                },
            },
        },
        "literal": {
            "Theme": {wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 1.0}},
            "Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "range": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {},
    },
    "accompany": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.6666666666666666,
                }
            },
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.2},
            },
        },
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "whine": {
        "metaphor": {
            "Agent": {
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "complete": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "refer": {
        "metaphor": {
            "Agent": {
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 1.0,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.25,
                    "score": 0.3333333333333333,
                },
            }
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Recipient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "post": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "pronounce": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "prohibit": {"metaphor": {}, "literal": {}},
    "disperse": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "imagine": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.2},
                wn.synset("person.n.01"): {"frequency": 4, "conf": 0.8},
            },
            "Stimulus": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.3333333333333333},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.3333333333333333},
            },
        },
    },
    "pitch": {
        "metaphor": {
            "Theme": {
                wn.synset("sound.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Destination": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "hate": {"metaphor": {}, "literal": {}},
    "repel": {"metaphor": {}, "literal": {}},
    "flush": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "police": {"metaphor": {}, "literal": {}},
    "blend": {
        "metaphor": {
            "Patient": {
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "tempt": {"metaphor": {}, "literal": {}},
    "restrict": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            }
        },
        "literal": {},
    },
    "expunge": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "scold": {"metaphor": {}, "literal": {}},
    "withdraw": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "intend": {"metaphor": {}, "literal": {}},
    "wind": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "modify": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "scrap": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "interpret": {
        "metaphor": {
            "Theme": {
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "conceive": {
        "metaphor": {
            "Attribute": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "integrate": {
        "metaphor": {
            "Patient": {
                wn.synset("region.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Patient": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "retreat": {"metaphor": {}, "literal": {}},
    "drag": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            }
        },
        "literal": {"Theme": {wn.synset("artifact.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "overwhelm": {"metaphor": {}, "literal": {}},
    "prompt": {
        "metaphor": {
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "salute": {
        "metaphor": {},
        "literal": {
            "Recipient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "pique": {"metaphor": {}, "literal": {}},
    "tire": {"metaphor": {}, "literal": {}},
    "swear": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "shatter": {
        "metaphor": {
            "Patient": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {},
    },
    "rid": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}}},
    },
    "load": {
        "metaphor": {
            "Destination": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "fly": {
        "metaphor": {},
        "literal": {
            "Location": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "decay": {
        "metaphor": {
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "entice": {"metaphor": {}, "literal": {}},
    "mount": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "strengthen": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4, "score": 1.0},
                wn.synset("substance.n.01"): {"frequency": 2, "conf": 0.4, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "sprawl": {"metaphor": {}, "literal": {}},
    "urge": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "bury": {
        "metaphor": {
            "Theme": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {"Agent": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "smash": {
        "metaphor": {
            "Patient": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Patient": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "advance": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Theme": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "donate": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "shift": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.4, "score": 1.0},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.2, "score": 0.5},
            },
            "Agent": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
        },
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "descend": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "cool": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "comprise": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            }
        },
    },
    "distribute": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            }
        },
        "literal": {},
    },
    "telephone": {
        "metaphor": {
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "articulate": {
        "metaphor": {
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "frown": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "train": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.6666666666666666,
                },
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "ally": {"metaphor": {}, "literal": {}},
    "transmit": {"metaphor": {}, "literal": {}},
    "cheat": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("sound.n.04"): {"frequency": 1, "conf": 1.0}}},
    },
    "fuel": {"metaphor": {}, "literal": {}},
    "damage": {"metaphor": {}, "literal": {}},
    "wring": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "slide": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("artifact.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "traverse": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "confide": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Topic": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "auction": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
            },
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "exert": {
        "metaphor": {
            "Theme": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}}
        },
        "literal": {},
    },
    "deepen": {
        "metaphor": {
            "Patient": {
                wn.synset("sound.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "loom": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "respond": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 0.6666666666666666,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            }
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("machine.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Recipient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "interview": {
        "metaphor": {},
        "literal": {
            "Recipient": {wn.synset("person.n.01"): {"frequency": 4, "conf": 1.0}},
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "wheel": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "envisage": {
        "metaphor": {},
        "literal": {
            "Attribute": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "rag": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "restore": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "nail": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Patient": {
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
        },
        "literal": {},
    },
    "solve": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("state.n.02"): {"frequency": 4, "conf": 0.6666666666666666},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "arouse": {"metaphor": {}, "literal": {}},
    "improvise": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "overhear": {
        "metaphor": {},
        "literal": {"Stimulus": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "gaze": {
        "metaphor": {
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Theme": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "disgruntle": {"metaphor": {}, "literal": {}},
    "appreciate": {"metaphor": {}, "literal": {}},
    "chuckle": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "devote": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "punish": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {"Patient": {wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0}}},
    },
    "desert": {"metaphor": {}, "literal": {}},
    "alleviate": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("state.n.02"): {"frequency": 2, "conf": 1.0}}},
    },
    "benefit": {"metaphor": {}, "literal": {}},
    "gather": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "glaze": {"metaphor": {}, "literal": {}},
    "flock": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "sigh": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 5, "conf": 1.0}}},
    },
    "widen": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "marvel": {"metaphor": {}, "literal": {}},
    "shoot": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "necessitate": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 1.0}}
        },
    },
    "perch": {"metaphor": {}, "literal": {}},
    "compose": {"metaphor": {}, "literal": {}},
    "hurt": {"metaphor": {}, "literal": {}},
    "scribble": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("communication.n.02"): {"frequency": 2, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "boost": {
        "metaphor": {
            "Theme": {
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
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "enamel": {"metaphor": {}, "literal": {}},
    "import": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("vehicle.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "date": {"metaphor": {}, "literal": {}},
    "gallop": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "prop": {"metaphor": {}, "literal": {}},
    "redecorate": {
        "metaphor": {},
        "literal": {
            "Destination": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "grind": {"metaphor": {}, "literal": {}},
    "obscure": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}},
            "Patient": {wn.synset("region.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "jog": {"metaphor": {}, "literal": {}},
    "suspect": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "side": {"metaphor": {}, "literal": {}},
    "witness": {
        "metaphor": {
            "Stimulus": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "allocate": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.75},
                wn.synset("natural_phenomenon.n.01"): {"frequency": 1, "conf": 0.25},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "rank": {
        "metaphor": {
            "Attribute": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "criticize": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "free": {
        "metaphor": {
            "Theme": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "addict": {
        "metaphor": {},
        "literal": {
            "Stimulus": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "touch": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            }
        },
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "shine": {
        "metaphor": {
            "Theme": {
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {},
    },
    "mark": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "praise": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "ensue": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "bind": {
        "metaphor": {
            "Destination": {
                wn.synset("substance.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.6, "score": 1.0},
                wn.synset("organization.n.01"): {
                    "frequency": 2,
                    "conf": 0.4,
                    "score": 1.0,
                },
            },
        },
        "literal": {},
    },
    "recognise": {
        "metaphor": {
            "Stimulus": {
                wn.synset("person.n.01"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.6666666666666666,
                }
            }
        },
        "literal": {"Stimulus": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "price": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "correct": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Patient": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "enact": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {},
    },
    "decrease": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "cleanse": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "expose": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "contribute": {
        "metaphor": {
            "Recipient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "imply": {
        "metaphor": {
            "Topic": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Topic": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "abate": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("natural_phenomenon.n.01"): {"frequency": 1, "conf": 1.0}
            }
        },
    },
    "contemplate": {"metaphor": {}, "literal": {}},
    "bridle": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Destination": {wn.synset("substance.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "cling": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "correspond": {"metaphor": {}, "literal": {}},
    "charm": {"metaphor": {}, "literal": {}},
    "rectify": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}}},
    },
    "drench": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Destination": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
        },
        "literal": {},
    },
    "print": {"metaphor": {}, "literal": {}},
    "underlie": {"metaphor": {}, "literal": {}},
    "spot": {
        "metaphor": {},
        "literal": {
            "Stimulus": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "hurtle": {"metaphor": {}, "literal": {}},
    "exclude": {
        "metaphor": {
            "Theme": {
                wn.synset("region.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "beat": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Patient": {
                wn.synset("region.n.03"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("region.n.01"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                    "score": 1.0,
                },
            },
        },
        "literal": {},
    },
    "probe": {
        "metaphor": {
            "Location": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "request": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "elaborate": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "whip": {
        "metaphor": {
            "Location": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "convert": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.75},
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.25},
            }
        },
    },
    "direct": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 3,
                    "conf": 0.75,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.25, "score": 1.0},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "process": {"metaphor": {}, "literal": {}},
    "persuade": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Predicate": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "bond": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "command": {"metaphor": {}, "literal": {}},
    "curse": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("clothing.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "slash": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "reinforce": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "photograph": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "tear": {"metaphor": {}, "literal": {}},
    "beg": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "dim": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "shimmer": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "murder": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "picture": {"metaphor": {}, "literal": {}},
    "plat": {"metaphor": {}, "literal": {}},
    "cry": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "concede": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.5, "score": 0.5},
            },
        },
        "literal": {
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "abandon": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "hesitate": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}}},
    },
    "intersect": {"metaphor": {}, "literal": {}},
    "cite": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("food.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
            "Attribute": {wn.synset("substance.n.01"): {"frequency": 2, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "tie": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "evaporate": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "defend": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "overstate": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "license": {"metaphor": {}, "literal": {}},
    "stride": {"metaphor": {}, "literal": {}},
    "awaken": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "forge": {
        "metaphor": {
            "Product": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "rebuild": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "dodge": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "invent": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "portray": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "pant": {"metaphor": {}, "literal": {}},
    "skirt": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("region.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "equip": {"metaphor": {}, "literal": {}},
    "bath": {
        "metaphor": {
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "notify": {
        "metaphor": {},
        "literal": {
            "Recipient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "celebrate": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
        },
    },
    "intersperse": {"metaphor": {}, "literal": {}},
    "affirm": {
        "metaphor": {},
        "literal": {"Topic": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "unlock": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "dispel": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}}},
    },
    "cram": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "award": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.6666666666666666,
                }
            }
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "depart": {"metaphor": {}, "literal": {}},
    "absorb": {
        "metaphor": {
            "Theme": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "arm": {"metaphor": {}, "literal": {}},
    "despise": {"metaphor": {}, "literal": {}},
    "instruct": {
        "metaphor": {},
        "literal": {
            "Topic": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Recipient": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "relax": {"metaphor": {}, "literal": {}},
    "remark": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Topic": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "dissolve": {
        "metaphor": {
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "belong": {"metaphor": {}, "literal": {}},
    "enter": {
        "metaphor": {
            "Destination": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 0.5},
            },
            "Theme": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
        },
        "literal": {
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.2},
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 0.4},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.2},
                wn.synset("state.n.04"): {"frequency": 1, "conf": 0.2},
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 3, "conf": 0.75},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
            },
        },
    },
    "stalk": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "harm": {"metaphor": {}, "literal": {}},
    "tug": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("state.n.01"): {"frequency": 2, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.25},
            },
        },
    },
    "mix": {
        "metaphor": {
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "calculate": {"metaphor": {}, "literal": {}},
    "fling": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Theme": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
            "Destination": {wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "communicate": {
        "metaphor": {
            "Agent": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}}
        },
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "explore": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            }
        },
    },
    "shin": {
        "metaphor": {
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "capture": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 4, "conf": 0.8, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.2,
                    "score": 1.0,
                },
            },
        },
        "literal": {},
    },
    "trot": {"metaphor": {}, "literal": {}},
    "cooperate": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "store": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "tramp": {"metaphor": {}, "literal": {}},
    "swap": {
        "metaphor": {},
        "literal": {
            "Location": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "wrap": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Destination": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "care": {
        "metaphor": {},
        "literal": {
            "Stimulus": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "safeguard": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "wrestle": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "shadow": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "placate": {"metaphor": {}, "literal": {}},
    "appeal": {"metaphor": {}, "literal": {}},
    "specify": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 0.75},
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.25},
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
            "Attribute": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "lure": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "commission": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "traipse": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "chart": {
        "metaphor": {
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "flow": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "blaze": {"metaphor": {}, "literal": {}},
    "decline": {
        "metaphor": {
            "Patient": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.5,
                }
            }
        },
        "literal": {
            "Patient": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "dry": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "stoop": {"metaphor": {}, "literal": {}},
    "burgle": {"metaphor": {}, "literal": {}},
    "throttle": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "abolish": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "gesture": {
        "metaphor": {},
        "literal": {
            "Agent": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Recipient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "trail": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "stall": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "jump": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "howl": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "ride": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            }
        },
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "position": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "reanimate": {
        "metaphor": {
            "Patient": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {},
    },
    "hurl": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Destination": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "document": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "sharpen": {
        "metaphor": {
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5, "score": 1.0},
            }
        },
        "literal": {},
    },
    "lug": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "flirt": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "bar": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "circle": {"metaphor": {}, "literal": {}},
    "curb": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}}
        },
    },
    "collate": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "sketch": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "reopen": {
        "metaphor": {
            "Patient": {
                wn.synset("communication.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "flee": {
        "metaphor": {},
        "literal": {
            "Location": {wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "fool": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "interweave": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "stroke": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "disclose": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "tour": {"metaphor": {}, "literal": {}},
    "flood": {
        "metaphor": {
            "Agent": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "connect": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "jerk": {
        "metaphor": {
            "Theme": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {"Theme": {wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "vibrate": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "confront": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "abuse": {
        "metaphor": {
            "Theme": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}}
        },
        "literal": {},
    },
    "utilize": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "conform": {"metaphor": {}, "literal": {}},
    "emit": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("sound.n.04"): {"frequency": 2, "conf": 1.0}}},
    },
    "delay": {
        "metaphor": {
            "Theme": {
                wn.synset("region.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "stroll": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "grace": {
        "metaphor": {
            "Theme": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "squeak": {"metaphor": {}, "literal": {}},
    "prickle": {"metaphor": {}, "literal": {}},
    "situate": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "plague": {"metaphor": {}, "literal": {}},
    "tip": {
        "metaphor": {},
        "literal": {
            "Destination": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "drill": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "block": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "usher": {
        "metaphor": {},
        "literal": {
            "Beneficiary": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "flare": {"metaphor": {}, "literal": {}},
    "blunt": {
        "metaphor": {
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "rationalize": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "select": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
        },
    },
    "guess": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("state.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 5, "conf": 1.0}},
        },
    },
    "intrude": {"metaphor": {}, "literal": {}},
    "suspend": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "halt": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "prolong": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}}},
    },
    "pout": {"metaphor": {}, "literal": {}},
    "compromise": {"metaphor": {}, "literal": {}},
    "issue": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.3333333333333333,
                }
            },
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 2, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.16666666666666666,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.16666666666666666},
                wn.synset("communication.n.02"): {
                    "frequency": 4,
                    "conf": 0.6666666666666666,
                },
            },
        },
    },
    "initiate": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "fade": {"metaphor": {}, "literal": {}},
    "moor": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("vehicle.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "purchase": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "stiffen": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "giggle": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "shovel": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "postpone": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "revitalize": {"metaphor": {}, "literal": {}},
    "smack": {"metaphor": {}, "literal": {}},
    "coincide": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "enquire": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}}},
    },
    "sing": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "compete": {"metaphor": {}, "literal": {}},
    "accuse": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "swim": {"metaphor": {}, "literal": {}},
    "shoo": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("animal.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "fasten": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "endure": {"metaphor": {}, "literal": {}},
    "peel": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "pack": {
        "metaphor": {
            "Destination": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("artifact.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
                wn.synset("person.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                    "score": 1.0,
                },
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "humiliate": {"metaphor": {}, "literal": {}},
    "growl": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "depose": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "appoint": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "neglect": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}}
        },
    },
    "sniff": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "insert": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "race": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "dedicate": {"metaphor": {}, "literal": {}},
    "classify": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 0.6666666666666666,
                },
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.3333333333333333},
            },
            "Attribute": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "unfold": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}}
        },
    },
    "twist": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "endorse": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 0.5},
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "disband": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("sound.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "hug": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "distort": {"metaphor": {}, "literal": {}},
    "wail": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "dampen": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "distinguish": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 0.5},
                wn.synset("plant.n.01"): {"frequency": 1, "conf": 0.25},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.25},
            },
            "Agent": {wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "recur": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "memorize": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Topic": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "straggle": {"metaphor": {}, "literal": {}},
    "cough": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "extract": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {
            "Theme": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "scan": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "rattle": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "thin": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "fascinate": {"metaphor": {}, "literal": {}},
    "hunt": {
        "metaphor": {
            "Location": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "guide": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "ascertain": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}}
        },
    },
    "debate": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "derail": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "condemn": {
        "metaphor": {},
        "literal": {
            "Attribute": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.5},
            },
        },
    },
    "amuse": {"metaphor": {}, "literal": {}},
    "upgrade": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "tumble": {
        "metaphor": {
            "Patient": {
                wn.synset("natural_phenomenon.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {},
    },
    "lock": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            }
        },
        "literal": {"Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "slither": {"metaphor": {}, "literal": {}},
    "pity": {"metaphor": {}, "literal": {}},
    "persist": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "mock": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "jam": {"metaphor": {}, "literal": {}},
    "assert": {
        "metaphor": {
            "Theme": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "conceal": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "itemize": {
        "metaphor": {},
        "literal": {"Topic": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "rent": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "lecture": {"metaphor": {}, "literal": {}},
    "appraise": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0}}},
    },
    "tread": {
        "metaphor": {
            "Theme": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {},
    },
    "converge": {"metaphor": {}, "literal": {}},
    "thrive": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "guard": {
        "metaphor": {},
        "literal": {
            "Beneficiary": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "afflict": {"metaphor": {}, "literal": {}},
    "accelerate": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "labour": {"metaphor": {}, "literal": {}},
    "dart": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "underline": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "volunteer": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "perpetuate": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "ruin": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "undo": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 0.5},
                wn.synset("clothing.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "scoop": {
        "metaphor": {
            "Destination": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "steepen": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "squeeze": {"metaphor": {}, "literal": {}},
    "dislike": {"metaphor": {}, "literal": {}},
    "disgrace": {"metaphor": {}, "literal": {}},
    "whisper": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Topic": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "scorch": {"metaphor": {}, "literal": {}},
    "fiddle": {"metaphor": {}, "literal": {}},
    "disintegrate": {
        "metaphor": {
            "Patient": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "evacuate": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "conquer": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Patient": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "exhaust": {"metaphor": {}, "literal": {}},
    "trap": {
        "metaphor": {},
        "literal": {
            "Destination": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("natural_phenomenon.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "top": {"metaphor": {}, "literal": {}},
    "clip": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "regain": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 2, "conf": 1.0}}
        },
    },
    "fold": {"metaphor": {}, "literal": {}},
    "drip": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "deflect": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "nourish": {"metaphor": {}, "literal": {}},
    "shape": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "assemble": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "punch": {
        "metaphor": {},
        "literal": {
            "Location": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Patient": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "shriek": {"metaphor": {}, "literal": {}},
    "commemorate": {
        "metaphor": {
            "Theme": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}}
        },
        "literal": {},
    },
    "straddle": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "meld": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "persevere": {"metaphor": {}, "literal": {}},
    "disentangle": {"metaphor": {}, "literal": {}},
    "manipulate": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("food.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "annoy": {"metaphor": {}, "literal": {}},
    "summon": {
        "metaphor": {
            "Patient": {
                wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "transport": {"metaphor": {}, "literal": {}},
    "blackmail": {"metaphor": {}, "literal": {}},
    "heighten": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}},
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "pin": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "pounce": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "raid": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Location": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "acquiesce": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "reassure": {"metaphor": {}, "literal": {}},
    "brush": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Patient": {wn.synset("physical_entity.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "throng": {
        "metaphor": {},
        "literal": {
            "Location": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "fulfill": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "bounce": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
            "Agent": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "stitch": {
        "metaphor": {
            "Product": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "alternate": {"metaphor": {}, "literal": {}},
    "reproduce": {"metaphor": {}, "literal": {}},
    "spark": {"metaphor": {}, "literal": {}},
    "interrogate": {"metaphor": {}, "literal": {}},
    "impress": {"metaphor": {}, "literal": {}},
    "install": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "constrict": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "clear": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "stem": {"metaphor": {}, "literal": {}},
    "tap": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
            "Patient": {wn.synset("person.n.01"): {"frequency": 2, "conf": 1.0}},
        },
    },
    "renew": {"metaphor": {}, "literal": {}},
    "elect": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.3333333333333333,
                },
                wn.synset("person.n.01"): {"frequency": 2, "conf": 0.6666666666666666},
            }
        },
    },
    "mop": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "scar": {"metaphor": {}, "literal": {}},
    "eradicate": {"metaphor": {}, "literal": {}},
    "speculate": {"metaphor": {}, "literal": {}},
    "apprehend": {"metaphor": {}, "literal": {}},
    "wince": {
        "metaphor": {},
        "literal": {
            "Stimulus": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "schedule": {"metaphor": {}, "literal": {}},
    "hinge": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 3, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "gamble": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "stagger": {"metaphor": {}, "literal": {}},
    "curl": {"metaphor": {}, "literal": {}},
    "resolve": {"metaphor": {}, "literal": {}},
    "burst": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "billow": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "nurture": {"metaphor": {}, "literal": {}},
    "rule": {
        "metaphor": {
            "Theme": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}}
        },
        "literal": {},
    },
    "dive": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "disgust": {"metaphor": {}, "literal": {}},
    "rejoice": {"metaphor": {}, "literal": {}},
    "splay": {"metaphor": {}, "literal": {}},
    "excite": {"metaphor": {}, "literal": {}},
    "transfer": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Theme": {
                wn.synset("abstraction.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {
            "Destination": {
                wn.synset("state.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("location.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Theme": {wn.synset("person.n.01"): {"frequency": 3, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "pray": {"metaphor": {}, "literal": {}},
    "broadcast": {"metaphor": {}, "literal": {}},
    "deprive": {"metaphor": {}, "literal": {}},
    "lounge": {"metaphor": {}, "literal": {}},
    "pile": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "warm": {"metaphor": {}, "literal": {}},
    "embrace": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {"Agent": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "restrain": {
        "metaphor": {
            "Patient": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "reside": {
        "metaphor": {
            "Location": {
                wn.synset("abstraction.n.06"): {
                    "frequency": 2,
                    "conf": 1.0,
                    "score": 0.6666666666666666,
                }
            }
        },
        "literal": {
            "Location": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "flash": {
        "metaphor": {
            "Theme": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "disguise": {
        "metaphor": {
            "Destination": {
                wn.synset("state.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "imitate": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "repay": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {},
    },
    "constrain": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "lunch": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "collide": {"metaphor": {}, "literal": {}},
    "roam": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("animal.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("person.n.01"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "chant": {
        "metaphor": {
            "Topic": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {},
    },
    "plough": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "empty": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "escort": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "lace": {
        "metaphor": {
            "Patient": {
                wn.synset("food.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "streamline": {
        "metaphor": {
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "strain": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "emphasise": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "laud": {
        "metaphor": {},
        "literal": {
            "Theme": {
                wn.synset("idea.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "prune": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "garb": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "screech": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "evade": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "claw": {
        "metaphor": {
            "Patient": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "niggle": {"metaphor": {}, "literal": {}},
    "detach": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "saunter": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "sweat": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "scatter": {
        "metaphor": {
            "Theme": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {},
    },
    "predate": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "sympathize": {"metaphor": {}, "literal": {}},
    "enhance": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "knowledge": {"metaphor": {}, "literal": {}},
    "boast": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Topic": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
        },
        "literal": {},
    },
    "concoct": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "snub": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "unwind": {"metaphor": {}, "literal": {}},
    "aggregate": {"metaphor": {}, "literal": {}},
    "accommodate": {"metaphor": {}, "literal": {}},
    "contact": {"metaphor": {}, "literal": {}},
    "soak": {
        "metaphor": {
            "Destination": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "translate": {"metaphor": {}, "literal": {}},
    "enrich": {"metaphor": {}, "literal": {}},
    "strew": {"metaphor": {}, "literal": {}},
    "mute": {
        "metaphor": {
            "Patient": {
                wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "back": {
        "metaphor": {
            "Agent": {
                wn.synset("substance.n.01"): {"frequency": 1, "conf": 0.5, "score": 1.0},
                wn.synset("organization.n.01"): {
                    "frequency": 1,
                    "conf": 0.5,
                    "score": 1.0,
                },
            },
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "loosen": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Patient": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
    "discourage": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "inspect": {
        "metaphor": {},
        "literal": {"Location": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "deflower": {"metaphor": {}, "literal": {}},
    "pester": {"metaphor": {}, "literal": {}},
    "subscribe": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "sever": {
        "metaphor": {
            "Agent": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "pump": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "redress": {
        "metaphor": {
            "Recipient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "grab": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "intimate": {
        "metaphor": {},
        "literal": {"Topic": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}}},
    },
    "dunk": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "streak": {"metaphor": {}, "literal": {}},
    "coil": {"metaphor": {}, "literal": {}},
    "explode": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "accrue": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "beware": {"metaphor": {}, "literal": {}},
    "taste": {
        "metaphor": {},
        "literal": {"Stimulus": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "jab": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("body_part.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
            },
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "impede": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "storm": {"metaphor": {}, "literal": {}},
    "interpolate": {"metaphor": {}, "literal": {}},
    "veer": {"metaphor": {}, "literal": {}},
    "dehydrate": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "recede": {
        "metaphor": {
            "Theme": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}}
        },
        "literal": {},
    },
    "inspire": {"metaphor": {}, "literal": {}},
    "beckon": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "liberate": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("region.n.03"): {"frequency": 1, "conf": 1.0}}},
    },
    "span": {"metaphor": {}, "literal": {}},
    "signify": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "crown": {
        "metaphor": {
            "Destination": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "swag": {
        "metaphor": {
            "Theme": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "hollow": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "resort": {
        "metaphor": {
            "Patient": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "cringe": {"metaphor": {}, "literal": {}},
    "class": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Attribute": {wn.synset("substance.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "batter": {"metaphor": {}, "literal": {}},
    "sheathe": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "lump": {"metaphor": {}, "literal": {}},
    "bump": {"metaphor": {}, "literal": {}},
    "echo": {"metaphor": {}, "literal": {}},
    "caress": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "thump": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "spare": {
        "metaphor": {
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "dispatch": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "invalidate": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("state.n.02"): {"frequency": 1, "conf": 1.0}}},
    },
    "abound": {"metaphor": {}, "literal": {}},
    "ban": {
        "metaphor": {
            "Theme": {
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "tolerate": {"metaphor": {}, "literal": {}},
    "engross": {"metaphor": {}, "literal": {}},
    "waken": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "disturb": {"metaphor": {}, "literal": {}},
    "cod": {"metaphor": {}, "literal": {}},
    "embed": {
        "metaphor": {},
        "literal": {
            "Destination": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "litter": {
        "metaphor": {},
        "literal": {
            "Destination": {wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "totter": {"metaphor": {}, "literal": {}},
    "crackle": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("sound.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "preclude": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("communication.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "sink": {
        "metaphor": {
            "Patient": {
                wn.synset("physical_entity.n.01"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            }
        },
        "literal": {},
    },
    "pat": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "faze": {"metaphor": {}, "literal": {}},
    "trust": {"metaphor": {}, "literal": {}},
    "fancy": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "privatize": {"metaphor": {}, "literal": {}},
    "clothe": {
        "metaphor": {
            "Agent": {
                wn.synset("organization.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "inhale": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "cherish": {"metaphor": {}, "literal": {}},
    "colour": {"metaphor": {}, "literal": {}},
    "line": {"metaphor": {}, "literal": {}},
    "squat": {
        "metaphor": {
            "Agent": {wn.synset("plant.n.02"): {"frequency": 1, "conf": 1.0, "score": 1.0}}
        },
        "literal": {},
    },
    "delineate": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}}
        },
    },
    "harmonize": {
        "metaphor": {},
        "literal": {
            "Patient": {
                wn.synset("communication.n.02"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "correlate": {
        "metaphor": {},
        "literal": {"Patient": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "ship": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Destination": {wn.synset("body_part.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "rub": {
        "metaphor": {},
        "literal": {
            "Destination": {wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "cure": {"metaphor": {}, "literal": {}},
    "trade": {"metaphor": {}, "literal": {}},
    "anger": {"metaphor": {}, "literal": {}},
    "glimpse": {
        "metaphor": {},
        "literal": {
            "Stimulus": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 0.5},
                wn.synset("abstraction.n.06"): {"frequency": 1, "conf": 0.5},
            }
        },
    },
    "drone": {"metaphor": {}, "literal": {}},
    "flank": {"metaphor": {}, "literal": {}},
    "detail": {"metaphor": {}, "literal": {}},
    "deduce": {
        "metaphor": {},
        "literal": {
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("state.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "bask": {"metaphor": {}, "literal": {}},
    "bias": {"metaphor": {}, "literal": {}},
    "camp": {
        "metaphor": {},
        "literal": {
            "Location": {wn.synset("location.n.01"): {"frequency": 1, "conf": 1.0}},
            "Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "alarm": {"metaphor": {}, "literal": {}},
    "counter": {
        "metaphor": {},
        "literal": {
            "Theme": {wn.synset("communication.n.02"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "pocket": {
        "metaphor": {
            "Theme": {wn.synset("idea.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}}
        },
        "literal": {},
    },
    "thrash": {
        "metaphor": {},
        "literal": {
            "Patient": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
            "Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}},
        },
    },
    "discriminate": {
        "metaphor": {},
        "literal": {"Theme": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "lurk": {"metaphor": {}, "literal": {}},
    "edge": {
        "metaphor": {
            "Theme": {
                wn.synset("artifact.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            }
        },
        "literal": {},
    },
    "utter": {
        "metaphor": {
            "Topic": {
                wn.synset("sound.n.04"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 0.5}
            },
        },
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "drawl": {"metaphor": {}, "literal": {}},
    "disagree": {
        "metaphor": {},
        "literal": {"Agent": {wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0}}},
    },
    "fondle": {"metaphor": {}, "literal": {}},
    "gauge": {
        "metaphor": {
            "Theme": {
                wn.synset("communication.n.02"): {
                    "frequency": 1,
                    "conf": 1.0,
                    "score": 1.0,
                }
            },
            "Agent": {
                wn.synset("person.n.01"): {"frequency": 1, "conf": 1.0, "score": 1.0}
            },
        },
        "literal": {},
    },
}
