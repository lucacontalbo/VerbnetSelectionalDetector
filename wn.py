import nltk
nltk.download("wordnet")

from nltk.corpus import wordnet as wn

syns = wn.synsets("scalar")

#print(syns[0])
#print(dir(syns[0]))
#print(syns[0].lemmas())
#print(dir(syns[0].lemmas()[0]))
#print(syns[0].lemmas()[0].frame_strings())
for el in syns:
	print(el.hypernym_paths())
