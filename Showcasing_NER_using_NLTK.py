# Chunking except for somethin = Chinking

import re
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


"""
POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when


NE Type and Examples

ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian
"""

INDEX = 1


def preprocess_sentence(tokenized, index):
    tagged = nltk.pos_tag(nltk.word_tokenize(tokenized[index]))

    # To not show classify as named entity type and simply classify to NE
    named_entity_chunked = nltk.ne_chunk(tagged, binary=True)

    return named_entity_chunked


train_text = state_union.raw("2005-GWBush.txt")

tokenizer = PunktSentenceTokenizer(train_text)

sample_text = state_union.raw("2006-GWBush.txt")

tokenized_with_train = tokenizer.tokenize(sample_text)

with_train = preprocess_sentence(tokenized_with_train, INDEX)
with_train.draw()
