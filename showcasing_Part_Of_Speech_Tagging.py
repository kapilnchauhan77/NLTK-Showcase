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
"""

INDEX = 1


def preprocess_sentence(tokenized, index):
    return(nltk.pos_tag(nltk.word_tokenize(tokenized[index])))


# Without training PunktSentenceTokenizer

tokenizer = PunktSentenceTokenizer()

sample_text = state_union.raw("2006-GWBush.txt")

tokenized_without_train = tokenizer.tokenize(sample_text)

print("Without Training:")

without_train = preprocess_sentence(tokenized_without_train, INDEX)
print(without_train)


# With training PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")

tokenizer = PunktSentenceTokenizer(train_text)

sample_text = state_union.raw("2006-GWBush.txt")

tokenized_with_train = tokenizer.tokenize(sample_text)

print("\nWith Training:")

with_train = preprocess_sentence(tokenized_with_train, INDEX)
print(with_train)

print(f"\nResult after and before training is same for index: {INDEX}"
      if with_train == without_train
      else f"\nResult after and before training is not same for index: {INDEX}")

if with_train != without_train:

    print(f"\nTokenized Sentence without training at index-\
{INDEX}: \n{tokenized_without_train[INDEX]}")

    print(f"\nTokenized Sentence with training at index-\
{INDEX}: \n{tokenized_with_train[INDEX]}")

else:
    print(f"\nTokenized Sentence at index-\
{INDEX}: \n{tokenized_without_train[INDEX]}")
