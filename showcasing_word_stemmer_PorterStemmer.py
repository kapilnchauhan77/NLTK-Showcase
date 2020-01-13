from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

sentences = "Hi this is me Mr. Kapil.N.Chauhan showcasing nltk, \
PorterStemmer to myself.\n\
I am a pythoneer just pythoning my way pythonly to the python heaven."

stemmed_sentences = " ".join([ps.stem(word) for word in word_tokenize(sentences)])

print("Sentence:")
print(sentences)
print("\nStemmed Sentence:")
print(stemmed_sentences)
