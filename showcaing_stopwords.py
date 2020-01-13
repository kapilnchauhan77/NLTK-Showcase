from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

sentence = "Hi this is me Mr. Kapil.N.Chauhan showcasing nltk, stopwords to myself."

stop_words = stopwords.words("english")

filtered_sentence = " ".join(
    list(filter(lambda x: x not in stop_words, word_tokenize(sentence))))

ommited_words = list(filter(lambda x: x in stop_words, word_tokenize(sentence)))

print("Sentence:")
print(sentence)
print("\nFiltered Sentence:")
print(filtered_sentence)
print("\nOmmited Words:")
print(ommited_words)
print("\nStop Words:")
print(stop_words)
