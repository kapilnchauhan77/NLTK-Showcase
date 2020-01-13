import os
import nltk
import pickle
import random
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews


stop_words = stopwords.words("english")

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# print(" ".join(document[11][0]))
# print("\n" + "".join(document[11][1]))

all_words = [word.lower() for word in movie_reviews.words()]
# print(all_words)
# sns.countplot(all_words)
# plt.show()

all_words = nltk.FreqDist(all_words)
# print(dict(all_words.most_common(10)))
# print(all_words["Stupid"])
# plt.bar(list(dict(all_words.most_common(10)).keys()),
#         list(dict(all_words.most_common(10)).values()))
# plt.show()

word_features = list(filter(lambda x: x not in stop_words, list(all_words.keys())))[:3000]
# word_features = list(all_words.keys())[:3000]
# word_features_count = [all_words[feature] for feature in word_features]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# print(find_features(movie_reviews.words("neg/cv000_29416.txt")))

feature_set = [(find_features(words), category) for (words, category) in documents]

# print(len(feature_set))

training_set = feature_set[:int(len(feature_set)*0.9)]
testing_set = feature_set[int(len(feature_set)*0.9):]

if os.path.exists("nltk_Naive_Bayes_Classifier.pickle"):
    print("Loading Classifier...")
    with open("nltk_Naive_Bayes_Classifier.pickle", "rb") as f:
        classifier = pickle.load(f)
else:
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Saving Classifier...")
    with open("nltk_Naive_Bayes_Classifier.pickle", "wb") as f:
        pickle.dump(classifier, f)

print(f"Accuracy: {nltk.classify.accuracy(classifier, testing_set)}")

classifier.show_most_informative_features(15)
