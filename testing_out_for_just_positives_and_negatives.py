import os
import nltk
import pickle
import random
import seaborn as sns
from statistics import mode
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.classify import ClassifierI
from nltk.corpus import movie_reviews
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        try:
            return mode(votes)
        except:
            Counter(votes).most_common(1)[0][0]

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        try:
            choice_votes = votes.count(mode(votes))
        except:
            choice_votes = votes.count(Counter(votes).most_common(1)[0][0])
        conf = choice_votes / len(votes)
        return conf


stop_words = stopwords.words("english")

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

all_words = [word.lower() for word in movie_reviews.words()]

all_words = nltk.FreqDist(all_words)

word_features = list(filter(lambda x: x not in stop_words, list(all_words.keys())))[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


feature_set = [(find_features(words), category) for (words, category) in documents]

# Only for Negatives
# training_set = feature_set[int(len(feature_set)*0.05):int(len(feature_set)*0.95)]
# testing_set = feature_set[int(len(feature_set)*0.95):]

# Only for Positives
training_set = feature_set[int(len(feature_set)*0.05):int(len(feature_set)*0.95)]
testing_set = feature_set[:int(len(feature_set)*0.05)]

classifiers = ["SVC", "LinearSVC", "NuSVC", "LogisticRegression",
               "SGDClassifier", "MultinomialNB", "BernoulliNB",
               "RandomForestClassifier"]
clfs = {}

for classifier in classifiers:
    clfs[str(classifier)] = SklearnClassifier(eval(classifier)())
    clfs[str(classifier)].train(training_set)
    print(f"{str(classifier)} accuracy percent:",
          (nltk.classify.accuracy(clfs[str(classifier)], testing_set))*100)

voted_classifier = VoteClassifier(
    *[clfs[str(classifier)] for classifier in classifiers])

print("voted_classifier accuracy percent:",
      (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:", voted_classifier.classify(
    testing_set[0][0]), "Confidence %:",
    voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(
    testing_set[1][0]), "Confidence %:",
    voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(
    testing_set[2][0]), "Confidence %:",
    voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(
    testing_set[3][0]), "Confidence %:",
    voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(
    testing_set[4][0]), "Confidence %:",
    voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(
    testing_set[5][0]), "Confidence %:",
    voted_classifier.confidence(testing_set[5][0])*100)
