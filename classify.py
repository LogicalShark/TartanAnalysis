import sys
import re
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

wordgroups = [["online", "internet", "future", "2010"],
              ["war", "1900", "Truman"]]
colors = ["black"]
sizes = [10]
colors = ["red", "turquoise", "orange", "forestgreen", "purple", "teal",
          "navy",  "slategrey", "olive", "maroon", "peru", "orangered", "crimson"]


def plot2D(model, default=True):
    vocab = {}
    # for v in model.wv.vocab.keys():
    #     if any(v in g for g in wordgroups):
    #         vocab[v] = model.wv.vocab[v]
    # X = model[vocab]
    pca = PCA(n_components=2) if default else PCA(n_components=3)
    result = pca.fit_transform(model)
    words = list(vocab)
    pyplot.scatter(result[:, 0], result[:, 1]) if default else pyplot.scatter(
        result[:, 1], result[:, 2])
    for g, group in enumerate(wordgroups):
        for word in group:
            i = model.index(word)
            pyplot.annotate(
                word, xy=(result[i, 0], result[i, 1]), color=colors[g] if g < len(colors) else "black", fontsize=8 if g < len(sizes) else 10) if default else pyplot.annotate(
                    word, xy=(result[i, 1], result[i, 2]), color=colors[g] if g < len(colors) else "black", fontsize=8 if g < len(sizes) else 10)
    pyplot.show()


def plot3D(model, default=True):
    vocab = {}
    pca = PCA(n_components=3) if default else PCA(n_components=4)
    result = pca.fit_transform(model)
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result[:, 0], result[:, 1], result[:, 2]) if default else ax.scatter(
        result[:, 1], result[:, 2], result[:, 3])
    for g, group in enumerate(wordgroups):
        for word in group:
            i = model.index(word)
            ax.text(result[i, 0], result[i, 1], result[i, 2], word, None, color=colors[g] if g < len(
                colors) else "black", fontsize=8 if g < len(sizes) else 10) if default else ax.text(result[i, 1], result[i, 2], result[i, 3], word, None, color=colors[g] if g < len(
                    colors) else "black", fontsize=8 if g < len(sizes) else 10)
    pyplot.show()


def processPapers():
    papers = load_files("processed/")
    X, y, z = papers.data, papers.target, papers.target_names
    # print(X[0], y[0], z)
    try:
        vecf = open("tfidf", "rb")
        tfidfconverter = pickle.load(vecf)
    except:
        # vectorizer = CountVectorizer(
        #     max_features=1000, min_df=0.01, max_df=0.7, stop_words=stopwords.words('english'))
        # X = vectorizer.fit_transform(papers).toarray()
        tfidfconverter = TfidfVectorizer(max_features=1000, sublinear_tf=True, min_df=0.01, max_df=0.7,
                                        stop_words=stopwords.words('english'))  # , ngram_range=(1, 2))
        tfidfconverter = TfidfTransformer()
        features = tfidfconverter.fit_transform(X).toarray()
        f = open("features", "wb+")
        pickle.dump(features, f)
        f.close()
        vecf = open("tfidf", "wb+")
        pickle.dump(tfidfconverter, vecf)
    vecf.close()
    try:
        f = open("features", "rb")
        features = pickle.load(f)
    except:
        features = tfidfconverter.fit_transform(X).toarray()
        f = open("features", "wb+")
        pickle.dump(features, f)
    f.close()
    return tfidfconverter, features, y, z


def newlinesep(text):
    return text.split("\n")


def processSyntax():
    papers = load_files("syntax/")
    X, y, z = papers.data, papers.target, papers.target_names
    try:
        vecf = open("tfidf_syntax", "rb")
        tfidfconverter = pickle.load(vecf)
    except:
        # vectorizer = CountVectorizer(
        #     max_features=1000, min_df=0.01, max_df=0.7, stop_words=stopwords.words('english'))
        # X = vectorizer.fit_transform(papers).toarray()
        tfidfconverter = TfidfVectorizer(max_features=500, sublinear_tf=True, min_df=0.01, max_df=0.9, tokenizer=newlinesep,
                                        ngram_range=(1,2), stop_words=[])
        # tfidfconverter = TfidfTransformer()
        print(len(X), len(y), len(z))
        features = tfidfconverter.fit_transform(X).toarray()
        f = open("features_syntax", "wb+")
        pickle.dump(features, f)
        f.close()
        vecf = open("tfidf_syntax", "wb+")
        pickle.dump(tfidfconverter, vecf)
    vecf.close()
    try:
        f = open("features_syntax", "rb")
        features = pickle.load(f)
    except:
        features = tfidfconverter.fit_transform(X).toarray()
        f = open("features_syntax", "wb+")
        pickle.dump(features, f)
    f.close()
    return tfidfconverter, features, y, z


def checkAccuracy(features, y, syntax = False):
    try:
        f = open("classifier" + ("_syntax" if syntax else ""), "rb")
        classifier = pickle.load(f)
    except:
        classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
        f = open("classifier" + ("_syntax" if syntax else ""), "wb+")
        pickle.dump(classifier, f)
    f.close()
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


def getCorrelations(features, tfidfconverter, y, z):
    N = 10
    for category_id in set(sorted(y)):
        features_chi2 = chi2(features, y)
        print("# '{}':".format(category_id))
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidfconverter.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("Most correlated unigrams: {}".format(
            '\n'.join(reversed(unigrams[-N:]))))
        print("Most correlated bigrams: {}".format(
            '\n'.join(reversed(bigrams[-N:]))))


if __name__ == "__main__":
    tfidfconverter, features, y, z = processSyntax()
    checkAccuracy(features, y, True)
    # print(features)
    # getCorrelations(features, tfidfconverter, y, z)
    # print(y)
    # plot2D(features)
