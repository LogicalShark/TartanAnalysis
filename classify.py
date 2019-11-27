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
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
import graphviz


class CustomNgramTfidfVectorizer(TfidfVectorizer):
    def _word_ngrams(self, tokens, stop_words=None):
        # First get tokens without stop words
        tokens = super(TfidfVectorizer, self)._word_ngrams(tokens, None)
        if stop_words is not None or self.stop_ngrams is not None:
            new_tokens = []
            for token in tokens:
                if token in self.stop_ngrams:
                    continue
                good = True
                for s in token.split(' '):
                    if s in stop_words:
                        good = False
                        break
                if good:
                    new_tokens.append(token)
            return new_tokens
        return tokens


def processPapers():
    print("Loading files...")
    papers = load_files("decades/")
    X, y, z = papers.data, papers.target, papers.target_names
    # try:
    #     vecf = open("tfidf", "rb")
    #     tfidf = pickle.load(vecf)
    # except:
    # e misread as o or c, a often misread as o
    # remove authors' names, colleges that didn't exist early
    # "continued on page," "www," etc. are not substantive content
    # stopw = ["bo", "tho", "ho", "arc", "cmu", "richard", "tom", "mike", "michael", "thomas", "david", "joe", "cit", "cfa",
    #             "ot", "rst", "th", "im", "inc", "ing", "skibo", "applied", "pillbox", "www",
    #             "org", "staffwriter", "andrew",  "pm", "bell", "com", "thetartan", "arts",]
    stopw = stopwords.words('english')
    stopbg = []  # ["carnegie mellon", "institute technology", "student senate", "student council","university center"
    # "fine arts", "continued page", "mellon university","carnegie tech", "carnegie institute"]
    # remove proper nouns and numbers from classifier
    pattern = r'(?u)\b[a-z][\w-]*\b'

    print("Creating vectorizer...")
    tfidf = CustomNgramTfidfVectorizer(max_features=50000, sublinear_tf=True, min_df=0.01, max_df=0.7,
                                       ngram_range=(1, 2), stop_words=stopw, token_pattern=pattern)
    tfidf.stop_ngrams = stopbg
    features = tfidf.fit_transform(X).toarray()
    f = open("features", "wb+")
    pickle.dump(features, f)
    f.close()
    vecf = open("tfidf", "wb+")
    pickle.dump(tfidf, vecf)
    vecf.close()
    return tfidf, features, X, y, z


def newlinesep(text):
    return text.split("\n")


def processSyntax():
    print("Loading files...")
    papers = load_files("syntax/")
    print("Creating vectorizer...")
    X, y, z = papers.data, papers.target, papers.target_names
    tfidf = TfidfVectorizer(max_features=10000, sublinear_tf=True, min_df=0.01, max_df=0.8, tokenizer=newlinesep,
                            ngram_range=(1, 2), stop_words=[])
    features = tfidf.fit_transform(X).toarray()
    f = open("features_syntax", "wb+")
    pickle.dump(features, f)
    f.close()
    vecf = open("tfidf_syntax", "wb+")
    pickle.dump(tfidf, vecf)
    vecf.close()
    features = tfidf.fit_transform(X).toarray()
    f = open("features_syntax", "wb+")
    pickle.dump(features, f)
    f.close()
    return tfidf, features, X, y, z


def checkAccuracy(tfidf, features, y, syntax=False):
    print("Creating classifier...")
    # classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier = DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=0)
    classifier.fit(X_train, y_train)
    f = open("classifier" + ("_syntax" if syntax else ""), "wb+")
    pickle.dump(classifier, f)
    f.close()
    y_pred = classifier.predict(X_test)
    # print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    export_graphviz(classifier, max_depth=8, out_file="tree.dot",
                    feature_names=tfidf.get_feature_names())


def getCorrelations(features, tfidf, X, y, z):
    N = 10
    features_chi2 = chi2(features, y)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("Most correlated unigrams: {}".format(
        '\n'.join(reversed(unigrams[-N:]))))
    print("Most correlated bigrams: {}".format(
        '\n'.join(reversed(bigrams[-N:]))))


if __name__ == "__main__":
    tfidf, features, X, y, z = processPapers()
    checkAccuracy(tfidf, features, y, False)
    getCorrelations(features, tfidf, X, y, z)
