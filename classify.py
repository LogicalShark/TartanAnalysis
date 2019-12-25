import os
import sys
import re
import pickle
import math
import random
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
import gc
import matplotlib.pyplot as plt
import codecs
folder = "decades_modified"
stopw = stopwords.words('english')
stopbg = []
# e misread as o or c, a often misread as o
# remove authors' names, colleges that didn't exist early
# "continued on page, " "www, " etc. are not substantive content
stopw += ["ment",  "boon",  "ence", "tion", "tions", "dents", "inter", "tlio", "doin", "dent", "ments", "ment", "pression", "fore", "ture", "sion",  "burg", "tholr", "pany", "utes", "ings", "ness", "plied", "tory", "tary", "moro",  "groat", "lias", "dally", "coun", "sary", "wont", "lowing", "wore", "adver", "nlng",
          "prac", "tills", "nore", "litde", "largo", "ally", "divi", "tane", "iving"] + ["scitech", "thetartan", "condense", "communications", "sponsored", "resources", "promptly", "holiday", "verification", "postage", "may", "june", "necessarily", "endorsed", "members", "edit", "opinions", "burgh", "supply", "construed", "genuine", "july", "mens", "feb", "march", "mans", "editorial", "artatt", "ariatt", "artatt", "ange", "forf", "less", "tention", "disp", "igan", "homo", "whore", "andrew", "rery", "nally", "clock", "page", "senate", "address", "cial", "scor", "herewith", "oast", "frosh", "firstyear", "typo", "aver", "ation", "ance", "chool", "cation", "feated", "cording", "eing", "rost", "welldefined", "pillbox", "horoscopes", "ination", "clusive", "tration", "offi", "tion", "tional", "plac", "direc", "larian", "youl", "rime", "erfect", "tant", "ouly", "ager", "lute", "ject", "jects", "ize", "ization", "schoo", "ferent", "anti", "tlion", "atop", "bein", "ular", "ably", "eous", "miliar", "irst", "negie", "lanan", "dren", "welt", "cent", "cently", "ited", "lation", "volved", "nome", "erne", "berg", "tice", "pers", "ries", "slon", "taff", "ning", "whic", "tits", "bove", "thor", "ond", "onds", "dale", "cons", "bout", "tives", "busi", "firstyear", "tures", "ceive", "tirely", "lated", "ternity", "megie", "mare", "letics", "rled", "tlon", "tIon", "tured", "sentative", "hlch", "thet", "ling", "pleted", "announc", "terial", "turer", "ture", "tirst", "nancial", "nity", "moil", "ience", "nitely", "vide", "stitute", "avor", "coore", "wrell", "ucation", "tage", "quires", "twentyfive", "produc", "nology", "tudy", "threefourths", "thier", "ting"]
#"eral","tive","ulty","stu","tain", "repre","ovor","sity","negie","forthe",
#"aff", "ful", "tne", "oft", "til", "ity", "tle", "fel", "bo", "tho", "ho", "ne", "arc", "ot", "rst", "aro", "im", "inc", "ing", "hg", "ber", "ter", "der", "fty", "ent", "mem", "bur", "son", "tbs", "mil", "num", "fom", "yer", "fot", "fol", "cel", "thi", "sch", "ill", "eral", "sup", "ers", "suc", "pro", "ght", "ono", "col", "nes", "pre", "dis", "ton", "ani", "ust", "th", "tbe", "con", "ble","bers","lll","partment","consid","tlie",
#"org", "www", "edu", "com", "wa", "vb",
#"mailing special", "right edit", "see page", "resources supply", "continued page", "examination periods", "year except",
#"second class", "class matter", "select prices", "largest stock", "stock hand",  "except holidays", "help call",
#"official opinion", "get acquainted", "standing man", "domestic manufacture", "rights reserved", "place daily",
# "letter intended", "information call", "sanitary machine", "print whatever", "perspiration odor", "usually members"m
# "opinion section", "bedroom suites", "certain looking", "national advertising", "political cartoon",


class CustomNgramTfidfVectorizer(TfidfVectorizer):
    def _word_ngrams(self, tokens, stop_words=None):
        # First get tokens without stop words
        tokens = super(TfidfVectorizer, self)._word_ngrams(tokens, None)
        if stop_words is not None or self.stop_ngrams is not None:
            new_tokens = []
            for token in tokens:
                if token.lower() in self.stop_ngrams:
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
    papers = load_files(folder+"/")
    X, y, z = papers.data, papers.target, papers.target_names
    # remove proper nouns, numbers, contractions from classifier
    pattern = r'(?u)\b[a-z][a-z\-]+[a-z][a-z]\b'
    # pattern = r'\n.*\n'
    print("Creating vectorizer...")
    tfidf = CustomNgramTfidfVectorizer(max_features=500000, sublinear_tf=True, min_df=8, max_df=0.75,
                                       stop_words=stopw, lowercase=False, token_pattern=pattern)  # , ngram_range = (1, 2))
    tfidf.stop_ngrams = stopbg
    gc.collect()
    features = tfidf.fit_transform(X)
    return papers, tfidf, features, X, y, z


def newlinesep(text):
    return text.split("\n")


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
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    export_graphviz(classifier, out_file=folder+"_tree.dot",
                    feature_names=tfidf.get_feature_names())


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:

        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def getCorrelations(tfidf, features):
    feature_names = np.array(tfidf.get_feature_names())
    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(features.tocoo())

    # extract only the top n; n here is 10
    keywords = extract_topn_from_vector(feature_names, sorted_items, 5)

    print("===Keywords===")
    for k in keywords:
        print(k, keywords[k])


def freqAnalysis(words):
    # For counting the total words in each document
    # count = [0 for d in range(30020)]
    # t = open("total.txt", 'w+')
    out = open("out.txt", 'w+')
    for word in words:
        out.write(word+"	")
    freqs = [[0 for decade in range(1900, 2020, 10)]
             for i in range(len(words))]
    for decade in range(1900, 2020, 10):
        directory = os.fsencode(folder+"/"+str(decade))
        # docs = 0
        # total = 0
        # for file in os.listdir(directory):
        #     filename = os.fsdecode(file)
        #     readf = codecs.open(folder+"/"+str(decade)+"/"+filename, 'r',
        #                         encoding="cp1252", errors="replace")
        #     filedata = readf.read()
        #     total += sum(1 for _ in re.finditer(
        #         r'\b[a-z][a-z][a-z\-]+[a-z]\b', filedata))
        #     for w in stopw:
        #         total -= sum(1 for _ in re.finditer(r'\b%s\b' %
        #                                             re.escape(w), filedata))
        #     docs += 1
        #     readf.close()
        # t.write(str(total)+"\t")
        total = [43005, 45932, 134411, 88306, 58563, 89185, 90242,
                 165864, 282737, 440705, 410157, 349998][(decade-1900)//10]
        for i, word in enumerate(words):
            count = 0
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                readf = codecs.open(folder+"/"+str(decade)+"/"+filename, 'r',
                                    encoding="cp1252", errors="replace")
                filedata = readf.read()
                readf.close()
                count += sum(1 for _ in re.finditer(r'\b%s\b' %
                                                    re.escape(word), filedata))
            freqs[i][(decade-1900)//10] = count/total
    for decade in range(1900, 2020, 10):
        out.write("\n")
        for i in range(len(words)):
            x = freqs[i][(decade-1900)//10]
            out.write(str(1000000*x)+"	")
    out.close()
    # t.close()

if __name__ == "__main__":
    papers, tfidf, features, X, y, z = processPapers()
    checkAccuracy(tfidf, features, y, False)
    words = ["car","airplane","plane","train","bus"]
    freqAnalysis(words)

    # Getting keywords from each decade
       
    # vectorizers = []
    # weights = [0 for n in range(12)]
    # pattern = r'(?u)\b[a-z][a-z\-]+[a-z][a-z]\b'
    # for decade in range(1900, 2020, 10):
    #     decadedata = []
    #     directory = os.fsencode(folder+"/"+str(decade))
    #     for file in os.listdir(directory):
    #         filename = os.fsdecode(file)
    #         readf = codecs.open(folder+"/"+str(decade)+"/"+filename, 'r',
    #                             encoding="cp1252", errors="replace")
    #         decadedata.append(readf.read())
    #         readf.close()
    #     tfidf = CustomNgramTfidfVectorizer(
    #         max_features=100000, sublinear_tf=True, stop_words=stopw, lowercase=False, token_pattern=pattern)
    #     tfidf.stop_ngrams = []
    #     gc.collect()
    #     features = tfidf.fit_transform(decadedata)
    #     vectorizers.append([tfidf, features])

    # # weights = [sum(vectorizers[n][0].vocabulary_.values())/1000 for n in range(12)]
    # # for w in range(10000):
    # #     i = 0
    # #     word = random.choice(list(vectorizers[random.randint(0,11)][0].vocabulary_.keys()))
    # #     for v in vectorizers:
    # #         if word in v[0].vocabulary_:
    # #             weights[i] += v[0].vocabulary_[word]
    # #         else:
    # #             weights[i] += 0
    # #         i += 1
    # # weights = [x/10000 for x in weights]
    # # print(weights)

    # # Average of several trials
    # weights = [9925, 10875, 23800, 16465, 10000, 8065,
    #            10660,	14880,	22950,	26150,	19440,	11820]
    # d = 1900
    # # for x in vectorizers:
    # #     print("\n"+str(d))
    # #     getCorrelations(x[0], x[1])
    # #     d += 10
    # out = open("out.txt", 'w+')
    # for word in words:
    #     out.write(word+"	")
    # out.write("\n")
    # i = 0
    # prev = 0;
    # for v in vectorizers:
    #     for word in words:
    #         if word in v[0].vocabulary_:
    #             out.write(str(math.log(v[0].vocabulary_[word]/prev))+"""	""")
    #             prev = v[0].vocabulary_[word]
    #         else:
    #             out.write(str(0)+"""	""")
    #             prev = 1
    #     i += 1
    #     out.write("\n")
