from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
import os
from nltk import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.phrases import Phrases, Phraser
from gensim.scripts.glove2word2vec import glove2word2vec

def makeModel(n):
    sentences = []
    for decade in range(1900, 2010, 10):
        try:
            for file in os.listdir("decades/"+str(decade)):
                filename = os.fsdecode(file)
                f = open("decades/" + str(decade) + "/" + filename, "r")
                txt = f.read()
                sentences += [word_tokenize(s) for s in sent_tokenize(txt)]
                f.close()
        except:
            continue
        if n == 1:
            model = Word2Vec(sentences, min_count=1)
            model.save("wv/model.bin")
        if n == 2:
            bigram = Phrases(sentences, max_vocab_size=50000)
            bigram_phraser = Phraser(bigram)
            model2 = Word2Vec(bigram_phraser[sentences], min_count=1)
            model2.save("wv/model.bin")


def mostSim(model):
    pos = input("Enter positives separated by ','\n").split(",")
    if len(pos) == 0:
        print("Error: empty positive\n")
        mostSim(model)
    neg = input("Enter negatives separated by ','\n").split(",")
    try:
        print([x[0] for x in model.wv.most_similar_cosmul(positive=pos, negative=neg)])
    except KeyError:
        print("Error: word not in vocabulary or incorrect delimiter between words\n")
    mostSim(model)


def approxLinear(model):
    f = open("wv/words.txt", "r", encoding="utf-8")
    words = []
    for group in f.read().split("\n"):
        words += group.split(",")
        if "" in words:
            words.remove("")
    f.close()
    outputs = []
    for i, first in enumerate(words):
        for j, second in enumerate(words[i:]):
            for third in words:
                for result in model.most_similar_cosmul(positive=[first, second], negative=[third], topn=1):
                    outputs.append(
                        (first, second, third, result[0], result[1]))
        outputs.sort(key=lambda x: x[4])
        outputs = outputs[:10]
    for (first, second, third, out, sim) in outputs:
        print(first + "\t+\t" + second + "\t-\t" +
              third + "\t=\t" + out + "\t\tsimilarity=" + str(sim))


# Color of labels in each group (dark colors recommended, defaults to black)
# Hex or https://matplotlib.org/gallery/color/named_colors.html
colors = ["red", "turquoise", "orange", "forestgreen", "purple", "teal",
          "navy",  "slategrey", "olive", "maroon", "peru", "orangered", "crimson"]

sizes = [10]

def plot2D(model, wordgroups, default=True):
    vocab = {}
    for v in model.wv.vocab.keys():
        if any(v in g for g in wordgroups):
            vocab[v] = model.wv.vocab[v]
    X = model[vocab]
    pca = PCA(n_components=2) if default else PCA(n_components=3)
    result = pca.fit_transform(X)
    words = list(vocab)
    pyplot.scatter(result[:, 0], result[:, 1]) if default else pyplot.scatter(
        result[:, 1], result[:, 2])
    for g, group in enumerate(wordgroups):
        for word in group:
            if not word in words:
                continue
            i = words.index(word)
            pyplot.annotate(
                word, xy=(result[i, 0], result[i, 1]), color=colors[g] if g < len(colors) else "black", fontsize=8 if g < len(sizes) else 10) if default else pyplot.annotate(
                    word, xy=(result[i, 1], result[i, 2]), color=colors[g] if g < len(colors) else "black", fontsize=8 if g < len(sizes) else 10)
    pyplot.show()


def plot3D(model, wordgroups, default=True):
    vocab = {}
    for v in model.wv.vocab.keys():
        if any(v in g for g in wordgroups):
            vocab[v] = model.wv.vocab[v]
    X = model[vocab]
    pca = PCA(n_components=3) if default else PCA(n_components=4)
    result = pca.fit_transform(X)
    words = list(vocab)
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result[:, 0], result[:, 1], result[:, 2]) if default else ax.scatter(
        result[:, 1], result[:, 2], result[:, 3])
    for g, group in enumerate(wordgroups):
        for word in group:
            if not word in words:
                continue
            i = words.index(word)
            ax.text(result[i, 0], result[i, 1], result[i, 2], word, None, color=colors[g] if g < len(
                colors) else "black", fontsize=8 if g < len(sizes) else 10) if default else ax.text(result[i, 1], result[i, 2], result[i, 3], word, None, color=colors[g] if g < len(
                    colors) else "black", fontsize=8 if g < len(sizes) else 10)
    pyplot.show()


def plotVecs(dim, model, default=True):
    f = open("wv/words.txt", "r", encoding="utf-8")
    groups = [g.split(",") for g in f.read().split("\n")]
    f.close()
    if dim == 2:
        plot2D(model, groups, default)
    else:
        plot3D(model, groups, default)


if __name__ == "__main__":
    # makeModel(1)
    model = Word2Vec.load("wv/model.bin")
    # TODO: clustering to create groups automatically?
    # plotVecs(3, model)
    # approxLinear(model)
    mostSim(model)
