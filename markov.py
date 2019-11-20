import time
import random
import math
import os
import codecs
import pickle
import re
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
import gib_detect_train

model_data = pickle.load(open('gib_model.pki', 'rb'))
model_mat = model_data['mat']
threshold = model_data['thresh']
d = TreebankWordDetokenizer()

table = {}
size = 0
order = 5
year = 1950


def addToTable(text, order):
    global table, size
    tokens = word_tokenize(text)
    # Make the index table
    for i in range(len(tokens) - order):
        sub = d.detokenize(tokens[i:i+order])
        if gib_detect_train.avg_transition_prob(sub, model_mat) < threshold:
            continue
        if not sub in table:
            table[sub] = {}
            table[sub]["SIZE"] = 0
            size += 1
    # Count the following strings for each string
    for j in range(len(tokens) - order - order):
        index = d.detokenize(tokens[j:j+order])
        if gib_detect_train.avg_transition_prob(index, model_mat) < threshold:
            continue
        k = j + order
        following = d.detokenize(tokens[k:k+order])
        if gib_detect_train.avg_transition_prob(following, model_mat) < threshold:
            continue
        if not following in table[index] and len(following) > 0:
            table[index][following] = 1
            table[index]["SIZE"] += 1
        elif len(following) > 0:
            table[index][following] += 1
            table[index]["SIZE"] += 1


def createText(start, length, table, order, size):
    keys = list(table.keys())
    chars = start
    if len(start) == 0 or not start in table:
        chars = createNextChars(
            table[random.choice(keys)])

    output = chars
    for k in range(length//order):
        newchars = createNextChars(table[chars])
        if (not newchars is None and len(newchars) > 0):
            chars = newchars
            if not re.match(r'^\s*[.!?,\'\"\(\)\[\]:;]', newchars):
                output += " "
            output += newchars
    return output


def createNextChars(followOptions):
    if (followOptions is None):
        return ""
    if (not "SIZE" in followOptions):
        return ""
    rand = math.floor(random.random() * (followOptions["SIZE"] - 1))
    for k in followOptions:
        if k != "SIZE":
            weight = followOptions[k]
            if rand <= weight:
                return k
            rand -= weight


def nextLetter(s):
    if (s is None or len(s) == 0):
        return ""
    return s.replace(r'([a-zA-Z])[^ a-zA-Z] *$', )

# ---------------------------------Main functions---------------------------------


if __name__ == "__main__":
    start_time = time.time()
    length = 200
    # for y in range(1940, 1950):
    try:
        modelf = open("model_"+str(order)+"_TAR_"+str(year), "rb")
        table = pickle.load(modelf)
    except:
        directory = os.fsencode("processed/")
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and filename.startswith("TAR_"+str(year)):
                readf = codecs.open("processed/"+filename, 'r',
                                    encoding="utf-8", errors='ignore')
                text = readf.read()
                readf.close()
                addToTable(text, order)
    modelf = open("model_"+str(order)+"_TAR_"+str(year), "wb+")
    pickle.dump(table, modelf)

    modelf.close()
    out = createText("", length, table, order, size)
    print(out)
    print("--- %s seconds ---" % (time.time() - start_time))
