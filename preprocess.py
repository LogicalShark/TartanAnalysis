import os
import fileinput
import codecs
import re
import pickle
import nltk
from pattern.en import parsetree


def preprocess(filename):
    readf = codecs.open("raw/"+filename, 'r',
                        encoding="utf-8", errors='ignore')
    filedata = readf.read()
    readf.close()

    # # hyphens often split words between lines
    # filedata = re.sub(r'(?<=[A-Za-z])(\uFFFD|\-)\s+\n', '', filedata)
    # # unrecognized characters are usually apostrophes/quotation marks
    # filedata = re.sub(r'\uFFFD', '\'', filedata)

    # writef = codecs.open("processed/"+filename, 'w+',
    #                      encoding="utf-8", errors='ignore')
    # writef.write(filedata)
    # writef.close()

    writef = codecs.open("syntax/"+filename, 'w+',
                         encoding="utf-8", errors='ignore')
    for sent in nltk.sent_tokenize(filedata, language='english'):
        # print(sent)
        options = parsetree(sent) # replace "raise StopIteration" with "return"
        if len(options) > 0:
            for chunk in options[0].chunks:
                writef.write(str(chunk.type + str([w.type for w in chunk.words])) + "\n")
    writef.close()


if __name__ == "__main__":
    directory = os.fsencode("raw/")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            preprocess(filename)
