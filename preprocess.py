import os
import fileinput
import codecs
import re

def preprocess(filename):
    readf = codecs.open("raw/"+filename, 'r',
                        encoding="utf-8", errors='ignore')
    filedata = readf.read()
    readf.close()

    # hyphens often split words between lines
    filedata = re.sub(r'(?<=[A-Za-z])(\uFFFD|\-)\s+', '', filedata)
    # unrecognized characters are usually apostrophes/quotation marks
    filedata = re.sub(r'\uFFFD', '\'', filedata)

    writef = codecs.open("processed/"+filename, 'w+',
                         encoding="utf-8", errors='ignore')
    writef.write(filedata)
    writef.close()

if __name__ == "__main__":
    directory = os.fsencode("raw/")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            preprocess(filename)
