import os
import time
import fileinput
import codecs
import re
import pickle
import nltk
from nltk import word_tokenize
from pattern.en import parsetree
import gib_detect_train

model_data = pickle.load(open('gib_model.pki', 'rb'))
model_mat = model_data['mat']
threshold = model_data['thresh']


def processText(filedata):
    # hyphens often split words between lines
    filedata = re.sub(r'\r', '', filedata)
    # m = re.search(r'[^a-zA-Z0-9\s\.]+\n', filedata)
    # if not m is None:
    #     print(m.group(0))
    filedata = re.sub(r'[^a-zA-Z0-9]*[^a-zA-Z0-9\s\.]+\n', '', filedata)
    # note: unrecognized characters are usually apostrophes/quotation marks
    filedata = re.sub(r'[^\w\s\.\"\',;:!?@#$%\^&\*\(\)\<\>/\\\{\}\[\]\|\+=\-_]', '\'', filedata)
    # fix typos
    filedata = re.sub(r'stu[\s\.,\'\*\|\-]+dent', 'student', filedata)
    filedata = re.sub(r'Stu[\s\.,\'\*\|\-]+dent', 'Student', filedata)
    filedata = re.sub(r'fi[\.-:;\|\']+rst', 'first', filedata)
    filedata = re.sub(r'\si\.+rst\s', ' first ', filedata)
    filedata = re.sub(r'[\.-:;\|\'\s]+rst[\.-:;\|\'\s]+', ' first ', filedata)

    sents = nltk.sent_tokenize(filedata)
    processed = ""
    for s in sents:
        if gib_detect_train.avg_transition_prob(s, model_mat) < threshold:
            continue
        processed += " "+s
    return processed


def preprocess(filename):
    print("processing", filename)
    decade = 0
    readf = codecs.open("raw/"+filename, 'r',
                        encoding="cp1252", errors="replace")
    filedata = readf.read()
    readf.close()

    processed = processText(processText(filedata))
    for year in range(1900, 2020):
        if filename.startswith("TAR_"+str(year)):
            if year < 1920:
                decade = 1900
            decade = year-(year % 10)
            break

    if not os.path.exists(os.path.dirname("decades/"+str(decade)+"/")):
        os.makedirs(os.path.dirname("decades/"+str(decade)+"/"))

    writef = codecs.open("decades/"+str(decade)+"/"+filename, 'w+',
                         encoding="utf-8", errors='ignore')
    writef.write(processed)
    writef.close()

    writef = codecs.open("syntax/"+str(decade)+filename, 'w+',
                         encoding="utf-8", errors='ignore')
    for sent in nltk.sent_tokenize(processed, language='english'):
        # if error, replace "raise StopIteration" with "return" in source file
        options = parsetree(sent)
        if len(options) > 0:
            for chunk in options[0].chunks:
                writef.write(
                    str(chunk.type + str([w.type for w in chunk.words])) + "\n")
    writef.close()


def processRaw():
    directory = os.fsencode("raw/")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        preprocess(filename)


def separateYears():
    for decade in range(1900, 2020, 10):
        directory = os.fsencode("decades/"+str(decade))
        try:
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                for year in range(1900, 2020):
                    if filename.startswith("TAR_"+str(year)):
                        readf = codecs.open(
                            "decades/"+str(decade)+"/"+filename, 'r', encoding="utf-8", errors='ignore')
                        if not os.path.exists(os.path.dirname("years/"+str(year)+"/")):
                            os.makedirs(os.path.dirname(
                                "years/"+str(year)+"/"))
                        writef = codecs.open(
                            "years/"+str(year)+"/"+filename, 'w+', encoding="utf-8", errors='ignore')
                        writef.write(readf.read())
                        readf.close()
                        writef.close()
        except:
            pass


if __name__ == "__main__":
    start_time = time.time()
    processRaw()
    separateYears()
    print("--- %s seconds ---" % (time.time() - start_time))
    pass
