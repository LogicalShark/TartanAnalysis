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
    filedata = re.sub(
        r'[^a-zA-Z0-9]*[^a-zA-Z0-9\s\.,;:!?@#$%*\)\(\{\}\[\]\&\"\']+ ?\n', '', filedata)
    # note: unrecognized characters are usually apostrophes/quotation marks
    filedata = re.sub(
        r'[^a-zA-Z0-9 \t\n\.\"\',;:!?@#$%\^&*\(\)\<\>/\\\{\}\[\]|+=\-_]', "'", filedata)
    filedata = re.sub(r'\n\n\n+', '\n\n', filedata)
    filedata = re.sub(r' [ \t]+', ' ', filedata)

    return filedata

def processDecades():
    prevPrevData = []
    prevData = []
    repeats = set()
    for decade in range(1900, 2020, 10):
        print("processing", str(decade))
        directory = os.fsencode("decades_modified/"+str(decade))
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            readf = codecs.open("decades_modified/"+str(decade)+"/"+filename, 'r',
                                encoding="cp1252", errors="replace")
            filedata = readf.read()
            readf.close()
            processed = processText(filedata)
            lines = processed.split("\n")
            reprocessed = ""
            for s in lines:
                if len(s) < 4 or gib_detect_train.avg_transition_prob(s, model_mat) < threshold or "......." in s:
                    pass
                # elif s in prevData or s in prevPrevData:
                #     repeats.add(s)
                else:
                    reprocessed += s+"\n"
            output = reprocessed
            # sents = nltk.sent_tokenize(reprocessed)
            # output = ""
            # for s in sents:
            #     if len(s) < 4 or gib_detect_train.avg_transition_prob(s, model_mat) < threshold:
            #         pass
            #     elif s in prevData or s in prevPrevData:
            #         repeats.add(s)
            #     else:
            #         output += s + " "
            writef = codecs.open("decades_modified/"+str(decade)+"/"+filename, 'w+',
                                 encoding="utf-8", errors='ignore')
            writef.write(output)
            writef.close()
            # prevPrevData = prevData
            # prevData = sents+lines

    # for decade in range(1900, 2020, 10):
    #     print("processing", str(decade))
    #     directory = os.fsencode("decades_modified/"+str(decade))
    #     for file in os.listdir(directory):
    #         filename = os.fsdecode(file)
    #         readf = codecs.open("decades_modified/"+str(decade)+"/"+filename, 'r',
    #                             encoding="cp1252", errors="replace")
    #         filedata = readf.read()
    #         readf.close()
    #         processed = ""
    #         for s in filedata.split("\n"):
    #             if len(s) < 4 or gib_detect_train.avg_transition_prob(s, model_mat) < threshold or s in repeats:
    #                 pass
    #             else:
    #                 processed += s+"\n"
    #         sents = nltk.sent_tokenize(processed)
    #         output = ""
    #         for s in sents:
    #             if len(s) < 4 or gib_detect_train.avg_transition_prob(s, model_mat) < threshold or s in repeats:
    #                 pass
    #             else:
    #                 output += s+" "
    #         writef = codecs.open("decades_modified/"+str(decade)+"/"+filename, 'w+',
    #                              encoding="utf-8", errors='ignore')
    #         writef.write(output)
    #         writef.close()
    #         # writef = codecs.open("syntax/"+str(decade)+"/"+filename, 'w+',
    #         #              encoding="utf-8", errors='ignore')
    #         # for sent in nltk.sent_tokenize(output, language='english'):
    #         #     # if errors, replace "raise StopIteration" with "return" in source file
    #         #     options = parsetree(sent)
    #         #     if len(options) > 0:
    #         #         for chunk in options[0].chunks:
    #         #             writef.write(
    #         #                 str(chunk.type + str([w.type for w in chunk.words])) + "\n")
    #         # writef.close()

def separateYears():
    for decade in range(1900, 2020, 10):
        directory = os.fsencode("decades_modified/"+str(decade))
        try:
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                for year in range(1900, 2020):
                    if filename.startswith("TAR_"+str(year)):
                        readf = codecs.open(
                            "decades_modified/"+str(decade)+"/"+filename, 'r', encoding="utf-8", errors='ignore')
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
    # processDecades()
    separateYears()
    print("--- %s seconds ---" % (time.time() - start_time))
    pass
