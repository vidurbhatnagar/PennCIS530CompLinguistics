#-----------------------------------------------------------------------
#IMPORTS
from __future__ import division
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
from nltk import word_tokenize
from nltk import pos_tag, map_tag
import sys
import os

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#UTIL FUNCTIONS
def loadFileLines(filepath):
    fileInstance = open(filepath)
    fileLines = []
    while True:
        line = fileInstance.readline().decode('utf8')
        if not line:
            fileInstance.close()
            break

        fileLines.append(line.strip())
    return fileLines


def saveListToFile(fileName, listName):
    outputFile = open(fileName,'wb')

    for key in listName:
        outputFile.write(str(unicode(key).encode('utf8')).replace('[','').replace(']',''))
        outputFile.write('\n')

    outputFile.close()
    
    return True

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#MAIN - TESTING THE CODE FROM MAIN

if __name__ == "__main__":
    print("=====ASSIGNMENT STARTED=====")

    #"""
    fileName = "actualWords.txt"
    allWords = loadFileLines(fileName)
    
    posTags = pos_tag(allWords)
    
    
    mapper = dict([('ADJ', 1), ('ADP', 2), ('ADV', 3), ('CONJ', 4), ('DET', 5), ('NOUN', 6), ('NUM', 7), ('PRT', 8), ('PRON', 9), ('VERB', 10), ('.', 11), ('X', 12)])
    simplifiedTags = [mapper[map_tag('en-ptb', 'universal', tag)] for word, tag in posTags]

    fileName = "posTags.txt"
    saveListToFile(fileName, simplifiedTags)
    print("===== SAVED POS FILE ===== ")
    
    fileName = "actualWords.txt"
    allWords = loadFileLines(fileName)
    wordsLen = []

    for word in allWords:
        wordsLen.append(len(word))

    fileName = "wordLengths.txt"
    saveListToFile(fileName, wordsLen)
    print("===== SAVED LENGTH FILE ===== ")
    
