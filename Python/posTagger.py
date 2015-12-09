#-----------------------------------------------------------------------
#IMPORTS
from __future__ import division
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk import pos_tag, map_tag
import numpy as np
import sys
import os

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#UTIL FUNCTIONS
def getSentenceFeatures(fileLines):
    itr = 0
    sentFeat = []
    for line in fileLines:
        sentences = sent_tokenize(line)
        wordsPerSent = np.array([len(word_tokenize(sentence)) for sentence in sentences])
        
        sentFeat.append([])
        sentFeat[itr].append(wordsPerSent.mean())
        sentFeat[itr].append(wordsPerSent.std())
        itr += 1
    return sentFeat

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

def printList(listName,limit):
    for itr in range(0,limit):
        print listName[itr]
        
    return

def printDict(dictName,limit):
    itr = 0
    for key,value in dictName.items():
        itr += 1
        print key + " : " + str(value)

        if(itr == limit):
            break
        
    return

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#MAIN - TESTING THE CODE FROM MAIN

if __name__ == "__main__":
    print("=====ASSIGNMENT STARTED=====")

    # Parts of Speech Tagger
    """
    fileName = "actualWords.txt"
    allWords = loadFileLines(fileName)
    
    posTags = pos_tag(allWords)
    
    
    mapper = dict([('ADJ', 1), ('ADP', 2), ('ADV', 3), ('CONJ', 4), ('DET', 5), ('NOUN', 6), ('NUM', 7), ('PRT', 8), ('PRON', 9), ('VERB', 10), ('.', 11), ('X', 12)])
    simplifiedTags = [mapper[map_tag('en-ptb', 'universal', tag)] for word, tag in posTags]

    fileName = "posTags.txt"
    saveListToFile(fileName, simplifiedTags)
    print("===== SAVED POS FILE ===== ")
    """
#-----------------------------------------------------------------------    
    # Word Lengths
    """
    fileName = "actualWords.txt"
    allWords = loadFileLines(fileName)
    wordsLen = []

    for word in allWords:
        wordsLen.append(len(word))

    fileName = "wordLengths.txt"
    saveListToFile(fileName, wordsLen)
    print("===== SAVED LENGTH FILE ===== ")
    """
#-----------------------------------------------------------------------
    # Average number of words per sentence
    # Sentence length dev
    
    trainFileName = 'project_articles_train'
    testFileName = 'project_articles_test'

    fileLines = loadFileLines(trainFileName)
    trainSentFeat = getSentenceFeatures(fileLines)
    fileName = "trainSentFeat.txt"
    saveListToFile(fileName, trainSentFeat)
    
    fileLines = loadFileLines(testFileName)
    testSentFeat = getSentenceFeatures(fileLines)
    fileName = "testSentFeat.txt"
    saveListToFile(fileName, testSentFeat)
    print("===== SAVED LENGTH FILE ===== ")
