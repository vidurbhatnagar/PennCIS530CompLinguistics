#-----------------------------------------------------------------------
#IMPORTS
from __future__ import division
from nltk.corpus import wordnet
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
from nltk import word_tokenize
from itertools import izip
from math import sqrt
from math import log
import subprocess
import itertools
import numpy
import sys
import os

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def getIndicesForWords(FVWords, words):
    wordsIndices = []
    for word in words:
        if word in FVWords:
            wordsIndices.append(FVWords.index(word)+1)
        
    return wordsIndices

def  get_idf(corpus):
    idf = Counter()
    courpusLength = len(corpus)

    for excerpt in corpus:
        idf.update(list(set(excerpt)))

    for df in idf:
        idf[df]=log(courpusLength/idf[df])

    return idf

def getTopWordsIndices(fileLines, featureSpaceDict, limit):    
    topWordsDict = OrderedDict(Counter(flatten(fileLines)).most_common(limit))
    printDict(topWordsDict,5)
    topWordsIndices = []

    for key in topWordsDict:
        topWordsIndices.append(featureSpaceDict[key]+1)
          
    return topWordsIndices

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#UTIL FUNCTIONS
def createDiffFiles(inFile,outFile1,outFile2):
    inF = open(inFile)
    outF1 = open(outFile1,'wb')
    outF2 = open(outFile2,'wb')
    while True:
        line = inF.readline().decode('utf8')
        if not line:
            inF.close()
            outF1.close()
            outF2.close()
            break

        lineParts = line.split("\t")
        if(int(lineParts[1]) == 0):
            outF1.write(str(unicode(lineParts[0]+'\n').encode('utf8')).replace('[','').replace(']',''))
        else:
            outF2.write(str(unicode(lineParts[0]+'\n').encode('utf8')).replace('[','').replace(']',''))

    return

def createFeatureVectors(fileLines, featureSpaceDict, outputFileName):
    outputFile = open(outputFileName,'wb')

    for line in fileLines:
        featureVector = [0] * len(featureSpaceDict)
        wordCounts = Counter(line)
        
        for word in wordCounts:
            featureVector[int(featureSpaceDict[word])] = wordCounts[word]

        outputFile.write(unicode(str(featureVector).replace('[','').replace(']','')).encode('utf8'))
        outputFile.write('\n')
        
    outputFile.close()        
    return True

def createFeatureSpace(fileLines):
    wordList = list(set(flatten(fileLines)))
    
    return {x:i for i,x in enumerate(wordList)}

def  wordTokenize(line):
   return word_tokenize(line.lower())

def flatten(listoflists):
   return list(itertools.chain.from_iterable(listoflists))

def loadFileLines(fileName):
    fileInstance = open(fileName)
    fileLines = []
    fileTags = []
    
    while True:
        line = fileInstance.readline().decode('utf8')
        if not line:
            fileInstance.close()
            break

        lineParts = line.split("\t")
        fileLines.append(wordTokenize(lineParts[0].strip()))
        fileTags.append(lineParts[-1].strip())
    
    return fileLines, fileTags

def loadDirectory(dirName):
    dirFiles = os.listdir(dirName)
    if dirFiles == 'none':
        dirFiles = []
    fileList = [os.path.join(dirName, fileName) for fileName in dirFiles]
    
    return fileList

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

##LOADING DATA   
    """
    trainFileName = 'project_articles_train'
    #trainFileName = 'trainDummy'
    trainFileLines, trainFileTags = loadFileLines(trainFileName)
    #outputFileName = 'trainLabels.txt'
    #saveListToFile(outputFileName,trainFileTags);
    print("=====TRAINING DATA LOADED=====")

    testFileName = 'project_articles_test'
    #testFileName = 'testDummy'
    testFileLines, testFileTags = loadFileLines(testFileName)
    print("=====TESTING DATA LOADED=====")
    
    allFileLines = []
    allFileLines.extend(trainFileLines)
    allFileLines.extend(testFileLines)
    """

##CREATING FEATURE SPACE   
    """
    featureSpaceDict = createFeatureSpace(allFileLines)
    featureSpaceList = [None] * len(featureSpaceDict)

    for key,value in featureSpaceDict.items():
        featureSpaceList[int(value)] = key
    outputFileName = 'actualWords.txt'
    saveListToFile(outputFileName,featureSpaceList);
    print("=====FEATURE SPACE CREATED=====")
    
    featureVectorList = createFeatureVectors(trainFileLines, featureSpaceDict, 'trainX.txt')
    print("=====TRAIN FEATURE VECTORS CREATED=====")

    featureVectorList = createFeatureVectors(testFileLines, featureSpaceDict,'testX.txt')
    print("=====TEST FEATURE VECTORS CREATED=====")
    """
    
##TOP WORDS INDICS   
    """
    topWordsIndices = getTopWordsIndices(allFileLines, featureSpaceDict, len(featureSpaceDict))
    outputFileName = 'topWordsIndices.txt'
    saveListToFile(outputFileName,topWordsIndices);
    print("=====TOP WORDS INDICES CREATED=====")
    """

##TFIDF VALUE FILE
    """
    fileName = 'actualWords.txt'
    dummmyWords,actualWords = loadFileLines(fileName)
    idf = get_idf(allFileLines)
    idfOrdered = []
    
    for word in actualWords:
        idfOrdered.append(idf[word])
        
    outputFileName = 'wordIdfs.txt'
    saveListToFile(outputFileName,idfOrdered);
    """

##SEPARATING TEXT INTO 2 FILES
    """
    inFile = 'project_articles_train'
    outFile1 = 'nonGina.txt'
    outFile2 = 'gina.txt'

    createDiffFiles(inFile,outFile1,outFile2)
    """

##CREATING INDICES FOR CHISQ RANKED WORDS
    fileName = 'actualWords.txt'
    dummmyWords,actualWords = loadFileLines(fileName)
    fileName = 'rankedChiWords.txt'
    dummmyWords,chiWords = loadFileLines(fileName)

    wordIndices = getIndicesForWords(actualWords, chiWords)
    saveListToFile('rankedChiIndices.txt',wordIndices)

    #printList(wordIndices,10)
    #print len(featureSpaceDict)
    #print len(topWordsIndices)
    #printDict(featureSpaceDict, len(featureSpaceDict))
    #printList(topWordsIndices, len(topWordsIndices))
    print("=====PRINTING COMPLETE =====")



    

