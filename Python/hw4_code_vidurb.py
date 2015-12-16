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
from math import fabs
import subprocess
import itertools
import numpy
import sys
import os

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Q1
def findCommonWords(tf1,tf2, topN):
    lines1 = loadFileLines(tf1)
    lines2 = loadFileLines(tf2)
    words1 = dict()
    words2 = dict()
    chi1 = dict()
    chi2 = dict()
    
    for topicWords in lines1:
        key, chi, rank = topicWords.split()
        words1[key] = float(rank)
        chi1[key] = float(chi)

    for topicWords in lines2:
        key, chi, rank = topicWords.split()
        words2[key] = float(rank)
        chi2[key] = float(chi)

    commonWords = dict()
    wordset = set(words1).intersection(set(words2))
    for word in wordset:
        if(chi1[word] > 10 or chi2[word] > 10):
            if word == "history":
                print chi1[word]
                print chi2[word]
            commonWords[word] = fabs(words1[word] - words2[word])
    
    orderedWordCounts1 = OrderedDict(Counter(commonWords).most_common(topN))

    diffWords = dict()
    wordset = set(words1).difference(set(words2))
    for word in wordset:
        if chi1[word] > 10:
            diffWords[word] = chi1[word]

    orderedWordCounts2 = OrderedDict(Counter(diffWords).most_common(topN))

    diffWords = dict()
    wordset = set(words2).difference(set(words1))
    for word in wordset:
        if chi2[word] > 10:
            diffWords[word] = chi2[word]

    orderedWordCounts3 = OrderedDict(Counter(diffWords).most_common(topN))

    orderedWords = []
    orderedWords.extend(orderedWordCounts1.keys())
    orderedWords.extend(orderedWordCounts2.keys())
    orderedWords.extend(orderedWordCounts3.keys())
    
    return orderedWords

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Q1
def load_topic_words(topicFile, topN):
    topicFileLines = loadFileLines(topicFile)
    topicWordsDict = dict()
    
    for topicWords in topicFileLines:
        key, value = topicWords.split()
        topicWordsDict[key] = float(value)

    topNTopicWordsDict = OrderedDict(Counter(topicWordsDict).most_common(topN))
    for key in topNTopicWordsDict.keys():
        if topNTopicWordsDict[key] < 10:
            del topNTopicWordsDict[key]

    topNTopicWords = topNTopicWordsDict.keys()       
    return topNTopicWords

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Q2.1
def cluster_keywords_wn(keyList, outputFile):
    wordClusters = []
    wordLemmas = dict()
    
    for word in keyList:
        wordLemmas[word] = set()

        wordSynsets = wordnet.synsets(word);
        wordLemmas[word] = wordLemmas[word].union(getLemmasFromSynsets(wordSynsets))
       
        for wordSynset in wordSynsets:
            wordHypers = wordSynset.hypernyms()
            wordLemmas[word] = wordLemmas[word].union(getLemmasFromSynsets(wordHypers))

            wordHypos = wordSynset.hyponyms()
            wordLemmas[word] = wordLemmas[word].union(getLemmasFromSynsets(wordHypos))

    for currIter in range(0,len(keyList)):
        currWord = keyList[currIter]
        if wordLemmas[currWord] != set([-1]):
            wordCluster = currWord;

            for nextIter in range(currIter+1, len(keyList)):
                nextWord = keyList[nextIter]
                if len(wordLemmas[currWord].intersection(wordLemmas[nextWord]))>0:
                    wordCluster = wordCluster + ',' + nextWord
                    wordLemmas[nextWord] = set([-1])
                    
            wordClusters.append(wordCluster)

    saveListToFile(outputFile, wordClusters)
    return True;

#Q2.2
def expand_keywords_dp(keyList, inputDir, outputFile):
    lexparserCmd = ['/home1/c/cis530/hw4/lexparser.sh']
    lexparserCmd.extend(loadDirectory(inputDir))
    lexparserOutput = subprocess.check_output(lexparserCmd)

    expansionDict =  defaultdict(set)
    for line in lexparserOutput.split(')'):
        line = line.strip()
        if len(line) > 0:
            words = line.split(', ')
            firstWord = words[0].split('(')[1].rsplit('-',1)[0]
            firstCount = int(words[0].split('(')[1].rsplit('-',1)[1])
            secondWord = words[1].rsplit('-',1)[0]
            secondCount = int(words[1].rsplit('-',1)[1])

            if(firstCount!=0 and secondCount!=0 and firstWord != secondWord):
                expansionDict[firstWord].add(secondWord)
                expansionDict[secondWord].add(firstWord)

    expansionArray = []
    for word in keyList:
        expansionString = word + '\t'

        if word in expansionDict:
            for expWord in expansionDict[word]:
                expansionString += expWord + ','

        expansionArray.append(expansionString[:-1]);
        
    saveListToFile(outputFile, expansionArray)
    return True

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Q3.1
def summarize_baseline(inputDir, outputFile):
    summary = []
    summaryLength = 0
    
    fileNames = sorted(loadDirectory(inputDir));
    for fileName in fileNames:
        if(summaryLength < 100):
            firstSentence = readFirstSentence(fileName)
            summary.append(firstSentence)
            summaryLength += len(word_tokenize(firstSentence))

    saveListToFile(outputFile, summary)
    return True

#Q3.2
def summarize_kl(inputDir, outputFile):
    stopWordsFile = 'stopwords.txt'
    stopWords = loadFileLines(stopWordsFile)
    
    # Building the counts per sentence in the corpus
    # Also, keeping track of all occurrences of words and all sentences
    corpusSentences = []
    corpusWords = []
    corpusWordsCounts = []
    corpusWordsTotal = 0
    sentenceWordsCounts = []
    fileNames = sorted(loadDirectory(inputDir));
    for fileName in fileNames:
        fileSentences = loadFileLines(fileName)
        corpusSentences.extend(fileSentences)

        for sentence in fileSentences:
            sentenceWOStop = [word for word in word_tokenize(sentence) if word not in stopWords]
            corpusWords.extend(sentenceWOStop)
            sentenceWordsCounts.append(Counter(sentenceWOStop))

    corpusWordsCounts = Counter(corpusWords)
    corpusWordsTotal = len(corpusWords)

    # Building the summary based on KL Divergence
    summary = []
    summaryWords = []
    summaryLength = 0
    sentencesConsidered = []
    while summaryLength < 100 and len(sentencesConsidered) < len(corpusSentences):
        minKL = sys.float_info.max
        minIndex = 0
        
        # Find the sentence that minimizes the KL Divergence
        for sentIter in range(0,len(corpusSentences)):
            if sentIter not in sentencesConsidered:
                summaryWordsCounts = Counter(summaryWords) +  sentenceWordsCounts[sentIter]
                summaryWordsTotal = sum(summaryWordsCounts.values())

                currKL = 0
                for word in summaryWordsCounts.keys():
                    p = summaryWordsCounts[word]/summaryWordsTotal
                    q = corpusWordsCounts[word]/corpusWordsTotal

                    if p!=0 and q!=0:
                        currKL += p*log(p/q)
                        
                if currKL < minKL and currKL != 0:
                    minKL = currKL
                    minIndex = sentIter
        
        sentencesConsidered.append(minIndex)        
        summaryWords.extend(sentenceWordsCounts[minIndex].keys())
        minSentence = corpusSentences[minIndex]
        summary.append(minSentence)
        summaryLength += len(word_tokenize(minSentence))

    saveListToFile(outputFile, summary)
    return True

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#UTIL FUNCTIONS
def readFirstSentence(filepath):
        fileInstance = open(filepath)
        firstSentence = fileInstance.readline().decode('utf8')
        fileInstance.close()
        
        return firstSentence[:-1]
    
def getLemmasFromSynsets(synsets):
    lemmas = set()
    for synset in synsets:
            for lemma in synset.lemmas():
                lemmas.add(lemma)

    return lemmas

def loadFileLines(filepath):
    fileInstance = open(filepath)
    fileLines = []
    while True:
        line = fileInstance.readline().decode('utf8')
        if not line:
            fileInstance.close()
            break

        fileLines.append(line[:-1])
    return fileLines

def loadDirectory(dirName):
    dirFiles = os.listdir(dirName)
    if dirFiles == 'none':
        dirFiles = []
    fileList = [os.path.join(dirName, fileName) for fileName in dirFiles]
    
    return fileList

def saveListToFile(fileName, listName):
    outputFile = open(fileName,'wb')

    for key in listName:
        outputFile.write(unicode(key).encode('utf8'))
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
#Q0:

    topicFile1 = 'TSFiles/gina.ts'
    topicFile2 = 'TSFiles/nonGina.ts'
    rankedWords = findCommonWords(topicFile1,topicFile2, 5000)
    saveListToFile('rankedChiWords.txt',rankedWords)
    #printList(orderedWordCounts,200)

#Q1
    """
    topicFile = 'TSFiles/dev_10.ts'
    topNTopicWords = load_topic_words(topicFile, 20)
    for word in topNTopicWords:
        print word
    print("==========")
    """

#Q2.1
    """
    topicFile = 'TSFiles/dev_10.ts'
    topNTopicWords = load_topic_words(topicFile, 20)
    outputFile = 'hw4_2_1.txt'
    cluster_keywords_wn(topNTopicWords, outputFile)
    print("====================")
    """

#Q2.2
    """
    topicFile = 'TSFiles/dev_10.ts'
    topNTopicWords = load_topic_words(topicFile, 200)
    outputFile = 'hw4_2_2.txt'
    inputDir = '/home1/c/cis530/hw4/dev_input/dev_10'
    expand_keywords_dp(topNTopicWords, inputDir, outputFile)
    print("==============================")
    """

#Q3.1
    """
    inputDir = 'dev_input/dev_10' 
    outputFile = 'hw4_3_1.txt'

    summarize_baseline(inputDir, outputFile)
    print("========================================")
    """

#Q3.2
    """
    inputDir = 'dev_input/dev_10' 
    outputFile = 'hw4_3_2.txt'
    summarize_kl(inputDir, outputFile)
    print("==================================================")
    """

#Q4.1
    """
    for fqdn in loadDirectory('dev_input'):
        folderName = fqdn.split('\\')[1]
        inputDir = 'dev_input\\' + folderName

        outputFile = 'baseline\sum_' + folderName + '.txt'
        summarize_baseline(inputDir, outputFile)
        print("========================================")
        
        outputFile = 'kl\sum_' + folderName + '.txt'
        summarize_kl(inputDir, outputFile)
        print("==================================================")
    """
