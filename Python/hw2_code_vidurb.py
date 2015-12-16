#-----------------------------------------------------------------------
#IMPORTS
from __future__ import division
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
from nltk import word_tokenize
from nltk import sent_tokenize
from itertools import izip
from math import sqrt
from math import log
import subprocess
import itertools
#import numpy
import sys
import os


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#GLOBAL VARIABLES
vidurb_dataDir = '/home1/v/vidurb/CL530/Project/Python/'
vidurb_answersDir ='/home1/v/vidurb/CL530/Project/Python/SRILM/'
    
srilm = '/home1/c/cis530/srilm/'

vidurb_corpus = ''
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Q4
def analyzeLM(models, output):
    utilFunctions = UtilFunctions()

    outputFile = open(output,'wb')
    for className,modelFile in sorted(dict(models).items()):
        classCounts = utilFunctions.loadFileLines(modelFile)
        classCountsDict = {}
        classCountsDict = dict((classCount.split('\t')[0], classCount.split('\t')[-1])  for classCount in classCounts)

        vocabSize = len(classCountsDict)
        freqDict = dict((key, value) for key, value in classCountsDict.items() if int(value) > 5)
        rareDict = dict((key, value) for key, value in classCountsDict.items() if int(value) == 1)
        fracFreq = len(freqDict) / vocabSize
        fracRare = len(rareDict) / vocabSize

        outputFile.write(className + ',' + str(vocabSize) + ',' + str(fracFreq)  + ',' + str(fracRare) + '\n')

    outputFile.close()        
    return

def get_entropy(unigramModelFile, unigramCountsFile):
    utilFunctions = UtilFunctions()

    unigramModel = utilFunctions.loadFileLines(unigramModelFile)
    unigramModelDict = dict((unigram.split('\t')[-1].replace('\n',''), unigram.split('\t')[0].replace('\n',''))  for unigram in unigramModel)

    unigramCounts = utilFunctions.loadFileLines(unigramCountsFile)
    unigramCountsDict = dict((unigramCount.split('\t')[0].replace('\n',''), unigramCount.split('\t')[-1].replace('\n',''))  for unigramCount in unigramCounts)

    commonTokens = list(set(unigramModelDict).intersection(set(unigramCountsDict)))

    entropy = 0.0
    totalCount = 0.0
    """
    for commonToken in commonTokens:
        entropy += -1 * float(unigramCountsDict[commonToken]) * float(unigramModelDict[commonToken])
        totalCount += float(unigramCountsDict[commonToken])
    """
    for commonToken in commonTokens:
        logProb = float(unigramModelDict[commonToken])
        logProbBase2 = logProb * log(10,2)
        entropy += -1 * logProbBase2 * pow(2,logProbBase2)

    return entropy

def get_type_token_ratio(unigramCountsFile):
    utilFunctions = UtilFunctions()

    unigramCounts = utilFunctions.loadFileLines(unigramCountsFile)
    unigramCountsDict = dict((unigramCount.split('\t')[0].replace('\n',''), float(unigramCount.split('\t')[-1].replace('\n','')))  for unigramCount in unigramCounts)

    return (len(unigramCountsDict)/sum(unigramCountsDict.values()))

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Q3
def lm_predict(models, testFile, predFile):
    utilFunctions = UtilFunctions()
    testFileLines = utilFunctions.loadFileLines(testFile)

    outputFile = open(predFile,'wb')
    for line in testFileLines:
        pplMin = sys.float_info.max
        pplMinClass = ''
        
        for className, modelFile in sorted(dict(models).items()):
            ppl = srilm_ppl(modelFile, line)
            if ppl < pplMin:
                pplMinClass = className
                pplMin = ppl
                
        outputFile.write(unicode(pplMinClass).encode('utf8'))
        outputFile.write('\n')

    outputFile.close()        
    return

def confusion_matrix(labels, predictions):
    utilFunctions = UtilFunctions()
    featureSpace = utilFunctions.create_feature_space(labels)
    
    confMatrix = [[0]*len(featureSpace) for i in range(len(featureSpace))]
    labelCounts = Counter(labels)

    for i in range(len(labels)):
        confMatrix[featureSpace[labels[i]]][featureSpace[predictions[i]]] += 1 / labelCounts[labels[i]]

    return confMatrix

def evaluate(labelFile, predFile):
    utilFunctions = UtilFunctions()

    labels = utilFunctions.loadFileLines(labelFile)
    predictions = utilFunctions.loadFileLines(predFile)

    utilFunctions = UtilFunctions()
    featureSpace = utilFunctions.create_feature_space(labels)
    labelCounts = Counter(labels)
    
    confMatrix = confusion_matrix(labels, predictions)
            
    accuracy = 0.0
    for key,value in featureSpace.items():
        accuracy +=  confMatrix[value][value] * labelCounts[key]

    accuracy = accuracy/len(labels)
    
    return (accuracy, confMatrix)

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Q2
def srilm_bigram_models(inputFile, outputDir):
    basename = os.path.basename(inputFile)
    
    uniModelFilePath = os.path.join(outputDir, basename+'.uni.lm')
    uniCountFilePath = os.path.join(outputDir, basename+'.uni.counts')
    biModelFilePath = os.path.join(outputDir, basename+'.bi.lm')
    knModelFilePath = os.path.join(outputDir, basename+'.bi.kn.lm')

    tempFilePath = os.path.join(outputDir, 'vidurb_TempFile')
    srilmPreprocess(inputFile, tempFilePath, True)
    
    subprocess.check_output([srilm + 'ngram-count','-text', tempFilePath, '-lm', uniModelFilePath, '-write', uniCountFilePath, '-order', '1', '-addsmooth', '0.25'])
    subprocess.check_output([srilm + 'ngram-count','-text', tempFilePath, '-lm', biModelFilePath, '-order', '2', '-addsmooth', '0.25'])
    subprocess.check_output([srilm + 'ngram-count','-text', tempFilePath, '-lm', knModelFilePath, '-order', '2', '-kndiscount'])
    
    return

def srilm_ppl(modelFile, rawText):
    tempFilePath = 'vidurb_TempFile'
    srilmPreprocess(rawText,tempFilePath,False)

    ppl = subprocess.check_output([srilm + 'ngram','-lm', modelFile, '-ppl', tempFilePath])

    return float(ppl.split('ppl=')[1].split(' ')[1])
 
def  srilmPreprocess(inputFile, tempFile, isFile):
    utilFunctions = UtilFunctions()

    if isFile:
        sentenceList = utilFunctions.loadFileExcerpts(inputFile)
    else:
        sentenceList = utilFunctions.tokenizeSentences(inputFile)

    outputFile = open(tempFile,'w')
    for sentence in sentenceList:
        outputFile.write((sentence+'\n').encode('utf8'))

    outputFile.close()    

    return

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Q1
class BigramModel:
    UNKTOKEN = 'UNK'
    
    def __init__ (self, trainFiles):
        utilFunctions = UtilFunctions()

        corpusSentenceList = utilFunctions.loadFileListExcerpts(trainFiles)
        corpusWordList = utilFunctions.tokenizeWords(corpusSentenceList)
        unkDict = utilFunctions.createUNKDict(corpusWordList)
        
        finalWordList = [];    
        for sentence in corpusSentenceList:
            finalWordList.append('<s>')
            finalWordList.extend(word_tokenize(sentence))
            finalWordList.append('</s>')

        #changing all Unknowns to UNK
        finalWordList = [BigramModel.UNKTOKEN if word in unkDict else word for word in finalWordList]
        finalWordDict = dict.fromkeys(finalWordList,0) #stripping down to unique vocab

        languageModel = defaultdict(int)
        for i in range(1,len(finalWordList)):
            languageModel[(finalWordList[i-1], finalWordList[i])] += 1
      
        self.finalWordList = finalWordList
        self.finalWordDict = finalWordDict
        self.languageModel = languageModel
        
        return
        
    def logprob(self, priorContext, targetWord):
        alpha = 0.25

        if priorContext not in self.finalWordDict:
            priorContext = BigramModel.UNKTOKEN

        if targetWord not in self.finalWordDict:
            targetWord = BigramModel.UNKTOKEN

        logProb = log((self.languageModel[(priorContext,targetWord)] + alpha)/(Counter(self.finalWordList)[priorContext] + alpha*len(self.finalWordDict)),2)

        return logProb
    
class UtilFunctions:
    def create_feature_space(self, wordlist):    
        return {x:i for i,x in enumerate(list(set(wordlist)))}

    def createUNKDict(self, wordlist):
        unkDict = Counter(wordlist)
        for key,value in unkDict.items():
            if value != 1:
                unkDict.pop(key)

        return unkDict
    
    def  tokenizeWords(self, sentenceList):
        wordlist = []
        for sentence in sentenceList:
            wordlist.extend(word_tokenize(sentence))
            
        return wordlist

    def  tokenizeSentences(self, excerpt):
        return sent_tokenize(excerpt)

    def loadFileLines(self, filepath):
        fileInstance = open(filepath)
        fileLines = []
       
        while True:
            excerptSentences = fileInstance.readline().decode('utf8')
            if not excerptSentences: break
            fileLines.append(excerptSentences)
          
        return fileLines

    def loadFileExcerpts(self, filepath):
        fileInstance = open(filepath)
        fileSentences = []
       
        while True:
            excerptSentences = self.tokenizeSentences(fileInstance.readline().decode('utf8'))
            if not excerptSentences: break
            fileSentences.extend(excerptSentences)
          
        return fileSentences
    
    def loadFileListExcerpts(self, fileList):
        #returns a 2-d structure - array of sentences(string array)
        
        allFileSentences = []

        for f in fileList:
            allFileSentences.extend(self.loadFileExcerpts(f))
          
        return allFileSentences

    def flatten(self, listoflists):
        return list(itertools.chain.from_iterable(listoflists))

    def saveDictToFile(fileName, dictName):
        outputFile = open(fileName,'wb')

        for key,value in dictName.iteritems():
            outputFile.write(unicode(key).encode('utf8'))
            outputFile.write('\t' + str(value) + '\n')

        outputFile.close()
        
        return

    def saveListToFile(fileName, listName):
        outputFile = open(fileName,'wb')

        for key in listName:
            outputFile.write(unicode(key).encode('utf8'))
            outputFile.write('\n')

        outputFile.close()
        
        return

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#MAIN - TESTING THE CODE FROM MAIN

if __name__ == "__main__":    
    #utilFunctions = UtilFunctions()
    fileList = [vidurb_dataDir + 'gina.txt',vidurb_dataDir + 'nonGina.txt']
    #fileList.extend([vidurb_dataDir + 'train/objective.txt', vidurb_dataDir + 'train/results.txt'])

#Q1
    """
    bigramModel = BigramModel(fileList)
    
    print('=========Bigram Language Model Built=========')   

    #Testing LogProb
    priorContext = '<s>'
    targetWord = 'A'
        
    print('Log Probability: ' + ' ' + priorContext + ', ' + targetWord)
    print(bigramModel.logprob(priorContext,targetWord))   

    print('=========Q1=========')
    """
    
#Q2
    """
    outputDir = vidurb_answersDir
    for fileName in fileList:
        srilm_bigram_models(fileName, outputDir)
        
    print('=========SRILM Language Model Built=========')   

    #Testing PPL
    #modelFile = os.path.join(vidurb_answersDir, 'background.txt.bi.lm')
    #rawText = utilFunctions.loadFileLines(fileList[0])[0]
    #print(srilm_ppl(modelFile, rawText))
    
    print('=========Q2=========')
    """

#Q3
    """
    for i in range(len(confMatrix)):
        for j in range(len(confMatrix)):
            confMatrix[i][j] = round(float(confMatrix[i][j]), 2)
    """
    
    #"""
    models = []
    models.append(('1', vidurb_answersDir + 'gina.txt.bi.kn.lm'))
    models.append(('0', vidurb_answersDir + 'nonGina.txt.bi.kn.lm'))
    
    testFile = vidurb_dataDir + 'project_articles_test'
    predFile = vidurb_answersDir + 'test.txt'

    lm_predict(models, testFile, predFile)
    
    #"""
    
    """
    labelFile = vidurb_dataDir + 'test/labels_all.txt'
    predFile = vidurb_answersDir + 'hw2_3_1.txt'
    outputFileName = vidurb_answersDir + 'confusionMatrix.txt'
    
    labels = utilFunctions.loadFileLines(labelFile)
    predictions = utilFunctions.loadFileLines(predFile)
    confMatrix = confusion_matrix(labels, predictions)
    
    outputFile = open(outputFileName,'w')
    outputFile.writelines(','.join(str(j) for j in i) + '\n' for i in confMatrix)
    outputFile.close()
    
    print(confMatrix)
    print('=========Confusion Matrix Built=========')

    evaluation = evaluate(labelFile, predFile)
    print(evaluation)
    print('=========Evaluation Done=========')

    models1 = []
    models1.append(('background', vidurb_answersDir + 'background.txt.bi.kn.lm'))
    models1.append(('objective', vidurb_answersDir + 'objective.txt.bi.kn.lm'))
    models1.append(('methods', vidurb_answersDir + 'methods.txt.bi.kn.lm'))
    models1.append(('results', vidurb_answersDir + 'results.txt.bi.kn.lm'))
    models1.append(('conclusions', vidurb_answersDir + 'conclusions.txt.bi.kn.lm'))

    models2 = []
    models2.append(('background', vidurb_answersDir + 'background.txt.uni.lm'))
    models2.append(('objective', vidurb_answersDir + 'objective.txt.uni.lm'))
    models2.append(('methods', vidurb_answersDir + 'methods.txt.uni.lm'))
    models2.append(('results', vidurb_answersDir + 'results.txt.uni.lm'))
    models2.append(('conclusions', vidurb_answersDir + 'conclusions.txt.uni.lm'))

    models3 = []
    models3.append(('background', vidurb_answersDir + 'background.txt.bi.lm'))
    models3.append(('objective', vidurb_answersDir + 'objective.txt.bi.lm'))
    models3.append(('methods', vidurb_answersDir + 'methods.txt.bi.lm'))
    models3.append(('results', vidurb_answersDir + 'results.txt.bi.lm'))
    models3.append(('conclusions', vidurb_answersDir + 'conclusions.txt.bi.lm'))

    modelsList = [models1, models2, models3]

    testFile = vidurb_dataDir + 'test/excerpts.txt'
    labelFile = vidurb_dataDir + 'test/labels_all.txt'
    predFile = 'vidurb_TempFile.txt'
    outputFileName = vidurb_answersDir + 'hw2_3_4.txt'
    outputFile = open(outputFileName,'w')

    for model in modelsList:
        lm_predict(model, testFile, predFile)
        evaluation = evaluate(labelFile, predFile)
        
        outputFile.write( model[0][1].split('.txt.')[1] + '\t' + str(evaluation[0]) + '\n')

        print('=========Evaluation Completed for Model=========')

    outputFile.close()

    print('=========Q3=========')
        
    """ 

#Q4
    """ 
    models = []
    models.append(('background', vidurb_answersDir + 'background.txt.uni.counts'))
    models.append(('objective', vidurb_answersDir + 'objective.txt.uni.counts'))
    models.append(('methods', vidurb_answersDir + 'methods.txt.uni.counts'))
    models.append(('results', vidurb_answersDir + 'results.txt.uni.counts'))
    models.append(('conclusions', vidurb_answersDir + 'conclusions.txt.uni.counts'))

    outputFile = vidurb_answersDir + 'vocabularySize.txt'

    analyzeLM(models,outputFile)
    """
    
#Q4.2
    """
    models = []
    models.append(('background', vidurb_answersDir + 'background.txt.uni.lm'))
    models.append(('objective', vidurb_answersDir + 'objective.txt.uni.lm'))
    models.append(('methods', vidurb_answersDir + 'methods.txt.uni.lm'))
    models.append(('results', vidurb_answersDir + 'results.txt.uni.lm'))
    models.append(('conclusions', vidurb_answersDir + 'conclusions.txt.uni.lm'))
    
    counts = []
    counts.append(('background', vidurb_answersDir + 'background.txt.uni.counts'))
    counts.append(('objective', vidurb_answersDir + 'objective.txt.uni.counts'))
    counts.append(('methods', vidurb_answersDir + 'methods.txt.uni.counts'))
    counts.append(('results', vidurb_answersDir + 'results.txt.uni.counts'))
    counts.append(('conclusions', vidurb_answersDir + 'conclusions.txt.uni.counts'))

    outputFileName = vidurb_answersDir + 'hw2_4_2.txt'
    outputFile = open(outputFileName,'w')
    
    for i in range(len(models)):
        entropy = get_entropy(models[i][1], counts[i][1])
        outputFile.write(models[i][0] + ',' + str(entropy))
        print(models[i][0] + ',' + str(entropy))

    outputFile.close()

    outputFileName = vidurb_answersDir + 'hw2_4_3.txt'
    outputFile = open(outputFileName,'w')
    
    for i in range(len(counts)):
        ttr = get_type_token_ratio(counts[i][1])
        outputFile.write(counts[i][0] + ',' + str(ttr))
        print(counts[i][0] + ',' + str(ttr))
    
    outputFile.close()

    print('=========Q4=========')
    """
