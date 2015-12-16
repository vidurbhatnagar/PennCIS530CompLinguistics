#-----------------------------------------------------------------------
#IMPORTS
from __future__ import division
from nltk.corpus import wordnet
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
from nltk import word_tokenize,sent_tokenize,FreqDist, ConditionalFreqDist
from itertools import izip
from math import sqrt
from math import log
import subprocess
import itertools
import numpy
import sys
import os

def make_ngram_tuples(word_list,n):	
	return [tuple(word_list[x:x+n]) for x in xrange(len(word_list)-1)];

def get_unk_dict(fileLines,minWordCount):
	# get dict for unknowns
	tokens = [];
	unk_set = set();
	for excerpt in fileLines:
		for sentence in excerpt:
			tokens.extend(sentence);
	
	tok_counts = FreqDist(tokens);

	for token in tok_counts:
		if ( tok_counts[token] <= minWordCount ):
			unk_set.add(token);
	return unk_set;	
				

def get_ngram_counts_per_excerpt(excerpt,n,unk_dict): 
        # get n gram counts per excerpt
	tokens = [];
	startTokens = ['<s>' for i in range(n-1)];
        endTokens = ['</s>' for i in range(n-1)]; 

	for sentence in excerpt:
		tokens.extend(startTokens + sentence + endTokens);

	for i in range(len(tokens)):
		if ( tokens[i] in unk_dict):
			tokens[i] = 'UNK';	

	# Create bigram list
        ngrams = make_ngram_tuples(tokens,n);	

        # Record counts
        cnt_ngrams = FreqDist(ngrams)
        return cnt_ngrams;

def get_ngrams(fileLines, n, unk_dict):
    # Get n gram counts for corpus
    tokens = [];
    ngram_counts = FreqDist();
    for excerpt in fileLines:	 
	ngram_counts_exp = get_ngram_counts_per_excerpt(excerpt,n,unk_dict);	
	for ngram in ngram_counts_exp:
        	if ( ngram in ngram_counts ):
			val = ngram_counts[ngram];
		else:	
			val = 0;	
		ngram_counts[ngram] = val + ngram_counts_exp[ngram]; 
			
    ngram_counts.update(ngram_counts_exp);
 
    return ngram_counts;

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def getTopWordsIndices(fileLines, featureSpaceDict, limit):    
    topWordsDict = OrderedDict(Counter(flatten(fileLines)).most_common(limit))
    topWordsIndices = []

    for key in topWordsDict:
        topWordsIndices.append(featureSpaceDict[key]+1)
          
    return topWordsIndices

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#UTIL FUNCTIONS
def createFeatureVectors(fileLines, featureSpaceDict, outputFileName, n, unk_dict):
    outputFile = open(outputFileName,'wb')
 
    for excerpt in fileLines:
	featureVector = [0] * len(featureSpaceDict) 
        n_gram_counts = get_ngram_counts_per_excerpt(excerpt,n, unk_dict);		

        for n_gram in n_gram_counts: 
		if ( n_gram in featureSpaceDict ):			
			#print str(n_gram) + '\t' + str(n_gram_counts[n_gram]);
                	#print 'In!!!';
			#print str(featureSpaceDict[n_gram]);
			featureVector[int(featureSpaceDict[n_gram])] = n_gram_counts[n_gram]
	    
        outputFile.write(unicode(str(featureVector).replace('[','').replace(']','')).encode('utf8'))
        outputFile.write('\n')
        
    outputFile.close()        
    return True

def getTopNgrams(n_gram_counts,limit):
   n_grams_top = n_gram_counts.most_common(limit); 
   return n_grams_top;

def getTopNgramIndices(n_grams_top,featureSpaceDict):
    topNgramIndices = [];
    for ngram in n_grams_top:
        topNgramIndices.append(featureSpaceDict[ngram]+1)

    return topNgramIndices;
	
def createFeatureSpace(n_gram_counts):
    n_grams = [n_gram_count[0] for n_gram_count in n_gram_counts];
    return OrderedDict({x:i for i,x in enumerate(n_grams)});

def  wordTokenize(line):
   return word_tokenize(line.lower())

def flatten(listoflists):
   return list(itertools.chain.from_iterable(listoflists))

def loadFileLines(fileName):
    fileInstance = open(fileName)
    fileLines = []
    fileTags = []
     
    while True:
	fileSentences = [];
        
	line = fileInstance.readline().decode('utf8') 
	if not line:
            fileInstance.close()
            break

        lineParts = line.split("\t")
	sentences = sent_tokenize(lineParts[0].strip());
	
	for sentence in sentences:
		fileSentences.append(wordTokenize(sentence)); 	

	fileLines.append(fileSentences);
        fileTags.append(lineParts[-1].strip())
     
    return fileLines, fileTags

def loadDirectory(dirName):
    dirFiles = os.listdir(diName)
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

def saveNgramsToFile(fileName, listName):
    outputFile = open(fileName,'wb')

    for key in listName:	
        outputFile.write(str(unicode(key[0][0]).encode('utf8')) + ',' + str(unicode(key[0][1]).encode('utf8')) + '\t' + str(key[1]));
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
        print str(key) + " : " + str(value)

        if(itr == limit):
            break
        
    return


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#MAIN - TESTING THE CODE FROM MAIN

if __name__ == "__main__":
    print("=====ASSIGNMENT STARTED=====")

    minOccurences = 5;
    ngramNum = 2;
    featureSpaceLimit = 10000;

    trainFileName = 'project_articles_train'
    testFileName = 'project_articles_test'
   
    #trainFileName = 'trainDummy'; 
    trainFileLines, trainFileTags = loadFileLines(trainFileName)
    #outputFileName = 'trainLabels.txt'
    #saveListToFile(outputFileName,trainFileTags);
    print("=====TRAINING DATA LOADED=====")

    #testFileName = 'testDummy' 
    testFileLines, testFileTags = loadFileLines(testFileName)
    print("=====TESTING DATA LOADED=====")

    allFileLines = []
    allFileLines.extend(trainFileLines)
    allFileLines.extend(testFileLines)

    # Get UNK dict
    # Minimum word occurences
    unk_dict = get_unk_dict(allFileLines,minOccurences);  

    # Get n_gram counts
    # n = n gram
    n_gram_counts = get_ngrams(allFileLines,ngramNum,unk_dict);
	
    # Get n_grams for creating feature space
    # top n ngrams	 
    n_grams_top = getTopNgrams(n_gram_counts,featureSpaceLimit); 

    # Create feature space
    featureSpaceDict = createFeatureSpace(n_grams_top);     
    
    #featureSpaceList = [None] * len(featureSpaceDict) 
    #for key,value in featureSpaceDict.items():
    # 	featureSpaceList[int(value)] = key
    outputFileName = 'n_grams.txt'
    saveNgramsToFile(outputFileName,n_grams_top);
    print("=====FEATURE SPACE CREATED=====")

    # Save Top n-grams indices
    #n_gram_top_indices = getTopNgramIndices(n_grams_top,featureSpaceDict);
    #outputFileName = 'topNgramIndices.txt'
    #saveListToFile(outputFileName,n_gram_top_indices);

    featureVectorList = createFeatureVectors(trainFileLines, featureSpaceDict, 'trainX.txt', ngramNum, unk_dict)
    print("=====TRAIN FEATURE VECTORS CREATED=====")
 
    featureVectorList = createFeatureVectors(testFileLines, featureSpaceDict,'testX.txt', ngramNum, unk_dict)
    print("=====TEST FEATURE VECTORS CREATED=====")

