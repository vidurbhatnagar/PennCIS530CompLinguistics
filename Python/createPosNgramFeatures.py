#-----------------------------------------------------------------------
#IMPORTS
from __future__ import division
from nltk.corpus import wordnet
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
from nltk import word_tokenize,sent_tokenize,FreqDist, ConditionalFreqDist
from nltk import pos_tag, map_tag
from itertools import izip
from math import sqrt
from math import log
import subprocess
import itertools
import numpy
import sys
import os

def get_pos_tag_dict(fileLines):
	pos_tag_dict = {};
	tokens = [];
	for excerpt in fileLines:
        	for sentence in excerpt:
                        tokens.extend(sentence);

	tokens = list(set(tokens));	
	pos_tags = pos_tag(tokens);
        pos_tag_dict = {pos_tuple[0]:pos_tuple[1] for pos_tuple in pos_tags};	
	
	return pos_tag_dict;
	
 
def get_pos_tags(excerpt,pos_tag_dict):
    posTags = [];
    for sentence in excerpt:
	posTags.append([pos_tag_dict[word] for word in sentence]);	

    #mapper = dict([('ADJ', 1), ('ADP', 2), ('ADV', 3), ('CONJ', 4), ('DET', 5), ('NOUN', 6), ('NUM', 7), ('PRT', 8), ('PRON', 9), ('VERB', 10), ('.', 11), ('X', 12)])
    #simplifiedTags = [mapper[map_tag('en-ptb', 'universal', tag)] for word, tag in posTags]

    return posTags;

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
				

def get_ngram_counts_per_excerpt(excerpt,n,pos_tag_dict): 
        # get n gram counts per excerpt
	tokens = [];
	startTokens = ['<s>' for i in range(n-1)];
        endTokens = ['</s>' for i in range(n-1)]; 

	taggedExcerpt = get_pos_tags(excerpt,pos_tag_dict);
	#print taggedExcerpt;
	for sentence in taggedExcerpt:
		tokens.extend(startTokens + sentence + endTokens);

	#for i in range(len(tokens)):
	#	if ( tokens[i] in unk_dict):
	#		tokens[i] = 'UNK';	

	# Create bigram list
        ngrams = make_ngram_tuples(tokens,n);	

        # Record counts
        cnt_ngrams = FreqDist(ngrams)
        return cnt_ngrams;

def get_ngrams(fileLines, n, pos_tag_dict):
    # Get n gram counts for corpus
    tokens = [];
    ngram_counts = FreqDist();
    for excerpt in fileLines:	 	
	ngram_counts_exp = get_ngram_counts_per_excerpt(excerpt,n,pos_tag_dict);	
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
def createFeatureVectors(fileLines, featureSpaceDict, outputFileName, n, pos_tag_dict):
    outputFile = open(outputFileName,'wb')
 
    for excerpt in fileLines:
	featureVector = [0] * len(featureSpaceDict)  	
	n_gram_counts = get_ngram_counts_per_excerpt(excerpt,n,pos_tag_dict);		

        for n_gram in n_gram_counts: 
		#print str(n_gram) + '\t' + str(n_gram_counts[n_gram]);
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

    #minOccurences = 5;
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

    ##################################---------> Pos tagged Ngrams 
   
    pos_tag_dict = get_pos_tag_dict(allFileLines); 

    # Get n_gram counts
    # n = n gram
    n_gram_counts = get_ngrams(allFileLines,ngramNum,pos_tag_dict);
    for n_gram in n_gram_counts:
	print str(n_gram) + ' ' + str(n_gram_counts[n_gram]);

    # Get n_grams for creating feature space
    # top n ngrams
    n_grams_top = getTopNgrams(n_gram_counts,featureSpaceLimit);

    # get Feature Space
    featureSpaceDict = createFeatureSpace(n_grams_top);     
    print featureSpaceDict;    

    #featureSpaceList = [None] * len(featureSpaceDict) 
    #for key,value in featureSpaceDict.items():
    # 	featureSpaceList[int(value)] = key
    outputFileName = 'n_grams_postags.txt'
    saveNgramsToFile(outputFileName,n_grams_top); 
    print("=====FEATURE SPACE CREATED=====")
    # Create feature space
    # Save Top n-grams indices
    #n_gram_top_indices = getTopNgramIndices(n_grams_top,featureSpaceDict);
    #outputFileName = 'topNgramIndices.txt'
    #saveListToFile(outputFileName,n_gram_top_indices);
	
    featureVectorList = createFeatureVectors(trainFileLines, featureSpaceDict,'trainX.txt', ngramNum,pos_tag_dict) 
    print("=====TRAIN FEATURE VECTORS CREATED=====")
 
    featureVectorList = createFeatureVectors(testFileLines, featureSpaceDict,'testX.txt', ngramNum,pos_tag_dict)
    print("=====TEST FEATURE VECTORS CREATED=====")

