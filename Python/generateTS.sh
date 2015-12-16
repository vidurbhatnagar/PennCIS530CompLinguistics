cd  /home1/v/vidurb/CL530/Project/Python/TSConfigFiles/
rm *

cd /home1/v/vidurb/CL530/Project/Python/dev_input
for entry in *; do
    echo "==== Do not change these values ====
stopFilePath = stoplist-smart-sys.txt
performStemming = N
backgroundCorpusFreqCounts = bgCounts-Giga.txt
topicWordCutoff = 0.1

==== Directory to compute topic words on ====

inputDir = /home1/v/vidurb/CL530/Project/Python/dev_input/$entry

==== Output File ====
outputFile = /home1/v/vidurb/CL530/Project/Python/TSFiles/$entry.ts" >> /home1/v/vidurb/CL530/Project/Python/TSConfigFiles/$entry

done

cd /home1/v/vidurb/CL530/Project/Python/TSConfigFiles
pwdName=`pwd`
configFiles=`ls`

cd /home1/c/cis530/hw4/TopicWords-v2/
for entry in $configFiles; do
    echo "TS for " $entry
    configFile=$pwdName"/"$entry
    java -Xmx1000m TopicSignatures $configFile >> /home1/v/vidurb/CL530/Project/Python/dump.txt
done
