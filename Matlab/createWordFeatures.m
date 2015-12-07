function wordFeatX = createWordFeatures(featuresX, wordLengths)
    % Total number of characters (C)
    % Total number of words (N)
    % Average length per word (in characters)
    % Vocabulary richness (total different words/N)
    
    numRows = size(featuresX,1);
    sumChars = sum(featuresX.*repmat(wordLengths,numRows,1),2);
    sumWords = sum(featuresX,2);
    avgLen = sumChars./sumWords;
    vocabRichness = sum(sign(featuresX),2)./sumWords;
    
    wordFeatX = zeros(numRows,4);
    wordFeatX(:,1) = sumChars;
    wordFeatX(:,2) = sumWords;
    wordFeatX(:,3) = avgLen;
    wordFeatX(:,4) = vocabRichness;
    
end