function featuresXStemRanked = stemFeaturesWithRanking(featuresX, wordsActual, rankSortedX, numX)
    %% Get the top indices 
    topX = rankSortedX(1:numX);
   
    %% Stemming each word in wordsActual
    wordsStemmed = cell(size(wordsActual));
    for iter = 1:length(wordsActual)
        wordsStemmed{iter} = porterStemmer(wordsActual{iter});
    end
    
    %% Collapsing Counts and Removing duplicates
    featuresXStemRanked = zeros(size(featuresX,1),numX);
    for iter= 1:numX
        % Find all common indices to this word
        commonIndices = find(strcmp(wordsStemmed, wordsStemmed{topX(iter)}));

        % Sum all entries in the topX(iter) column
        featuresXStemRanked(:,iter) = sum(featuresX(:,commonIndices),2);
    end

end
