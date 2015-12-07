function featuresXStemRanked = stemFeaturesWithRanking(featuresX, wordsActual, rankSortedX, cutoffRank)

    %% Select features till cutoffRank
    topX = rankSortedX(1:cutoffRank,1);

    %% Stemming each word in wordsActual
    wordsStemmed = cell(size(wordsActual));
    for iter = 1:length(wordsActual)
        wordsStemmed{iter} = porterStemmer(wordsActual{iter});
    end

    %% Findind duplicate indices after stemming
    [~,uniqueIndices] = unique(wordsStemmed);
    duplicateIndices = setdiff(1:length(wordsStemmed), uniqueIndices)';

    %% Collapsing Counts and Removing duplicates
    featuresXStemRanked = featuresX;
    for iter= 1:length(duplicateIndices)
        % Find all common indices to this duplicate entry
        commonIndices = find(strcmp(wordsStemmed, wordsStemmed{duplicateIndices(iter)}));

        % Check if any of the commonIndices is in topX
        if sum(ismember(commonIndices,topX)) > 0
            % Sum all entries in the first common index and set all other common
            % indices to 0 so that they don't sum up again
            featuresXStemRanked(:,commonIndices(1)) = sum(featuresXStemRanked(:,commonIndices),2);
            featuresXStemRanked(:,commonIndices(2:end)) = 0;

            % Add the first common index to vTopFeatures, so as to prevent it
            % from getting deleted
            topX(end+1) = commonIndices(1);
        end
    end

    % Set non-top feature indices to -1
    nonTopX = setdiff(1:size(rankSortedX,1), topX)';
    featuresXStemRanked(:,nonTopX) = -1;

    % Set duplicate indices to -1
    featuresXStemRanked(:,duplicateIndices) = -1;

    % Delete all -1 indices
    featuresXStemRanked(:,find(featuresXStemRanked(1,:)==-1)) = [];

end
