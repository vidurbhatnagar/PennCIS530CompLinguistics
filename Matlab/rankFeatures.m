function featuresXRanked = rankFeatures(featuresX, featuresXRanked, cutoffRank)

    %% Select features till cutoffRank
    [~, order] = sort(featuresXRanked(:,2));
    rankSortedX = featuresXRanked(order,:);
    topX = rankSortedX(1:cutoffRank,1);

    featuresXRanked = featuresX(:,topX);

end
