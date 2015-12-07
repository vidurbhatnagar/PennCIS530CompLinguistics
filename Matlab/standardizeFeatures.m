%% This function normalizes data per feature i.e column wise
% Use it when combining different feature types - 
% like words and image features

function featuresXNormalized = standardizeFeatures(featuresX)
    numSamples = size(featuresX,1);
    meanX = repmat(mean(featuresX),[numSamples,1]);
    stdDevX = repmat(std(featuresX),[numSamples,1]);
    
    featuresXNormalized = (featuresX-meanX)./stdDevX;
    featuresXNormalized(isnan(featuresXNormalized)) = 0;
end
