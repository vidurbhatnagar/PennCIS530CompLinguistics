%% This function returns relative freq data per observation i.e row wise

function featuresXFreqRel = freqRelFeatures(featuresX)
    numCols = size(featuresX,2);

    % sum in rows, since each row depicts a user
    sumX = sum(featuresX,2);
    featuresXFreqRel = featuresX./repmat(sumX,[1,numCols]);
end