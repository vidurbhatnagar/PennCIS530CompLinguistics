function featuresX = normFeatures(featuresX)
    tic
    numCols = size(featuresX,2);
    l2X = sqrt(sum(featuresX.^2,2));
    for iter = 1:numCols
        featuresX(:,iter) = featuresX(:,iter)./l2X;
    end
    toc
end