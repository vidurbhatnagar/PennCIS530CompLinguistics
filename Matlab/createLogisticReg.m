numX = [1000,1500,2000,3000];
numX = [3000];
allTrainX = cell(length(numX),1);

for iter = 1:length(numX)
    colIndices = 1:numX(iter);

%     allTrainX{iter} = (trainX_postag_bigrams_uni(:,colIndices));
%     allTrainX{iter} = [trainPosFX,trainWordFX,trainSentFX];
        allTrainX{iter} = standardizeFeatures([trainStem5000X(:,colIndices),trainPosFX,trainWordFX,trainSentFX]);
    
%     colIndices = 1:numX(iter);
%     allTrainX{iter} = trainStem5000X(:,colIndices);    
%     allTrainX{iter} = [trainStem5000X(:,colIndices),trainPosFX,trainWordFX,trainSentFX];
%     allTrainX{iter} = standardizeFeatures([trainStem5000X(:,colIndices),trainPosFX,trainWordFX,trainSentFX]);

%     allTrainX{iter} = [trainPosFX,trainWordFX,trainSentFX];
%     allTrainX{iter} = standardizeFeatures([trainPosFX,trainWordFX,trainSentFX]);
    
%     colIndices = tfidfTopIndices(1:numX(iter));
%     rowIndices = 1:size(trainY,1);
%     allTrainX{iter} = standardizeFeatures([...
%                             tfidfX(rowIndices,colIndices),...
%                             trainPosFX,trainWordFX,trainSentFX]);
%     allTrainX{iter} = allX;
end

% mnrfit needs labels to be positive integers
selTrainY = trainY;
selTrainY(selTrainY==1) = 2;
selTrainY(selTrainY==0) = 1;

trainAcc = zeros(size(allTrainX,1),1);
testAcc = zeros(size(allTrainX,1),1);

%% Divide dataset into 2 parts - held-in for K-folds model, held-out for testing
for iter = 1:size(allTrainX,1)
    selTrainX = allTrainX{iter};
    
    %Stratified CVPartition
    cvPartition = cvpartition(selTrainY,'Holdout',.20);
    heldInIndices = training(cvPartition,1);
    heldOutIndices = test(cvPartition,1);
    heldInX = selTrainX(heldInIndices,:);
    heldInY = selTrainY(heldInIndices,:);
    heldOutX = selTrainX(heldOutIndices,:);
    heldOutY = selTrainY(heldOutIndices,:);
    
    bestModel = mnrfit(heldInX,heldInY);
    bestModel(isnan(bestModel)) = 0;

    [~,predicted] = max(mnrval(bestModel,heldInX),[],2);
    trainAcc(iter) = mean(predicted==heldInY)
    [~,predicted] = max(mnrval(bestModel,heldOutX),[],2);
    testAcc(iter) = mean(predicted==heldOutY)
end



