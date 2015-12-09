%% SVM with HeldOut
numX = [1000,1500,2000,3000];
numX = [2000];
allTrainX = cell(length(numX),1);

for iter = 1:length(numX)
    colIndices = 1:numX(iter);

%     allTrainX{iter} = train5000X(:,colIndices);
%     allTrainX{iter} = [train5000X(:,colIndices),trainPosFX,trainWordFX,trainSentFX];
%     allTrainX{iter} = standardizeFeatures([train5000X(:,colIndices),trainPosFX,trainWordFX,trainSentFX]);
    
%     colIndices = 1:numX(iter);
%     allTrainX{iter} = trainStem5000X(:,colIndices);    
%     allTrainX{iter} = [trainStem5000X(:,colIndices),trainPosFX,trainWordFX,trainSentFX];
%     allTrainX{iter} = standardizeFeatures([trainStem5000X(:,colIndices),trainPosFX,trainWordFX,trainSentFX]);

%     allTrainX{iter} = [trainPosFX,trainWordFX,trainSentFX];
%     allTrainX{iter} = standardizeFeatures([trainPosFX,trainWordFX,trainSentFX]);
    
    colIndices = tfidfTopIndices(1:numX(iter));
    rowIndices = 1:size(trainX,1);
    allTrainX{iter} = standardizeFeatures([...
                            tfidfX(rowIndices,colIndices),...
                            trainPosFX,trainWordFX,trainSentFX]);
end

%Stratified CVPartition
cvPartition = cvpartition(trainY,'Holdout',.20);
heldInIndices = training(cvPartition,1);
heldOutIndices = test(cvPartition,1);

trainAcc = zeros(length(numX),1);
testAcc = zeros(length(numX),1);

for iter = 1:length(numX)
    selTrainY = trainY;
    selTrainX = allTrainX{iter};
    
    heldInX = selTrainX(heldInIndices,:);
    heldInY = selTrainY(heldInIndices,:);
    heldOutX = selTrainX(heldOutIndices,:);
    heldOutY = selTrainY(heldOutIndices,:);

    bestModel = fitcsvm(heldInX,heldInY,'NumPrint',100);
    
%     models = fitcsvm(heldInX,heldInY,'kfold',10,'NumPrint',100);
%     errors = kfoldLoss(models,'mode','individual','lossfun','classiferror');
%     [~,modelIndex] = min(errors);
%     bestModel = models.Trained{modelIndex};

    trainAcc(iter) = mean(predict(bestModel,heldInX)==heldInY)
    testAcc(iter) = mean(predict(bestModel,heldOutX)==heldOutY)
end
