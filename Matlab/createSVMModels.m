%% SVM with HeldOut
numX = [1000,1500,2000,3000];
numX = [4640];
allTrainX = cell(length(numX),1);

for iter = 1:length(numX)
    colIndices = 1:numX(iter);

    allTrainX{iter} = trainChiX(:,colIndices);    
%     allTrainX{iter} = (trainX_postag_bigrams_uni(:,colIndices));
%     allTrainX{iter} = [trainPosFX,trainWordFX,trainSentFX];
%     allTrainX{iter} = standardizeFeatures([train5000X(:,colIndices),trainPosFX,trainWordFX,trainSentFX]);
    
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

%Stratified CVPartition
cvPartition = cvpartition(trainY,'Holdout',.20);
heldInIndices = training(cvPartition);
heldOutIndices = test(cvPartition);

trainAcc = zeros(length(numX),1);
testAcc = zeros(length(numX),1);

for iter = 1:length(numX)
    selTrainY = trainY;
    selTrainX = allTrainX{iter};
    
    heldInX = selTrainX(heldInIndices,:);
    heldInY = selTrainY(heldInIndices,:);
    heldOutX = selTrainX(heldOutIndices,:);
    heldOutY = selTrainY(heldOutIndices,:);

    bestModel = fitcsvm(heldInX,heldInY,'KernelScale','auto','Standardize',true,'Verbose',1, 'NumPrint',1000, 'BoxConstraint', 0.7);
    
%     models = fitcsvm(heldInX,heldInY,'kfold',10,'NumPrint',100);
%     errors = kfoldLoss(models,'mode','individual','lossfun','classiferror');
%     [~,modelIndex] = min(errors);
%     bestModel = models.Trained{modelIndex};

    trainAcc(iter) = mean(predict(bestModel,heldInX)==heldInY)
    testAcc(iter) = mean(predict(bestModel,heldOutX)==heldOutY)
end
