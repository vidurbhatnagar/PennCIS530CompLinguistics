%% SVM with HeldOut
posX = createPOSFeatures(trainX, posTags);
wordFX = createWordFeatures(trainX, wordLengths);

allTrainX = cell(1,1);

% allTrainX{1} = trainX(:,topWordsIndices(1:1000));
% allTrainX{2} = trainX(:,topWordsIndices(1:1500));
% allTrainX{3} = trainX(:,topWordsIndices(1:2000));
% allTrainX{4} = trainX(:,topWordsIndices(1:2500));
% allTrainX{5} = trainX(:,topWordsIndices(1:3000));

% allTrainX{1} = freqRelFeatures(trainX(:,topWordsIndices(1:1000)));
% allTrainX{2} = freqRelFeatures(trainX(:,topWordsIndices(1:1500)));
% allTrainX{3} = freqRelFeatures(trainX(:,topWordsIndices(1:2000)));
% allTrainX{4} = freqRelFeatures(trainX(:,topWordsIndices(1:2500)));
% allTrainX{5} = freqRelFeatures(trainX(:,topWordsIndices(1:3000)));

% allTrainX{1} = normFeatures(trainX(:,topWordsIndices(1:1000)));
% allTrainX{2} = normFeatures(trainX(:,topWordsIndices(1:1500)));
% allTrainX{3} = normFeatures(trainX(:,topWordsIndices(1:2000)));
% allTrainX{4} = normFeatures(trainX(:,topWordsIndices(1:2500)));
% allTrainX{5} = normFeatures(trainX(:,topWordsIndices(1:3000)));

% allTrainX{1} = stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1000);
% allTrainX{2} = stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1500);
% allTrainX{3} = stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2000);
% allTrainX{4} = stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2500);
% allTrainX{5} = stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 3000);

% allTrainX{1} = [stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1000),posX];
% allTrainX{2} = [stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1500),posX];
% allTrainX{3} = [stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2000),posX];
% allTrainX{4} = [stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2500),posX];
% allTrainX{5} = [stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 3000),posX];

% allTrainX{1} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1000),posX]);
% allTrainX{2} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1500),posX]);
% allTrainX{3} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2000),posX]);
% allTrainX{4} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2500),posX]);
% allTrainX{5} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 3000),posX]);

% allTrainX{1} = [stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1000),wordFX];
% allTrainX{2} = [stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1500),wordFX];
% allTrainX{3} = [stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2000),wordFX];
% allTrainX{4} = [stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2500),wordFX];
% allTrainX{5} = [stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 3000),wordFX];
% 
% allTrainX{1} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1000),wordFX]);
% allTrainX{2} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1500),wordFX]);
% allTrainX{3} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2000),wordFX]);
% allTrainX{4} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2500),wordFX]);
% allTrainX{5} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 3000),wordFX]);

% allTrainX{1} =  [stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1000),posX,wordFX];
% allTrainX{2} = [stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1500),posX,wordFX];
% allTrainX{3} = [stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2000),posX,wordFX];
% allTrainX{4} = [stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2500),posX,wordFX];
% allTrainX{5} = [stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 3000),posX,wordFX];

allTrainX{1} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1000),posX,wordFX]);
% allTrainX{2} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1500),posX,wordFX]);
% allTrainX{3} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2000),posX,wordFX]);
% allTrainX{4} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2500),posX,wordFX]);
% allTrainX{5} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 3000),posX,wordFX]);

selTrainY = trainY;
trainAcc = zeros(size(allTrainX,1),1);
testAcc = zeros(size(allTrainX,1),1);
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

    bestModel = fitcsvm(heldInX,heldInY);

%     models = fitcsvm(heldInX,heldInY,'kfold',10,'NumPrint',100);
%     errors = kfoldLoss(models,'mode','individual','lossfun','classiferror');
%     [~,modelIndex] = min(errors);
%     bestModel = models.Trained{modelIndex};

    trainAcc(iter) = mean(predict(bestModel,heldInX)==heldInY)
    testAcc(iter) = mean(predict(bestModel,heldOutX)==heldOutY)
end