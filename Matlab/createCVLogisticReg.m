posX = createPOSFeatures(trainX, posTags);
wordFX = createWordFeatures(trainX, wordLengths);

allTrainX = cell(1,1);

allTrainX{1} = trainX(:,topWordsIndices(1:1000));
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

% allTrainX{1} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1000),posX,wordFX]);
% allTrainX{2} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 1500),posX,wordFX]);
% allTrainX{3} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2000),posX,wordFX]);
% allTrainX{4} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 2500),posX,wordFX]);
% allTrainX{5} = standardizeFeatures([stemFeaturesWithRanking(trainX, actualWords, topWordsIndices, 3000),posX,wordFX]);

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
    
    numFolds = 10;
    cvPartition = cvpartition(heldInY,'KFold',numFolds);
    models = cell(numFolds,1);
    errors = zeros(numFolds,1);

    for iterK = 1:numFolds
        kfTrainIndices = training(cvPartition,iterK);
        kfTestIndices = test(cvPartition,iterK);
        kfTrainX = heldInX(kfTrainIndices,:);
        kfTrainY = heldInY(kfTrainIndices);
        kfTestX = heldInX(kfTestIndices,:);
        kfTestY = heldInY(kfTestIndices);

        models{iterK} = mnrfit(kfTrainX,kfTrainY);
        models{iterK}(isnan(models{iterK})) = 0;

        [~,predicted] = max(mnrval(models{iterK},kfTestX),[],2);
        errors(iterK) = mean(predicted~=kfTestY);
    end
    
    [~,modelIndex] = min(errors);
    model = models{modelIndex};

    [~,predicted] = max(mnrval(model,heldInX),[],2);
    trainAcc(iter) = mean(predicted==heldInY)
    [~,predicted] = max(mnrval(model,heldOutX),[],2);
    testAcc(iter) = mean(predicted==heldOutY)
end



