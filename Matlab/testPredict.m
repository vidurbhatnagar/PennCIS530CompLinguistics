colIndices = 1:5000;
load('Model1_SVM_5K.mat')
testX = test5000X(:,colIndices);
predictions1 = predict(bestModel,testX);

load('Model2_SVM_5KStem.mat')
testX = testStem5000X(:,colIndices);
predictions2 = predict(bestModel,testX);

load('Model3_SVM_5K18.mat')
testX = [test5000X(:,colIndices),testPosFX,testWordFX,testSentFX];
predictions3 = predict(bestModel,testX);

load('Model4_SVM_5KNorm.mat')
testX = normFeatures(test5000X(:,colIndices));
predictions4 = predict(bestModel,testX);

load('Model5_SVM_5KStem18.mat')
testX = [testStem5000X(:,colIndices),testPosFX,testWordFX,testSentFX];
predictions5 = predict(bestModel,testX);

load('model6_SVM_18STD.mat')
testX = standardizeFeatures([testPosFX,testWordFX,testSentFX]);
predictions6 = predict(bestModel,testX);

colIndices = 1:10000;
load('model7_SVM_10KBi.mat')
testX = testX_bigrams(:,colIndices);
predictions7 = predict(bestModel,testX);

colIndices = 1:163;
load('model8_SVM_163Bi3.mat')
testX = testX_postag_bigrams_uni(:,colIndices);
predictions8 = predict(bestModel,testX);

colIndices = 1:5000;
load('model9_Logit_5K18.mat')
testX = [testStem5000X(:,colIndices),testPosFX,testWordFX,testSentFX];
predictions9 = predict(bestModel,testX);

colIndices = 1:4640;
load('model10_Logit_Chi.mat')
testX = testChiX(:,colIndices);
predictions10 = predict(bestModel,testX);

load('model11_SVM_Chi.mat')
testX = testChiX(:,colIndices);
predictions11 = predict(bestModel,testX);

load('model12_Logit_Chi18.mat')
testX = [testChiX(:,colIndices),testPosFX,testWordFX,testSentFX];
predictions12 = predict(bestModel,testX);

predictions = mode([predictions1,predictions2,predictions3,predictions4,...
               predictions5,predictions6,predictions7,predictions8,...
               predictions9,predictions10, predictions11, predictions12],2);