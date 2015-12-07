function posX = createPOSFeatures(featuresX, posTags)
    % 12 POS :: 
    % ADJ, ADP, ADV, CONJ, DET, NOUN, NUM, PRT, PRON, VERB, "." , X
    
    posX = zeros(size(featuresX,1),max(posTags));
    
    for iter = 1:max(posTags)
        posX(:,iter) = sum(featuresX(:,posTags==iter),2);
    end
%     [coeff,score,latent,tsquared,explained,mu] = pca(posX);
%     pcaPosX = score(:,1:3);
end