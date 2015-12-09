% normTf = normFeatures([trainX;testX]);

% Sum normTf over entire corpus
% sumNormTf = sum(normTf);

% Use sumNormTf to get ranking
tfidf = sumNormTf .* wordIdfs;
[~,tfidfTopIndices] = sort(tfidf,'descend');

% Now change counts (tf) in the vector space to normTF * wordIdfs counts
numRows = size(normTf,1);
tfidfX = zeros(size(normTf));

for iter = 1:numRows
        tfidfX(iter,:) = normTf(iter,:).* wordIdfs;
end