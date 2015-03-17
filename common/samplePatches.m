function patches = samplePatches(rawImages, patchSize, numPatches)
% rawImages is of size imWidth*imHeight by numImages
% We assume that imWidth = imHeight
imWidth = sqrt(size(rawImages,1));
imHeight = imWidth;
numImages = size(rawImages,2);
rawImages = reshape(rawImages,imWidth,imHeight,numImages); 

% Initialize patches with zeros.  
patches = zeros(patchSize*patchSize, numPatches);

% Maximum possible starting coordinate
maxWidth = imWidth - patchSize + 1;
maxHeight = imHeight - patchSize + 1;

% Sample!
for num = 1:numPatches
    x = randi(maxHeight);
    y = randi(maxWidth);
    imInd = randi(numImages);
    p = rawImages(x:x+patchSize-1,y:y+patchSize-1, imInd);
    patches(:,num) = p(:);
end
end