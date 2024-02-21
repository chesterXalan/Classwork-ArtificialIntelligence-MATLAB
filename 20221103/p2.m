clc; clear; close all
filenameImagesTrain = 'train-images-idx3-ubyte.gz';
filenameLabelsTrain = 'train-labels-idx1-ubyte.gz';
XTrain = processImagesMNIST(filenameImagesTrain);
YTrain = processLabelsMNIST(filenameLabelsTrain);

X(:, :, 1, :) = XTrain(:, :, 1, 1:2000);
X = double(X);

D(:, 1) = YTrain(1:2000, 1);
D = double(D);
D = categorical(D);

layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(9, 9, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, 'MaxEpochs', 10, ...
    'ExecutionEnvironment', 'gpu', 'Plots', 'training-progress');
net = trainNetwork(X, D, layers, options);

X(:, :, 1, :) = XTrain(:, :, 1, 2001:4000);
X = double(X);

D(:, 1) = YTrain(2001:4000, 1);
D = double(D);
D = categorical(D);

YPred = classify(net, X);
accuracy = sum(YPred == D)/numel(D);
[classfn, score] = classify(net, X(:, :, 1, 3));

figure, imshow(X(:, :, 1, 3))
title(['Class = ' int2str(int8(classfn)-1)])
map = gradCAM(net, X(:, :, 1, 3), classfn);

figure, imshow(X(:, :, 1, 3)); hold on
imagesc(map, 'AlphaData', 0.5); colormap jet
hold off; title('Grad-CAM')
