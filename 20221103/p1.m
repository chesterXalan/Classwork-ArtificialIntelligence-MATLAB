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

figure
perm = randperm(2000, 20);
for i = 1:20
    subplot(4, 5, i);
    imshow(X(:, :, perm(i)));
end
shg

layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(9, 9, 'Padding', 'same')
    reluLayer
    averagePooling2dLayer(2)
    fullyConnectedLayer(100)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, 'MaxEpochs', 10, ...
    'ExecutionEnvironment', 'gpu', 'Plots', 'training-progress');
net = trainNetwork(X, D, layers, options);

YPred = classify(net, X);
accuracy = sum(YPred == D)/numel(D);
