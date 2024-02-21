clc; clear; close all
filenameImagesTrain = 'train-images-idx3-ubyte.gz';
filenameLabelsTrain = 'train-labels-idx1-ubyte.gz';
XTrain = processImagesMNIST(filenameImagesTrain);
YTrain = processLabelsMNIST(filenameLabelsTrain);

X1(:, :, :) = XTrain(:, :, 1, 1:2000);
D1(:, 1) = YTrain(1:2000, 1);
D1 = double(D1); D1(D1 == 0) = 10;

W1 = 1e-2*randn([9 9 20]);
W5 = (2*rand(100, 2000)-1)*sqrt(6)/sqrt(360+2000);
Wo = (2*rand(10, 100)-1)*sqrt(6)/sqrt(10+100);

% Train
for epoch = 1:10
    disp(['epoch ' num2str(epoch)]);
    [W1, W5, Wo] = MnistConv(W1, W5, Wo, X1, D1);
end

% Test
X2(:, :, :) = XTrain(:, :, 1, 2001:3000);
D2(:, 1) = YTrain(2001:3000, 1);
D2 = double(D2); D2(D2 == 0) = 10;

acc = 0;
N = length(D2);
for k = 1:N
    x = X2(:, :, k); % Input, 28x28
    y1 = Conv(x, W1); % Convolution, 20x20x20
    y2 = ReLU(y1);
    y3 = Pool(y2); % Pool, 10x10x10
    y4 = reshape(y3, [], 1); % 2000
    v5 = W5*y4; % ReLU, 360
    y5 = ReLU(v5);
    v = Wo*y5; % Softmax, 10
    y = Softmax(v);
    [~, i] = max(y);
    if i == D2(k)
        acc = acc+1;
    end
end

acc = acc/N;
disp(['Accuracy is ' num2str(acc)]);
