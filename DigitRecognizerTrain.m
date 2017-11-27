% Digit Recognizer Trainer
% Trains a net for Digit Recognition.
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     24/11/2017  Royi Avital
%   *   First release.
%

%% General Parameters

run('InitScript.m');

addpath(genpath('./AuxiliaryFunctions'));

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Data Parameters

dataFolderPath      = './Data/';
netFolderPath       = 'NetModels/';
trainDataFileName   = 'tTrainImage.mat';
imageNumberFileName = 'vImageNum.mat';


%% Simulation Parameters

normalizeData       = OFF;
dataAugmentation    = ON;
netLayerModelIdx    = 2;


%% Load Data

% tTrainImage
load([dataFolderPath, trainDataFileName]);
% vImageNum
load([dataFolderPath, imageNumberFileName]);


%% Test Data

numRows     = size(tTrainImage, 1);
numCols     = size(tTrainImage, 2);
numChannels = 1;
numSamples  = size(tTrainImage, 3);

numClasses = length(unique(vImageNum));

% Data Shape - Height, Width, Number of Channels, Number of Samples
mImageData = reshape(tTrainImage, [numRows, numCols, numChannels, numSamples]);
vDataClass = categorical(vImageNum);

meanVal = mean(mImageData(:));
stdVal = std(mImageData(:));

if(normalizeData == ON)
    mImageData = (mImageData - meanVal) / stdVal;
end

mTrainData = mImageData(:, :, :, 1:40000);
vTrainClass = vDataClass(1:40000);

mValidationData     = mImageData(:, :, :, 40001:42000);
vValidationClass    = vDataClass(40001:42000);

if(dataAugmentation == ON)
    imageSource = augmentedImageSource([numRows, numCols], mTrainData, vTrainClass, 'DataAugmentation', imageDataAugmenter('RandRotation', [-7.5, 7.5]));
else
    imageSource = augmentedImageSource([numRows, numCols], mTrainData, vTrainClass);
end


%% Define Network

hNetLayerModel = SelectNetLayerModel(netLayerModelIdx, numRows, numCols, numChannels);

% Pre Processing


%% Training

% trainingOptions = trainingOptions('sgdm',...
%     'MaxEpochs', 3, ...
%     'Verbose', true,...
%     'Plots', 'training-progress');

trainingOptions = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.045, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.98, ...
    'LearnRateDropPeriod', 1, ...
    'L2Regularization', 0.00001, ...
    'MaxEpochs', 500, ...
    'MiniBatchSize', 200, ...
    'Momentum', 0.65, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {mValidationData, vValidationClass}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 5000, ...
    'Verbose', true, ...
    'VerboseFrequency', 50, ...
    'Plots', 'training-progress');

[hMnistNet, sTrainInfo] = trainNetwork(imageSource, hNetLayerModel, trainingOptions);


%% Save Data

sTrainParams.subStreamNumber    = subStreamNumber;
sTrainParams.normalizeData      = normalizeData;
sTrainParams.netLayerModelIdx   = netLayerModelIdx;
sTrainParams.dataAugmentation   = dataAugmentation;
sTrainParams.meanVal            = meanVal;
sTrainParams.stdVal             = stdVal;

save([netFolderPath, 'hNetModel', num2str(netLayerModelIdx, '%03d')], 'hMnistNet', 'sTrainInfo', 'trainingOptions', 'sTrainParams');


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

