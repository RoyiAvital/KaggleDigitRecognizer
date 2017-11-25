% Digit Recognizer - Create Submission
% References:
%   1.  https://www.kaggle.com/c/facial-keypoints-detection/discussion/4960
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

dataFolderPath      = 'Data/';
testDataFileName    = 'tTestImage.mat';
netFolderPath       = 'NetModels/';
netModelFileName    = 'hNetModel002.mat';

fileName        = 'SubmissionData.csv';
headerRowString = 'ImageId,Label';


%% Loading Data

load([dataFolderPath, testDataFileName]); %<! tTestImage
load([netFolderPath, netModelFileName]); %<! hMnistNet, sTrainInfo, trainingOptions, sTrainParams

normalizeData       = sTrainParams.normalizeData;
netLayerModelIdx    = sTrainParams.netLayerModelIdx;
meanVal             = sTrainParams.meanVal;
stdVal              = sTrainParams.stdVal;


%% Data Parameters

numRows     = size(tTestImage, 1);
numCols     = size(tTestImage, 2);
numChannels = 1;
numSamples  = size(tTestImage, 3);

% Data Shape - Height, Width, Number of Channels, Number of Samples
mImageData = reshape(tTestImage, [numRows, numCols, numChannels, numSamples]);

if(normalizeData == ON)
    mImageData = (mImageData - meanVal) / stdVal;
end


%% Classification

vPredictedClass = classify(hMnistNet, mImageData);


%% Writing Data

hFileId = fopen([dataFolderPath, fileName], 'w');
fprintf(hFileId, [headerRowString, '\n']);

for ii = 1:numSamples    
    fprintf(hFileId, [num2str(ii), ',']);
    fprintf(hFileId, [num2str(uint32(vPredictedClass(ii)) - 1), '\n']);
end

fclose(hFileId);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

