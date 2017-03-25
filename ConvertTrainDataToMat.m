% Convert Data to MAT
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     23/03/2017  Royi Avital
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
trainDataFileName   = 'train.csv';

numRows     = 28;
numCols     = 28;

imageDataIdx = 2; %<! X, Y per Features, Image Data


%% Training Data

numPx = numRows * numCols;

cRawData    = csvimport([dataFolderPath, trainDataFileName]);
numImages   = size(cRawData, 1) - 1; %<! Header row

tTrainImage = zeros([numRows, numCols, numImages], 'single'); %<! Data is in UINT8
vImageNum   = zeros([numImages, 1], 'single');

runTime = 0;
for ii = 1:numImages
    hProcImageTimer = tic();
    imageIdx = ii + 1;
    for jj = 1:(numPx + 1)
        if(jj == 1)
            vImageNum(ii) = single(cRawData{imageIdx, jj});
            continue;
        end
        pxIdx = jj - 1;
        [iRow, jCol] = ind2sub([numRows, numCols], pxIdx);
        tTrainImage(jCol, iRow, ii) = single(cRawData{imageIdx, pxIdx} / 255); %<! Data is Row Wise
    end
    procImagTime = toc(hProcImageTimer);
    runTime = runTime + procImagTime;
    disp(['Finished processing Image #', num2str(ii, '%04d'), ' out of ', num2str(numImages), ' images']);
    disp(['Processig Time       - ', num2str(procImagTime, '%08.3f'), ' [Sec]']);
    disp(['Total Run Time       - ', num2str(runTime, '%08.3f'), ' [Sec]']);
    disp(['Expected Run Time    - ', num2str((numImages / ii) * runTime, '%08.3f'), ' [Sec]']);
    disp([' ']);
end

save([dataFolderPath, 'tTrainImage'], 'tTrainImage');
save([dataFolderPath, 'vImageNum'], 'vImageNum');


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

