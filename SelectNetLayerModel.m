function [ hNetModel ] = SelectNetLayerModel( netModelIdx, numRows, numCols, numChannels )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

switch(netModelIdx)
    case(1)
        hNetModel = [
                        imageInputLayer([numRows, numCols, numChannels])
    
                        % (Kernel Size, Num Filters)
                        convolution2dLayer(3, 16, 'Padding', 1)
                        batchNormalizationLayer()
                        maxPooling2dLayer(2, 'Stride', 2)
                        reluLayer()

                        convolution2dLayer(3, 32, 'Padding', 1)
                        batchNormalizationLayer()
                        maxPooling2dLayer(2, 'Stride', 2)
                        reluLayer()

                        convolution2dLayer(3, 64, 'Padding', 1)
                        batchNormalizationLayer()
                        reluLayer()

                        fullyConnectedLayer(10)
                        softmaxLayer()
                        classificationLayer()];
    case(2)
        hNetModel = [
                        imageInputLayer([numRows, numCols, numChannels])
    
                        % (Kernel Size, Num Filters)
                        convolution2dLayer(5, 16, 'Padding', 0)
                        batchNormalizationLayer()
                        maxPooling2dLayer(2, 'Stride', 2)
                        leakyReluLayer()

                        convolution2dLayer(3, 32, 'Padding', 0)
                        batchNormalizationLayer()
                        maxPooling2dLayer(2, 'Stride', 2)
                        leakyReluLayer()

                        convolution2dLayer(3, 64, 'Padding', 1)
                        batchNormalizationLayer()
                        leakyReluLayer()

                        fullyConnectedLayer(64)
                        leakyReluLayer()
                        fullyConnectedLayer(10)
                        softmaxLayer()
                        classificationLayer()];
    case(3)
        hNetModel = [
                        imageInputLayer([numRows, numCols, numChannels])
    
                        fullyConnectedLayer(128)
                        leakyReluLayer()

                        fullyConnectedLayer(64)
                        batchNormalizationLayer()
                        leakyReluLayer()

                        fullyConnectedLayer(64)
                        batchNormalizationLayer()
                        leakyReluLayer()

                        fullyConnectedLayer(10)
                        softmaxLayer()
                        classificationLayer()];
end


end

