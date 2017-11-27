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
                        convolution2dLayer(5, 50, 'Padding', 0)
                        maxPooling2dLayer(2, 'Stride', 2)
                        reluLayer()
                        batchNormalizationLayer()

                        convolution2dLayer(3, 40, 'Padding', 1)
                        maxPooling2dLayer(2, 'Stride', 2)
                        reluLayer()
                        batchNormalizationLayer()

                        convolution2dLayer(3, 30, 'Padding', 1)
                        reluLayer()
                        batchNormalizationLayer()

                        fullyConnectedLayer(48)
                        leakyReluLayer()
                        dropoutLayer(0.1)
                        fullyConnectedLayer(32)
                        leakyReluLayer()
                        dropoutLayer(0.1)
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

