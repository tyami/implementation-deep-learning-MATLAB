%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Deep learning study - Neural network
%
% Deep Belief Network (DBN)
%   - Fine tuning
%
% coded by T.Yang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc

% filepath
[parent, ~, ~]  = fileparts(pwd);
DATA_FILEPATH   = [parent '\data\'];

load([DATA_FILEPATH 'trainedParameters']);
trainedW        = [weights flip(weights, 2)];
trainedB        = [bias_hid flip(bias_vis, 2)];

clear weights bias_hid bias_vis

% training dataset
tr_images       = loadMNISTImages([DATA_FILEPATH 'train-images.idx3-ubyte']);
tr_labels       = loadMNISTLabels([DATA_FILEPATH 'train-labels.idx1-ubyte']);

% test dataset
te_images       = loadMNISTImages([DATA_FILEPATH 't10k-images.idx3-ubyte']);
te_labels       = loadMNISTLabels([DATA_FILEPATH 't10k-labels.idx1-ubyte']);



%% NN Architecture
% Input layer  (WI) : 784
% Hidden layer (WH) : layer: 1 ~ nL-2L
% Output layer (WO) : layer: nL-1

% set the number of layers and nodes
nNode           = [784, 1000, 500, 250, 2, 250, 500, 1000, 784];
nL              = length(nNode);

% set activation functions of each layer
%   examples: sigmoid, relu, lrelu, tanh, softmax (ing)
activation.L1   = 'sigmoid';
activation.L2   = 'sigmoid';
activation.L3   = 'sigmoid';
activation.L4   = 'linear';
activation.L5   = 'sigmoid';
activation.L6   = 'sigmoid';
activation.L7   = 'sigmoid';
activation.L8   = 'sigmoid';

% initialize node parameters
fn              = fieldnames(activation);
for iL = 1:nL-1
    if iL < nL /2
        weights.(fn{iL})    = trainedW{iL};
    else
        weights.(fn{iL})    = trainedW{iL}';
    end
    bias.(fn{iL})       = trainedB{iL};
    
    dW.(fn{iL})         = zeros(size(weights.(fn{iL})));
    dB.(fn{iL})         = zeros(size(bias.(fn{iL})));
end

% objective function: SSE, CE
objective_func  = 'SSE';


%% Training parameters
nEpoch          = 100;
VALIDATION_ERROR    = zeros(nEpoch, 1);

szBatch         = 100;
eta             = 0.01; % learning rate

% weight decay
lambda          = 0.0002;

% set update method
%   example: vanilla, M (Momentum), NAG (Nesterov's Accelerated Gradient)
momentum_method = 'M';
mu              = 0.5; % momentum constant: 0.5, 0.9, 0.95, 0.99


%% Figure setting
figure(2); gcf;
aHand           = axes;


%% training dataset
nData           = size(tr_images, 2);
nTrain          = nData * 0.8;

fprintf('\n\n\n[TEST DATASET] nData: %d\n\n\n', nData);

for iEpoch = 1:nEpoch
    fprintf('[Training...] %d / %d\n', iEpoch, nEpoch);
    
    % setmentation - training // validation
    idxTrain        = randperm(nData, nTrain);
    idxValid        = ~ismember(1:nData, idxTrain);
    
    % training
    [weights, bias, dW, dB, E ] = FUNC_NN_TRAIN(tr_images(:,idxTrain), tr_labels(idxTrain), nL,...
                                                weights, bias, dW, dB, activation, objective_func,...
                                                szBatch, eta, lambda, mu, momentum_method);

    % validation
    [VALIDATION_ERROR(iEpoch), ~, ~, ~]  = FUNC_NN_TEST(tr_images(:,idxValid), tr_labels(idxValid), nL,...
                                                weights, bias, activation);
                                            
    % plot
    plot(aHand, 1:iEpoch, VALIDATION_ERROR(1:iEpoch) * 100);
    
    set(aHand, 'XLim', [0 nEpoch], 'yLim', [0 20]);
    title(aHand, 'Validation error rate by epochs')
    xlabel(aHand, 'Epochs');
    ylabel(aHand, 'Error rate (%)')
    pause(0.001);
end


%% test dataset
nData           = size(te_images, 2);
fprintf('\n\n\n[TEST DATASET] nData: %d\n\n\n', size(te_images, 2));

[ERROR, CONFUSION, IND, PER]  = FUNC_NN_TEST(te_images, te_labels, nL,...
                                            weights, bias, activation);

                                        
%% check wrong-answered cases
REAL        = 2;
ESTIM       = 3;

IND{REAL+1, ESTIM+1}
display_network(te_images(:,IND{REAL+1, ESTIM+1}));