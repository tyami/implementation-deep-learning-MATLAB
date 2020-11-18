%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Deep learning study - Neural network
%
% 1 hidden layer neural network
%
% coded by T.Yang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc

% filepath
[parent, ~, ~]  = fileparts(pwd);
LOAD_FILEPATH   = [parent '\data\'];

% training dataset
tr_images       = loadMNISTImages([LOAD_FILEPATH 'train-images.idx3-ubyte']);
tr_labels       = loadMNISTLabels([LOAD_FILEPATH 'train-labels.idx1-ubyte']);

% test dataset
te_images       = loadMNISTImages([LOAD_FILEPATH 't10k-images.idx3-ubyte']);
te_labels       = loadMNISTLabels([LOAD_FILEPATH 't10k-labels.idx1-ubyte']);

% data check
display_network(tr_images(:,1:10))
disp(tr_labels(1:10))


%% NN Architecture
% Input layer  (WI) : 784
% Hidden layer (WH) : 1 layer (100)     - L1: Y1 = sigmoid(W1 * Y0 + b1)
% Output layer (WO) : 10                - L2: Y2 = sigmoid(W2 * Y1 + b2)
nLayer          = 3;
nNode           = [784, 100, 10];
weights.L1      = 2*rand(nNode(1), nNode(2)) - 1;
weights.L2      = 2*rand(nNode(2), nNode(3)) - 1;

bias.L1         = zeros(1, nNode(2));
bias.L2         = zeros(1, nNode(3));

gradients.L1    = zeros(size(weights.L1));
gradients.L2    = zeros(size(weights.L2));

activation.L1   = 'sigmoid';
activation.L2   = 'sigmoid';


%% Trining parameters
nStep           = 10000;
szBatch         = 150;

learning_rate   = 1 / szBatch;

E               = zeros(nStep, 1);


%% NN Training
[~, nData]    	= size(tr_images);

for iStep = 1:nStep
    if mod(iStep, 100) == 0
        fprintf('[Training...] %d / %d\n', iStep, nStep);
    end
    
    % batch
    idxBatch    = randperm(nData, szBatch);
    
    Y0          = tr_images(:, idxBatch);
    trueY       = tr_labels(idxBatch) + 1;
    
    % batch data check - good !
%     display_network(Y0(:,1:10))
%     disp(trueY(1:10) - 1)


    % [STEP 1] FEED FORWARD
    % input -> hidden (L1): activation.L1
    Z1          = weights.L1' * Y0 + repmat(bias.L1', 1, szBatch);
    Y1          = FUNC_ACTIVATION(Z1, activation.L1);
    
    % hidden -> output (L2): activation.L2
    Z2          = weights.L2' * Y1 + repmat(bias.L2', 1, szBatch);
    Y2          = FUNC_ACTIVATION(Z2, activation.L2);
    
    
    % [STEP 2] OBJECTIVE FUNCTION: cross entrophy loss
    trueY_onehot    = FUNC_ONE_HOT_ENCODING(10, trueY);
    E(iStep)    = FUNC_OBJECTIVE_SSE(Y2, trueY_onehot);
   
    
    % [STEP 3] BACK PROPAGATION
    % output -> hidden (L2)
    dE_dY2      = Y2 - trueY_onehot;
    dY2_dZ2     = FUNC_ACTIVATION_PRIME(Z2, Y2, activation.L2); % Y2 .* (1 - Y2); % df(Z)/dZ - fprime
    delta.L2    = dE_dY2 .* dY2_dZ2;
    
    dZ2_dWL2    = Y1;
    
    gradients.L2    = dZ2_dWL2 * delta.L2';
    
    % hidden -> input (L1)
    dZ2_dY1     = weights.L2;
    dY1_dZ1     = FUNC_ACTIVATION_PRIME(Z1, Y1, activation.L1); % Y1 .* (1 - Y1); % df(Z)/dZ - fprime
    delta.L1    = (dZ2_dY1 * delta.L2) .* dY1_dZ1;
    
    dZ1_dWL1    = Y0;
    
    gradients.L1    = dZ1_dWL1 * delta.L1';
    
                    
    % [STEP 4] updates
    weights.L2  = weights.L2 - learning_rate * gradients.L2;
    weights.L1  = weights.L1 - learning_rate * gradients.L1;
    
end


%% PRINT training result
plot(1:iStep, E(1:iStep));
xlabel('Iterations');
ylabel('Error');
title('Cost function: ^1/_2(Y_{estimate} - Y_{true})^2');


%% NN Test
[~, nData]      = size(te_images);

Y0          = te_images;
trueY       = te_labels + 1;


% [STEP 1] FEED FORWARD: Y = Z(W'*T)
% input -> hidden (L1): sigmoid
Z1          = weights.L1' * Y0 + repmat(bias.L1', 1, nData);
Y1          = FUNC_ACTIVATION(Z1, activation.L1);

% hidden -> output (L2): sigmoid
Z2          = weights.L2' * Y1 + repmat(bias.L2', 1, nData);
Y2          = FUNC_ACTIVATION(Z2, activation.L2);

[~, estimateY]   = max(Y2);

ACCURACY    = sum(trueY == estimateY') / nData