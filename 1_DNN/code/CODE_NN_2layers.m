%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Deep learning study - Neural network
%
% 2 hidden layer neural network
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
% display_network(tr_images(:,1:10))
% disp(tr_labels(1:10))


%% NN Architecture
% Input layer  (WI) : 784
% Hidden layer (WH) : 2 layer (100)     - L1: Y1 = sigmoid(W1 * Y0 + b1)
%                                       - L2: Y2 = sigmoid(W2 * Y1 + b2)
% Output layer (WO) : 10                - L3: Y3 = sigmoid(W3 * Y2 + b3)
nNode           = [784, 1000, 500, 10];
weights.L1      = 2*rand(nNode(1), nNode(2)) - 1;
weights.L2      = 2*rand(nNode(2), nNode(3)) - 1;
weights.L3      = 2*rand(nNode(3), nNode(4)) - 1;

bias.L1         = 2*rand(1, nNode(2)) - 1;
bias.L2         = 2*rand(1, nNode(3)) - 1;
bias.L3         = 2*rand(1, nNode(4)) - 1;

gradients.L1    = zeros(size(weights.L1));
gradients.L2    = zeros(size(weights.L2));
gradients.L3    = zeros(size(weights.L3));


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
    Y1          = FUNC_ACTIVATION(Z1, 'sigmoid');
    
    % hidden1 -> hidden1 (L2): activation.L2
    Z2          = weights.L2' * Y1 + repmat(bias.L2', 1, szBatch);
    Y2          = FUNC_ACTIVATION(Z2, 'sigmoid');
    
    % hidden2 -> output (L3): activation.L3
    Z3          = weights.L3' * Y2 + repmat(bias.L3', 1, szBatch);
    Y3          = FUNC_ACTIVATION(Z3, 'sigmoid');
    
    
    % [STEP 2] OBJECTIVE FUNCTION: cross entrophy loss
    trueY_onehot    = FUNC_ONE_HOT_ENCODING(10, trueY);
    E(iStep)    = FUNC_OBJECTIVE_SSE(Y3, trueY_onehot);
    
    
    % [STEP 3] BACK PROPAGATION
    
    % output -> hidden2 (L3)
    dE_dY3      = Y3 - trueY_onehot;
    dY3_dZ3     = Y3 .* (1 - Y3); % df(Z)/dZ - fprime
    delta.L3    = dE_dY3 .* dY3_dZ3;
    
    dZ3_dWL3    = Y2;
    
    gradients.L3    = dZ3_dWL3 * delta.L3';
    
    % hidden2 -> hidden1 (L2)
    dZ3_dY2      = weights.L3;
    dY2_dZ2     = Y2 .* (1 - Y2); % df(Z)/dZ - fprime
    delta.L2    = (dZ3_dY2 * delta.L3) .* dY2_dZ2;
    
    dZ2_dWL2    = Y1;
    
    gradients.L2    = dZ2_dWL2 * delta.L2';
                    
    % hidden1 -> input (L1)
    dZ2_dY1     = weights.L2;
    dY1_dZ1     = Y1 .* (1 - Y1); % df(Z)/dZ - fprime
    delta.L1    = (dZ2_dY1 * delta.L2) .* dY1_dZ1;
    
    dZ1_dWL1    = Y0;
    
    gradients.L1    = dZ1_dWL1 * delta.L1';
    
                    
    % [STEP 4] updates
    weights.L3  = weights.L3 - learning_rate * gradients.L3;
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
Z1          = weights.L1' * Y0 + repmat(bias.L1', 1, nData);
Y1          = FUNC_ACTIVATION(Z1, 'sigmoid');

% hidden1 -> hidden1 (L2): sigmoid
Z2          = weights.L2' * Y1 + repmat(bias.L2', 1, nData);
Y2          = FUNC_ACTIVATION(Z2, 'sigmoid');

% hidden2 -> output (L3): sigmoid
Z3          = weights.L3' * Y2 + repmat(bias.L3', 1, nData);
Y3          = FUNC_ACTIVATION(Z3, 'sigmoid');

[~, estimateY]   = max(Y3);

ACCURACY    = sum(trueY == estimateY') / nData