%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Deep learning study - Neural network
%
% 2 layer neural network - XOR problem
%
% coded by T.Yang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc

% filepath
[parent, ~, ~]  = fileparts(pwd);
LOAD_FILEPATH   = [parent '\data\'];

% training dataset
tr_data         = [0 0; 0 1; 1 0; 1 1]';
tr_labels       = [0; 1; 1; 0]';

% test dataset
te_data         = tr_data;
te_labels       = tr_labels;


%% NN Architecture
% Input layer  (WI) : 2
% Hidden layer (WH) : 1 layer (10)     - L1: Y1 = sigmoid(W1 * Y0 + b1)
% Output layer (WO) : 1                - L2: Y2 = sigmoid(W2 * Y1 + b2)
nNode           = [2, 10, 1];
weights.L1      = 2*rand(nNode(1), nNode(2)) - 1;
weights.L2      = 2*rand(nNode(2), nNode(3)) - 1;

bias.L1         = zeros(1, nNode(2)); % 2*rand(nNode(2), 1) - 1;
bias.L2         = zeros(1, nNode(3)); % 2*rand(nNode(3), 1) - 1;

gradients.L1    = zeros(size(weights.L1));
gradients.L2    = zeros(size(weights.L2));


%% Trining parameters
nStep           = 10000;
% szBatch         = 1000;
learning_rate   = 0.1;

E               = zeros(nStep, 1);


%% NN Training
nData           = size(tr_data, 2);


for iStep = 1:nStep
    if mod(iStep,100) == 0
        fprintf('[Training...] %d / %d\n', iStep, nStep);
    end
    
    Y0      = tr_data;
    trueY   = tr_labels;
    
    % [STEP 1] FEED FORWARD: Y = Z(W'*T)
    % input -> hidden (L1): sigmoid
    Z1          = weights.L1' * Y0 + repmat(bias.L1', 1, nData);
    Y1          = FUNC_ACTIVATION_sigmoid(Z1);

    % hidden -> output (L2): sigmoid
    Z2          = weights.L2' * Y1 + repmat(bias.L2', 1, nData);
    Y2          = FUNC_ACTIVATION_sigmoid(Z2);

    
    % [STEP 2] OBJECTIVE FUNCTION: cross entrophy loss
    E(iStep)    = FUNC_OBJECTIVE_SSE_XOR(Y2, tr_labels);
    
    
    % [STEP 3] BACK PROPAGATION
    % output -> hidden (L2)
    dE_dY2      = Y2 - trueY;
    dY2_dZ2     = Y2 .* (1 - Y2); % df(Z)/dZ - fprime
    delta.L2    = dE_dY2 .* dY2_dZ2;
    
    dZ2_dWL2    = Y1;
    
    gradients.L2    = dZ2_dWL2 * delta.L2';

    % hidden -> input (L1)
    dZ2_dY1     = weights.L2;
    dY1_dZ1     = Y1 .* (1 - Y1); % df(Z)/dZ - fprime
    delta.L1    = (dZ2_dY1 * delta.L2) .* dY1_dZ1;
    
    dZ1_dWL1    = Y0;

    gradients.L1    = dZ1_dWL1 * delta.L1';
    
    
    % [STEP 4] updates
    weights.L2  = weights.L2 - learning_rate * gradients.L2;
    weights.L1  = weights.L1 - learning_rate * gradients.L1;
    
end


%% PRINT training result
plot(1:nStep, E);
xlabel('Iterations');
ylabel('Error');
title('Cost function: (Y_{estimate} - Y_{true})^2');


%% NN Test
nData           = size(te_data, 2);

estimateY       = zeros(1, nData);

Y0              = te_data;
trueY           = te_labels;

% [STEP 1] FEED FORWARD: Y = Z(W'*T)
% input -> hidden (L1): sigmoid
Z1          = weights.L1' * Y0 + repmat(bias.L1', 1, nData);
Y1          = FUNC_ACTIVATION_sigmoid(Z1);

% hidden -> output (L2): sigmoid
Z2          = weights.L2' * Y1 + repmat(bias.L2', 1, nData);
Y2          = FUNC_ACTIVATION_sigmoid(Z2);


Y2
estimateY(Y2 > 0.5)     = 1
trueY

sum(trueY == estimateY) / nData * 100
