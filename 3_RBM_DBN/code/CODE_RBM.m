%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Deep learning study - Neural network
%
% Restricted Boltzman Machine (RBM)
%
% coded by T.Yang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc

% filepath
LOAD_FILEPATH   = '../data/';

% training dataset
tr_images       = loadMNISTImages([LOAD_FILEPATH 'train-images.idx3-ubyte'])';
tr_labels       = loadMNISTLabels([LOAD_FILEPATH 'train-labels.idx1-ubyte'])';

% test dataset
te_images       = loadMNISTImages([LOAD_FILEPATH 't10k-images.idx3-ubyte'])';
te_labels       = loadMNISTLabels([LOAD_FILEPATH 't10k-labels.idx1-ubyte'])';

% data check
% display_network(tr_images(:,1:10))
% disp(tr_labels(1:10))


%% parameters
nData           = size(tr_images, 1);
nTrain          = 6000;

nEpoch          = 100;
szBatch         = 200;

nStep           = ceil(nTrain / szBatch);

nNode           = [784, 50];

% learning rate
eta             = 0.01; % learning rate

% set update method
mu              = 0; % momentum constant: 0.5, 0.9, 0.95, 0.99

% weight penalty
weightCost      = 0.0002;


%% initialize
weights         = 0.1 * randn(nNode);
bias_vis        = zeros(1, nNode(1));
bias_hid        = zeros(1, nNode(2));

dW_pos          = zeros(size(weights));
dW_neg          = zeros(size(weights));
dW              = zeros(size(weights));
dB_vis          = zeros(size(bias_vis));
dB_hid          = zeros(size(bias_hid));


E_sum           = zeros(nEpoch, 1);


%% Figure setting
figure(3); gcf;
aHand           = axes;


%% training
for iEpoch = 1:nEpoch    
    % order randomization
    idxTrain        = randperm(nData, nTrain);
    dat             = tr_images(idxTrain, :);
    
    for iStep = 1:nStep
        if iStep < nStep
            idxBatch    = (iStep - 1) * szBatch + 1 : iStep * szBatch;           
        else
            idxBatch    = (iStep - 1) * szBatch + 1 : nTrain;
        end
        datBatch        = dat(idxBatch, :); % step, batch에 해당하는 데이터 [200, 784]
        
        
        % [STEP 1] Positive phase
        % convert to the binary state {0, 1}
        state_vis       = 0.5 < datBatch;   % {0, 1} 값으로 변환
        
        % calculate p(h|v;w)
        prob_hid_pos    = FUNC_CALCULATE_CONDITIONAL_PROB(state_vis, weights, repmat(bias_hid, szBatch, 1));
                
        % weight update for positive phase term
        dW_pos          = (state_vis' * prob_hid_pos) ./ szBatch;
        dB_vis_pos      = sum(state_vis) ./ szBatch;
        dB_hid_pos      = sum(prob_hid_pos) ./ szBatch;
        
        
        % [STEP 2] Negative phase
        % convert to the binary state {0, 1}
        state_hid       = prob_hid_pos > rand(size(prob_hid_pos));
        
        % calculate p(v|h;w) - Gibbs sampling step 1/2
        prob_vis_neg    = FUNC_CALCULATE_CONDITIONAL_PROB(state_hid, weights', repmat(bias_vis, szBatch, 1));
        % calculate p(h|v;w) - Gibbs sampling step 2/2
        prob_hid_neg    = FUNC_CALCULATE_CONDITIONAL_PROB(prob_vis_neg, weights, repmat(bias_hid, szBatch, 1));
                
        % weight update for negative phase term
        dW_neg          = prob_vis_neg' * prob_hid_neg ./ szBatch;
        dB_vis_neg      = sum(prob_vis_neg) ./ szBatch;
        dB_hid_neg      = sum(prob_hid_neg) ./ szBatch;
        
        
        % [STEP 3] error
        E               = sum(sum( (state_vis - prob_vis_neg ).^2 ));
        E_sum(iEpoch)   = E_sum(iEpoch) + E;
        
        
        % [STEP 4] weights update
        dW              = mu*dW + eta*(dW_pos - dW_neg - weightCost*weights); % 마지막 term 은 weight penalty regularization 방법
        dB_vis          = mu*dB_vis + eta*(dB_vis_pos - dB_vis_neg);
        dB_hid          = mu*dB_hid + eta*(dB_hid_pos - dB_hid_neg);
        
        weights         = weights + dW;
        bias_vis        = bias_vis + dB_vis;
        bias_hid        = bias_hid + dB_hid;
        
    end
    
    fprintf('Epoch %d error %6.1f\n', iEpoch, E_sum(iEpoch));
    
    % plot
    plot(aHand, 1:iEpoch, E_sum(1:iEpoch)/100000);
    
    set(aHand, 'XLim', [0 nEpoch], 'yLim', [0 45]);
    title(aHand, 'Original-Estimated error sum by epochs')
    xlabel(aHand, 'Epochs');
    ylabel(aHand, 'Error sum')
    pause(0.001);
end


%% check
nDataTe         = size(te_images, 1);
nTest           = 100;
idxTest         = randperm(nDataTe, nTest);
datBatch        = te_images(idxTest, :); % step, batch에 해당하는 데이터 [784, szBatch]

% plot for original validation data
figure(1);
set(gcf, 'units', 'normalized', 'position', [.05 .1 .4 .8]);
for i = 1: 10
    for j = 1:10
        subplot(10, 10,(i-1)*10 + j)
        imagesc(reshape(datBatch((i-1)*10 + j, :), 28, 28))
        axis off; axis square;
    end
end
colormap(gray)
print(gcf, '../result/original.png', '-dpng', '-r300')



% [STEP 1] Positive phase
% convert to the binary state {0, 1}
state_vis       = 0.5 < datBatch;   % {0, 1} 값으로 변환

% calculate p(h|v;w)
prob_hid_pos    = FUNC_CALCULATE_CONDITIONAL_PROB(weights, state_vis, repmat(bias_hid, nTest, 1));

% [STEP 2] Negative phase
% convert to the binary state {0, 1}
state_hid       = prob_hid_pos > rand(size(prob_hid_pos));

% calculate p(v|h;w) - Gibbs sampling step 1/2
prob_vis_neg    = FUNC_CALCULATE_CONDITIONAL_PROB(weights', state_hid, repmat(bias_vis, nTest, 1));



% plot for reconstructed data
figure(2);
set(gcf, 'units', 'normalized', 'position', [.55 .1 .4 .8]);
for i = 1: 10
    for j = 1:10
        subplot(10, 10,(i-1)*10 + j)
        imagesc(reshape(prob_vis_neg((i-1)*10 + j, :), 28, 28))
        axis off; axis square;
    end
end
colormap(gray)
print(gcf, '../result/reconstruction.png', '-dpng', '-r300')























