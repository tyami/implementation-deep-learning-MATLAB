%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Deep learning study - Neural network
%
% Deep Belief Network (DBN)
%
% coded by T.Yang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc

% filepath
[parent, ~, ~]  = fileparts(pwd);
DATA_FILEPATH   = [parent '\data\'];

% training dataset
tr_images       = loadMNISTImages([DATA_FILEPATH 'train-images.idx3-ubyte'])';
tr_labels       = loadMNISTLabels([DATA_FILEPATH 'train-labels.idx1-ubyte'])';

% test dataset
te_images       = loadMNISTImages([DATA_FILEPATH 't10k-images.idx3-ubyte'])';
te_labels       = loadMNISTLabels([DATA_FILEPATH 't10k-labels.idx1-ubyte'])';

% data check
display_network(tr_images(1:100, :)');
% disp(tr_labels(1:100));


%% parameters
nTrain          = 60000;

nEpoch          = 10;
szBatch         = 100;

nNode           = [784, 1000, 500, 250, 2];

nL              = length(nNode);

% learning rate
eta             = 0.1; % learning rate

% weight penalty
lambda          = 0.0002;

% set update method
mu              = 0.5; % momentum constant: 0.5, 0.9, 0.95, 0.99


%% initialize
dat{1}          = tr_images(1:nTrain, :);


%% training
for iL = 1:nL-1
    % initialize for each block
    nHid        = nNode(iL + 1);
    
    if iL == nL-1 % 마지막은 exponential 쓰지 않고 그냥 logit
        [weights{iL}, bias_vis{iL}, bias_hid{iL}, E_avg{iL}, dat{iL+1}]    = FUNC_RBM(iL, dat{iL}, nHid, eta, mu, lambda, nTrain, nEpoch, szBatch, 1);
    else
        [weights{iL}, bias_vis{iL}, bias_hid{iL}, E_avg{iL}, dat{iL+1}]    = FUNC_RBM(iL, dat{iL}, nHid, eta, mu, lambda, nTrain, nEpoch, szBatch, 0);
    end
end

save([DATA_FILEPATH 'trainedParameters'], 'weights', 'bias_vis', 'bias_hid', 'E_avg');


%% scatter plot with 2 dimensions (training data, dat{5})
% nL = 5;
figure;
set(gcf, 'units', 'normalized', 'position', [.2 .2 .4 .6]);
aHand           = axes;
hold on; axis square; axis off;

clist       = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1; .5 0 0; 0 0 .5; 0 .5 0; .8 .8 .8];

dat_nL      = dat{nL};

for iClass = 1:10
    scatter(dat_nL(tr_labels == iClass,1), dat_nL(tr_labels == iClass,2), '.', 'filled', 'markerEdgeColor', clist(iClass, :))
end


%% reconstruction
nL = 5
for iL = nL-1:-1:1
    
    if iL == nL-1
        datReconst{iL}      = FUNC_DBN_RECONSTRUCTION(dat{iL+1}, weights{iL}, bias_vis{iL});
    else
        datReconst{iL}      = FUNC_DBN_RECONSTRUCTION(datReconst{iL+1}, weights{iL}, bias_vis{iL});    
    end

end


% check
nTest           = 100;
idxTest         = 1:nTest;
% idxTest         = randperm(60000, nTest);

% plot for original validation data
figure(1);
set(gcf, 'units', 'normalized', 'position', [.05 .1 .4 .8]);

datDisp        = dat{1}(idxTest, :); % step, batch에 해당하는 데이터 [szBatch, 784]
display_network(datDisp')

% plot for reconstructed data
figure(2);
set(gcf, 'units', 'normalized', 'position', [.55 .1 .4 .8]);

datDisp        = datReconst{1}(idxTest, :); % step, batch에 해당하는 데이터 [szBatch, 784]
display_network(datDisp')



%% test data dim reduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% test data (n=10000)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ndatTest           = 10000;
dat_te{1}       = te_images(1:ndatTest, :);
nData           = length(dat_te{1});

for iL = 1:nL-1
    
    if iL == nL-1
        dat_te{iL+1}      = FUNC_CALCULATE_CONDITIONAL_PROB(dat_te{iL}, weights{iL}, repmat(bias_hid{iL}, nData, 1), 1);
    else
        dat_te{iL+1}      = FUNC_CALCULATE_CONDITIONAL_PROB(dat_te{iL}, weights{iL}, repmat(bias_hid{iL}, nData, 1), 0);
    end 

end

figure;
set(gcf, 'units', 'normalized', 'position', [.2 .2 .4 .6]);
aHand           = axes;
hold on; axis square; axis off;

clist       = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1; .5 0 0; 0 0 .5; 0 .5 0; .8 .8 .8];

dat_nL      = dat_te{nL};

for iClass = 1:10
    scatter(dat_nL(te_labels == iClass,1), dat_nL(te_labels == iClass,2), '.', 'filled', 'markerEdgeColor', clist(iClass, :))
end


%% test data reconstruction
nL = 5; % deep level
for iL = nL-1:-1:1
    
    if iL == nL-1
        datReconst_te{iL}      = FUNC_CALCULATE_CONDITIONAL_PROB(dat_te{iL+1}, weights{iL}', repmat(bias_vis{iL}, nData, 1));
    else
        datReconst_te{iL}      = FUNC_CALCULATE_CONDITIONAL_PROB(datReconst_te{iL+1}, weights{iL}', repmat(bias_vis{iL}, nData, 1));
    end
    
end

% check
nTest           = 100;
idxTest         = 1:nTest;
% idxTest         = randperm(60000, nTest);

% plot for original validation data
figure(1);
set(gcf, 'units', 'normalized', 'position', [.05 .1 .4 .8]);

datDisp        = dat_te{1}(idxTest, :); % step, batch에 해당하는 데이터 [szBatch, 784]
display_network(datDisp')

% plot for reconstructed data
figure(2);
set(gcf, 'units', 'normalized', 'position', [.55 .1 .4 .8]);

datDisp        = datReconst_te{1}(idxTest, :); % step, batch에 해당하는 데이터 [szBatch, 784]
display_network(datDisp')


%% 각 레이어 크기를 정방형 (n^2)으로 만들었을 때, 각 레이어에서의 이미지 그려보기
% iL = 1;
% len = sqrt(nNode(iL));
% 
% 
% datBatch        = datReconst{iL}(idxTest, :); % step, batch에 해당하는 데이터 [szBatch, 784]
% % plot for reconstructed data
% figure(3);
% set(gcf, 'units', 'normalized', 'position', [.05 .1 .4 .8]);
% for i = 1: 10
%     for j = 1:10
%         subplot(10, 10,(i-1)*10 + j)
%         imagesc(reshape(datBatch((i-1)*10 + j, :), len, len))
%         axis off; axis square;
%     end
% end
% colormap(gray)
