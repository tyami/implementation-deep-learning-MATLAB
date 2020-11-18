function [ weights, bias_vis, bias_hid, E_sum, datOut ] = FUNC_RBM(iL, datIn, nHid, eta, mu, lambda, nTrain, nEpoch, szBatch, isEnd)
%Restricted Boltzman Machine Block (RBM) for Deep Belief Network (DBN)
%   Coded by T. Yang

[nData, nVis]   = size(datIn);
nStep           = ceil(nTrain / szBatch);


%% initialize
weights         = 0.1 * randn(nVis, nHid);
bias_vis        = zeros(1, nVis);
bias_hid        = zeros(1, nHid);

dW_pos          = zeros(size(weights));
dW_neg          = zeros(size(weights));
dW              = zeros(size(weights));
dB_vis          = zeros(size(bias_vis));
dB_hid          = zeros(size(bias_hid));

E_sum           = zeros(nEpoch, 1);

if isEnd
    eta         = eta / 100;
end


%% training
for iEpoch = 1:nEpoch
    % order randomization
    idxTrain        = randperm(nData, nTrain);
    dat             = datIn(idxTrain, :);
    
    if iEpoch > 5
        mu = 0.9;
    end
    
    for iStep = 1:nStep
        if iStep < nStep
            idxBatch    = (iStep - 1) * szBatch + 1 : iStep * szBatch;           
        else
            idxBatch    = (iStep - 1) * szBatch + 1 : nTrain;
        end
        datBatch        = dat(idxBatch, :); % step, batch에 해당하는 데이터 [200, 784]    
        
        
        % [STEP 1] Positive phase
        
        % calculate p(h|v;w)
        prob_hid_pos    = FUNC_CALCULATE_CONDITIONAL_PROB(datBatch, weights, repmat(bias_hid, szBatch, 1), isEnd);

        % weight update for positive phase term
        dW_pos          = (datBatch' * prob_hid_pos) / szBatch;
        dB_vis_pos      = sum(datBatch) / szBatch;
        dB_hid_pos      = sum(prob_hid_pos) / szBatch;
        
        
        % [STEP 2] Negative phase
        % convert to the binary state {0, 1}
        if isEnd % hinton 코드 참고함 - 마지막 layer에서만 왜 이렇게 해주는걸까?
            state_hid       = prob_hid_pos + randn(size(prob_hid_pos));
        else
            state_hid       = prob_hid_pos > rand(size(prob_hid_pos));
        end
        
        % calculate p(v|h;w) - Gibbs sampling step 1/2
        prob_vis_neg    = FUNC_CALCULATE_CONDITIONAL_PROB(state_hid, weights', repmat(bias_vis, szBatch, 1));
        % calculate p(h|v;w) - Gibbs sampling step 2/2
        prob_hid_neg    = FUNC_CALCULATE_CONDITIONAL_PROB(prob_vis_neg, weights, repmat(bias_hid, szBatch, 1), isEnd);
                
        % weight update for negative phase term
        dW_neg          = prob_vis_neg' * prob_hid_neg / szBatch;
        dB_vis_neg      = sum(prob_vis_neg) / szBatch;
        dB_hid_neg      = sum(prob_hid_neg) / szBatch;
        
        
        % [STEP 3] error
        E               = sum(sum( (datBatch - prob_vis_neg ).^2 )) / szBatch;
        E_sum(iEpoch)   = E_sum(iEpoch) + E;
        
        
        % [STEP 4] weights update
        [ weights, dW ] = FUNC_PARAMETERS_UPDATE_DBN( weights, (dW_pos-dW_neg), dW, eta, lambda, mu );
        [ bias_vis, dB_vis ] = FUNC_PARAMETERS_UPDATE_DBN( bias_vis, (dB_vis_pos-dB_vis_neg), dB_vis, eta, 0, mu );
        [ bias_hid, dB_hid ] = FUNC_PARAMETERS_UPDATE_DBN( bias_hid, (dB_hid_pos-dB_hid_neg), dB_hid, eta, 0, mu );
        
    end
    E_sum(iEpoch)       = E_sum(iEpoch) / nStep;
    fprintf('Block %d, Epoch %2d, Error avg %4.3f\n', iL, iEpoch, E_sum(iEpoch));
    
end


%% data for next layer
% calculate p(h|v;w)
datOut          = FUNC_CALCULATE_CONDITIONAL_PROB(datIn, weights, repmat(bias_hid, nData, 1), isEnd);


end

