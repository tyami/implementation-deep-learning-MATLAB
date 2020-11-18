function [ weights, bias, dW, dB, E ] = FUNC_NN_TRAIN(tr_data, tr_labels, nL, weights, bias, dW, dB, activation, objective_func, szBatch, eta, lambda, mu, momentum_method )
%[ weights, bias, dW, dB, E ] = FUNC_NN_TRAIN(tr_data, tr_labels, nL, weights, bias, dW, dB, activation, objective_func, szBatch, eta, update_method, mu )
% coded by T. Yang
% 170530

fn                  = fieldnames(activation);

[~, nData]          = size(tr_data);
nStep               = ceil(nData / szBatch);

% initialize
for iL = 1:nL-1
    gradW.(fn{iL})      = zeros(size(weights.(fn{iL})));
    gradB.(fn{iL})      = zeros(size(bias.(fn{iL})));
end
E                   = zeros(nStep, 1);



for iStep = 1:nStep    
    % batch
    if iStep < nStep
        idxBatch    = (iStep - 1) * szBatch + 1 : iStep * szBatch;
    else
        idxBatch    = (iStep - 1) * szBatch + 1 : nData;
    end
    Y0              = tr_data(:, idxBatch);
    trueY           = tr_labels(idxBatch) + 1;
    

    % [STEP 1] FEED FORWARD
    % input -> hidden (L1): activation.L1
    Z.(fn{1})    	= weights.(fn{1})' * Y0 + repmat(bias.(fn{1})', 1, szBatch);
    Y.(fn{1})       = FUNC_ACTIVATION(Z.(fn{1}), activation.(fn{1}));
    
    for iL = 2:nL-1
        % actvation.L{iL}
        Z.(fn{iL})  = weights.(fn{iL})' * Y.(fn{iL-1}) + repmat(bias.(fn{iL})', 1, szBatch);
        Y.(fn{iL})  = FUNC_ACTIVATION(Z.(fn{iL}), activation.(fn{iL}));
    end
    
    % [STEP 2] OBJECTIVE FUNCTION: cross entrophy loss
    trueY_onehot    = FUNC_ONE_HOT_ENCODING(10, trueY);
    E(iStep)        = FUNC_OBJECTIVE( Y.(fn{nL-1}), trueY_onehot, objective_func);
    
    
    % [STEP 3] BACK PROPAGATION
    
    % output layer -> last hidden layer (L(n-1))
    dE_dY           = FUNC_OBJECTIVE_PRIME(Y.(fn{nL-1}), trueY_onehot, objective_func);
    dY_dZ           = FUNC_ACTIVATION_PRIME(Z.(fn{nL-1}), Y.(fn{nL-1}), activation.(fn{nL-1})); % df(Z)/dZ - fprime
    delta.(fn{nL-1})    = dE_dY .* dY_dZ;
    
    dZ_dW           = Y.(fn{nL-2});
    
    gradW.(fn{nL-1})    = dZ_dW * delta.(fn{nL-1})';
    gradB.(fn{nL-1})    = ones(1, szBatch) * delta.(fn{nL-1})';
    
    % between hidden layers (L(n-1)~ L2)
    for iL = nL-2:-1:2
        dZ_dY       = weights.(fn{iL+1});
        dY_dZ       = FUNC_ACTIVATION_PRIME(Z.(fn{iL}), Y.(fn{iL}), activation.(fn{iL}));
        delta.(fn{iL})  = (dZ_dY * delta.(fn{iL+1})) .* dY_dZ;
        
        dZ_dW       = Y.(fn{iL-1});
        
        gradW.(fn{iL})  = dZ_dW * delta.(fn{iL})';
        gradB.(fn{iL})  = ones(1, szBatch) * delta.(fn{iL})';
    end
    
    % first hidden layer -> input layer (L1)
    dZ_dY           = weights.(fn{2});
    dY_dZ           = FUNC_ACTIVATION_PRIME(Z.(fn{1}), Y.(fn{1}), activation.(fn{1}));
    delta.(fn{1})   = (dZ_dY * delta.(fn{2})) .* dY_dZ;
    
    dZ_dW           = Y0;
    
    gradW.(fn{1})   = dZ_dW * delta.(fn{1})';
    gradB.(fn{1})   = ones(1, szBatch) * delta.(fn{1})';
    
                        
    % [STEP 4] updates
    for iL = nL-1:-1:1
        [weights.(fn{iL}), dW.(fn{iL})]       = FUNC_PARAMETERS_UPDATE(weights.(fn{iL}),gradW.(fn{iL}), dW.(fn{iL}),...
                                                    eta, lambda, mu, momentum_method);
        [bias.(fn{iL}), dB.(fn{iL})]          = FUNC_PARAMETERS_UPDATE(bias.(fn{iL}),gradB.(fn{iL}), dB.(fn{iL}),...
                                                    eta, lambda, mu, momentum_method);      
                                                
    end
    
end


%% PRINT training result
figure(1); gcf;
plot(1:nStep, E);
xlabel('Steps');
ylabel('Cost');
title('Cost function: ^1/_2(Y_{estimate} - Y_{true})^2');


end