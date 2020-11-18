function [ ERROR, CONFUSION, IND, PER ] = FUNC_NN_TEST(te_data, te_labels, nL, weights, bias, activation)
%[ dERROR, CONFUSION, IND, PER ] = FUNC_NN_TEST(te_data, te_labels, nL, weights, bias, activation, nStep)
% coded by T. Yang
% 170530

fn              = fieldnames(activation);

[~, nData]      = size(te_data);

Y0              = te_data;
trueY           = te_labels + 1;


% [STEP 1] FEED FORWARD

% input -> first hidden
Z.(fn{1})       = weights.(fn{1})' * Y0 + repmat(bias.(fn{1})', 1, nData);
Y.(fn{1})       = FUNC_ACTIVATION(Z.(fn{1}), activation.(fn{1}));

% the latter parts
for iL = 2:nL-1
    Z.(fn{iL})  = weights.(fn{iL})' * Y.(fn{iL-1}) + repmat(bias.(fn{iL})', 1, nData);
    Y.(fn{iL})  = FUNC_ACTIVATION(Z.(fn{iL}), activation.(fn{iL}));
end

[~, estimateY]  = max(Y.(fn{nL-1}));
estimateY   = estimateY';

% eaxmples of true Y and estimated Y
EXAMPLE     = [ trueY(1:50) estimateY(1:50) ];

% confusion matrix
trueY_onehot        = FUNC_ONE_HOT_ENCODING(10, trueY);
estimateY_onehot    = FUNC_ONE_HOT_ENCODING(10, estimateY);
[ERROR, CONFUSION, IND, PER]  = confusion(trueY_onehot, estimateY_onehot);


fprintf('           ERROR RATE: %2.3f%% \n', ERROR * 100);

end