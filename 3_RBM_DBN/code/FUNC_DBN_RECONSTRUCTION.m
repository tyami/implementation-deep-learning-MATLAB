function [ datReconst ] = FUNC_DBN_RECONSTRUCTION(datIn, weights, bias_vis)
%DBN reconstruction code
%   Coded by T. Yang

nData           = size(datIn, 1);

% convert to the binary state {0, 1}
state_hid       = datIn; %> rand(size(datIn));

% calculate p(v|h;w) - Gibbs sampling step 1/2
datReconst      = FUNC_CALCULATE_CONDITIONAL_PROB(state_hid, weights', repmat(bias_vis, nData, 1));
