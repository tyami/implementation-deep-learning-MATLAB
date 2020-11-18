function [ outP, dP ] = FUNC_PARAMETERS_UPDATE_DBN( inP, gradP, dP, eta, lambda, mu )
%[ outP, dP ] = FUNC_PARAMETERS_UPDATE_DBN( inP, gradP, dP, eta, lambda, mu )
% eta       : learning rate
% lambda    : weight decay constant 
% mu        : momentum constant
% momentum_method : vanilla / M / NAG

% modified for RBM (gradient descent -> gradient ascent)
% 원본은 FFNet 폴더 확인 !

dP          = eta * (gradP - lambda * inP) + mu * dP;
outP        = inP + dP;

end