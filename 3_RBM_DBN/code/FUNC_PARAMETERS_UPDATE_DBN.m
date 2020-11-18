function [ outP, dP ] = FUNC_PARAMETERS_UPDATE_DBN( inP, gradP, dP, eta, lambda, mu )
%[ outP, dP ] = FUNC_PARAMETERS_UPDATE_DBN( inP, gradP, dP, eta, lambda, mu )
% eta       : learning rate
% lambda    : weight decay constant 
% mu        : momentum constant
% momentum_method : vanilla / M / NAG

% modified for RBM (gradient descent -> gradient ascent)
% ������ FFNet ���� Ȯ�� !

dP          = eta * (gradP - lambda * inP) + mu * dP;
outP        = inP + dP;

end