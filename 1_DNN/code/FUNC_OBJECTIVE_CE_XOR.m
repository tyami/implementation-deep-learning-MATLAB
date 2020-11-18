function [ E ] = FUNC_OBJECTIVE_CE_XOR( Y, trueY )
%[ E ] = FUNC_OBJECTIVE_CE_XOR( Y, trueY )

nY              = length(Y);

% cross entrophy loss function of logistic value
E               = - (trueY' * log(Y) + (1-trueY)' * log(1-Y)) / nY;
end