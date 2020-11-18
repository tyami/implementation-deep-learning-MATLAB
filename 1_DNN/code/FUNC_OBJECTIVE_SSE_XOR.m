function [ E ] = FUNC_OBJECTIVE_SSE_XOR( Y, trueY )
%FUNC_OBJECTIVE_SSE_XOR( Y, trueY )

E               = sum((Y - trueY).^2) / 2;

end