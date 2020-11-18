function [ E ] = FUNC_OBJECTIVE( Y, trueY, func_type )
%[ E ] = FUNC_OBJECTIVE_SSE( Y, trueY, func_type )
% Y     : matrix of estimated Y (the result of output layer, 0~1)
% trueY : one hot encoded matrix of true Y
% func_type : SSE, CE
%
% [data structure] class X data samples
    whos trueY Y
    
switch func_type
    case 'SSE' % sum of squared error, norm2
        E       = sum(sum((Y - trueY).^2))/ 2;
    
    case 'MSE' % mean squared error
        

    case 'CE' % cross-entrophy
        E       = - sum( sum( trueY .*log(Y) ) );
        
end
end