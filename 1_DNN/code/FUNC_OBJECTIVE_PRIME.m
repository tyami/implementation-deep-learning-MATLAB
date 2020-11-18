function [ dE_dY ] = FUNC_OBJECTIVE_PRIME( Y, trueY, func_type )
%[ dE_dY ] = FUNC_OBJECTIVE_PRIME( Y, trueY )
% Y     : matrix of estimated Y (the result of output layer, 0~1)
% trueY : one hot encoded matrix of true Y
%
% [data structure] class X data samples

switch func_type
    case 'SSE' % sum of squared error, norm2
        dE_dY       = Y - trueY;
        
    case 'MSE' % mean squared error
        

    case 'CE' % cross-entrophy
%         dE_dY       = (Y - trueY) ./ (Y .* (1 - Y));
        dE_dY       = -(trueY ./ Y);
end
end