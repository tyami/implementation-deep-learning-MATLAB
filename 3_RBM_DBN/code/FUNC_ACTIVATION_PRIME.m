function [ dY_dZ ] = FUNC_ACTIVATION_PRIME( Z, Y, func_type)
%[ dY_dZ ] = FUNC_ACTIVATION_PRIME( Y, func_type, Z)
%   Z: relu, lrelu function only

switch func_type
    case 'linear'
        dY_dZ           = 1;
        
    case 'sigmoid'
        dY_dZ           = Y .* (1 - Y);
        
    case 'relu'
        dY_dZ           = ones(size(Y));
        dY_dZ(Z<=0)     = 0;
        
    case 'lrelu'
        dY_dZ           = ones(size(Y));
        dY_dZ(Z<=0)     = 0.1;
        
    case 'vlrelu'
        dY_dZ           = ones(size(Y));
        dY_dZ(Z<=0)     = 0.01;
        
    case 'tanh'
        dY_dZ           = 1 - Y.^2;
        
    case 'softmax' % 고쳐야함. 미분 어떻게 하지?
        dY_dZ           = Y .* (1 - Y);
end
end

