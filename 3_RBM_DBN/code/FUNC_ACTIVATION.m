function [ Y ] = FUNC_ACTIVATION( Z, func_type )
%[ Y ] = FUNC_ACTIVATION( Z, func_type )
%   Coded by T. Yang

switch func_type
    case 'linear'
        Y           = Z;
        
    case 'sigmoid'
        Y           = 1./(1+exp(-Z));
        
    case 'relu'
        Y           = max(0, Z);
        
    case 'lrelu'
        Y           = max(0.1*Z, Z);
        
    case 'vlrelu'
        Y           = max(0.01*Z, Z);
         
    case 'tanh'
        Y           = tanh(Z);
        
    case 'softmax'
        Y           = exp(Z) ./ sum(exp(Z));
end
end

