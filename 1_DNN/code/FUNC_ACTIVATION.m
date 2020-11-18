function [ Y ] = FUNC_ACTIVATION( Z, func_type )
%[ Y ] = FUNC_ACTIVATION( Z, func_type )
%   자세한 설명 위치

switch func_type
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

