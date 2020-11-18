function [ outP, dP ] = FUNC_PARAMETERS_UPDATE( inP, gradP, dP, eta, lambda, mu, momentum_method )
%[ outP, dP ] = FUNC_PARAMETERS_UPDATE( inP, gradP, dP, eta, lambda, mu, momentum_method )
% eta       : learning rate
% lambda    : weight decay constant 
% mu        : momentum constant
% momentum_method : vanilla / M / NAG

switch momentum_method
    case 'vanilla'
        dP          = - eta * (gradP - lambda * inP);
        outP        = inP + dP;
        
    case 'M'
        dP          = - eta * (gradP - lambda * inP) + mu * dP;
        outP        = inP + dP;
        
    case 'NAG'
        dP_prev     = dP;
        dP          = - eta * (gradP - lambda * inP) + mu * dP;
        
        outP        = inP + (1 + mu) * dP - mu*dP_prev;
end

end