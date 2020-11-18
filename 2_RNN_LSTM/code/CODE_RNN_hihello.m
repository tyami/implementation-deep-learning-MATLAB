%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Deep learning study - Neural network
%
% Recurrent Neural Network - basic. hihello
%
% coded by T.Yang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc

% dataset
dataX           = 'hihell';
dataX_onehot    = FUNC_ONE_HOT_ENCODING_HIHELLO(dataX);

dataY           = 'eiello';
dataY_onehot    = FUNC_ONE_HOT_ENCODING_HIHELLO(dataY);


%%
% hyper parameters
nDics           = 5;

szInput         = nDics;    % #input (hihell (ihello) -> 5: h, i, e, l, o)
szHidden        = szInput; % #hidden state의 vector 길이 (보통 nDic과 동일)
szOutput        = szInput; % #output (보통 nDIc과 동일)
sequence_length = length(dataX); % # time (hihell (ihello) -> 6)

eta             = 0.1; % learning rate

% objective function: SSE, CE
objective_func  = 'CE';

% initialize
W_ih            = 1/sqrt(szInput) * (2 * rand(szInput, szHidden) - 1); 
W_hh            = 1/sqrt(szHidden) * (2 * rand(szHidden, szHidden) - 1); 
W_ho            = 1/sqrt(szHidden) * (2 * rand(szHidden, szOutput) - 1);

b_h             = zeros(1, szHidden);
b_o             = zeros(1, szOutput);

h_init          = zeros(size(b_h))';

activation_h    = 'tanh';
activation_o    = 'softmax';

nStep           = 2000;
for iStep = 1:nStep
    
    if mod(iStep, 100) == 0
        fprintf('[Training...] %d / %d\n', iStep, nStep);
    end
    
    
    % [step 1] feed forward
    % [step 1-0] initialize
    net_h       = zeros(szHidden, sequence_length);
    h           = zeros(szHidden, sequence_length);
    net_o       = zeros(szOutput, sequence_length);
    o           = zeros(szOutput, sequence_length);
        
    for t = 1:sequence_length
        if t == 1 % w/ initial state
            h_prev  = h_init;
        else %  w/ previous state: h(t-1)
            h_prev  = h(:, t-1);
        end
        
        net_h(:, t) = W_ih' * dataX_onehot(:, t) +...
                        W_hh' * h_prev + b_h';
        h(:, t) = FUNC_ACTIVATION(net_h(:, t), activation_h) ;
                
        net_o(:, t) = W_ho' * h(:, t) + b_o';
        o(:, t) = FUNC_ACTIVATION(net_o(:, t), activation_o);
    end
    
    % [step 2] objective function - cross entrophy
    E(iStep, 1)     = FUNC_OBJECTIVE( o, dataY_onehot, objective_func);
    txto{iStep, 1}  = FUNC_ONE_HOT_DECODING_HIHELLO( o ); 
    fprintf('%s (%.3f)\n', txto{iStep, 1}, E (iStep));
    
    % [step 3] backpropagation through time (BPTT)
    % [step 3-0] initialize
    dE_dWih    = zeros(size(W_ih));
    dE_dWhh    = zeros(size(W_hh));
    dE_dWho    = zeros(size(W_ho));

    dE_dbh     = zeros(size(b_h));
    dE_dbo     = zeros(size(b_o));
    
    delta_h    = zeros(szHidden, sequence_length);
    
    for t = sequence_length:-1:1
        % [step 3-1] hidden->output(W)
        % delta_o   = dE_do * do_dnet(o)
        dE_do       = FUNC_OBJECTIVE_PRIME( o(:, t), dataY_onehot(:, t), objective_func );
        do_dneto    = FUNC_ACTIVATION_PRIME( net_o(:, t), o(:, t), activation_o );
        
        delta_o     = dE_do .* do_dneto;
        
        % output layer bias
        dE_dbo      = dE_dbo + delta_o';
        
        % hidden->output weight
        dneto_dw    = h(:, t);
        dE_dWho     = dE_dWho + dneto_dw * delta_o';

        % [step 3-2] input->hidden(V), recurrent(U)
        % delta_h   = dE_do * do_dnet(o) * dnet(o)_ds * ds_dnet(h)
        %           = SIGMA(output){ delta_o * dnet_ds * ds_dnet(h) } 
        dneto_dh    = W_ho;
        dh_dneth    = FUNC_ACTIVATION_PRIME( net_h(:, t), h(:, t), activation_h );
        
        % delta_hidden: t+1의 delta_h 값을 계속 누적
        %  - output에서 t+1 까지의 누적된 delta_h에 t시점의 delta_h를 더해줌
        delta_h(:, t)   = (delta_h(:, t) + dneto_dh * delta_o) .* dh_dneth;
        
        % hidden layer bias
        dE_dbh      = dE_dbh + delta_h(:, t)';
        
        % input->hidden weight
        dE_dWih     = dE_dWih + dataX_onehot(:, t) * delta_h(:, t)';
        
        if t > 1        
            % recurrent weight
            dE_dWhh     = dE_dWhh + h(:, t-1) * delta_h(:, t)';
            delta_h(:, t-1) = W_hh * delta_h(:, t); % t-1 시점의 delta_h 값에 누적합을 미리 넣어둠
        else        
            % recurrent weight
            dE_dWhh     = dE_dWhh + h_init * delta_h(:, t)';
        end
    end
    
    % [step 4] weight, bias update
    
    W_ih    = W_ih - eta * dE_dWih;
    W_ho    = W_ho - eta * dE_dWho;
    W_hh     = W_hh - eta * dE_dWhh;
    
    b_h      = b_h - eta * dE_dbh;
    b_o      = b_o - eta * dE_dbo;  
end

plot(E)