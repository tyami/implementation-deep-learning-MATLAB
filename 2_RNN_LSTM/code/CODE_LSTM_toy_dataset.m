%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Deep learning study - Neural network
%
% Long Short Term Memory Neural Network - using toy example
%
% coded by T.Yang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc

flag        = input('Do you want to make new training/test data ? (1/0)');

if flag
    % training dataset
    [txtTrain, datTrain_seq, datTrain_tgt]   = FUNC_CREATE_TOY_DATASET(60000);

    % % test dataset
    [txtTest, datTest_seq, datTest_tgt]      = FUNC_CREATE_TOY_DATASET(10000);
    save('dummy.mat',   'txtTrain', 'datTrain_seq', 'datTrain_tgt',...
                        'txtTest', 'datTest_seq', 'datTest_tgt');
    
else
    load('dummy.mat');
end


%%
% hyper parameters
nDic            = 7;    % #input (hihell (ihello) -> 5: h, i, e, l, o)

szInput         = nDic; % #input
szOutput        = szInput; % #output (보통 nDIc과 동일)
szHidden        = szInput; % #hidden state의 vector 길이

nData           = size(txtTrain, 1); % #sample 개수 (training sample/test sample ...)

eta             = 0.01; % learning rate


% objective function: SSE, CE
objective_func  = 'SSE';

% initialize weight matrices
W_z             = 1/sqrt(szHidden) * (2 * rand(szInput, szHidden) - 1); % weight for data
W_i             = 1/sqrt(szHidden) * (2 * rand(szInput, szHidden) - 1); % weight for input gate
W_f             = 1/sqrt(szHidden) * (2 * rand(szInput, szHidden) - 1); % weight for forget gate
W_o             = 1/sqrt(szHidden) * (2 * rand(szInput, szHidden) - 1); % weight for output gate

U_z             = 1/sqrt(szHidden) * (2 * rand(szHidden, szHidden) - 1); % weight for data
U_i             = 1/sqrt(szHidden) * (2 * rand(szHidden, szHidden) - 1); % weight for input gate
U_f             = 1/sqrt(szHidden) * (2 * rand(szHidden, szHidden) - 1); % weight for forget gate
U_o             = 1/sqrt(szHidden) * (2 * rand(szHidden, szHidden) - 1); % weight for output gate

% initialize bias vectors
b_gz            = zeros(1, szHidden);
b_gi            = zeros(1, szHidden);
b_gf            = zeros(1, szHidden);
b_go            = zeros(1, szHidden);

% concatenate weight (2hX4h) and bias (4hX1), z-i-f-o, W-U
ccn_W           = [ [W_z, W_i, W_f, W_o];
                    [U_z, U_i, U_f, U_o]];
ccn_b           = [b_gz, b_gi, b_gf, b_go];

% for hidden (LSTM) -> output
W_ho            = 1/sqrt(szHidden) * (2 * rand(szHidden, szOutput) - 1);
b_o             = zeros(1, szOutput);

% initial vectors
o_init          = zeros(szOutput, 1);
c_init          = zeros(szHidden, 1);

% activation functions
activation_z    = 'tanh';
activation_gate = 'sigmoid';
activation_h    = 'tanh';
activation_o    = 'sigmoid';

% threshold of E for accuracy
threshold       = 0.49;

% training
for iData = 1:nData
    if mod(iData, 100) == 0
        fprintf('[Training...] %d / %d\n', iData, nData);
    end
    
    X           = datTrain_seq{iData}';
    Y           = datTrain_tgt{iData}';
    
    sequence_length     = size(X, 2); % #데이터길이 (hihell (ihello) -> 6)
    
    % [step 0] initialize
    % gate
    ccn_dat     = zeros(szHidden+szInput, sequence_length);
    ccn_net     = zeros(szHidden*4, sequence_length);
    
    gate_z      = zeros(szHidden, sequence_length);
    gate_i      = zeros(szHidden, sequence_length);
    gate_f      = zeros(szHidden, sequence_length);
    gate_o      = zeros(szHidden, sequence_length);    
    
    c           = zeros(szHidden, sequence_length);
    
    fun_c       = zeros(szHidden, sequence_length);
    h           = zeros(szHidden, sequence_length);

    net_o       = zeros(szOutput, sequence_length);
    o           = zeros(szOutput, sequence_length);
    
    
    % [step 1] feed forward
    for t = 1:sequence_length
        if t == 1       % t = 1 (w/ initial cell state)
            c_prev      = c_init;
            h_prev      = o_init;
        else            % t > 1 (w/ previous cell state)  
            c_prev      = c(:, t-1);
            h_prev      = h(:, t-1);
        end
        
        % concatenate data vector (4hX1)
        ccn_dat(:, t)   = [X(:, t); h_prev];
        
        % calculate net 
        ccn_net(:, t)   = ccn_W' * ccn_dat(:, t) + ccn_b';
        
        % split concatenated vector (z-i-f-o)
        gate_z(:, t)    = FUNC_ACTIVATION(ccn_net(0*szHidden+1 : 1*szHidden, t), activation_z);
        gate_i(:, t)    = FUNC_ACTIVATION(ccn_net(1*szHidden+1 : 2*szHidden, t), activation_gate);
        gate_f(:, t)    = FUNC_ACTIVATION(ccn_net(2*szHidden+1 : 3*szHidden, t), activation_gate);
        gate_o(:, t)    = FUNC_ACTIVATION(ccn_net(3*szHidden+1 : 4*szHidden, t), activation_gate);
 
        % cell state, LSTM output (h)
        c(:, t)         = gate_f(:, t) .* c_prev + gate_i(:, t) .* gate_z(:, t);
        
        fun_c(:, t)     = FUNC_ACTIVATION(c(:, t), activation_h);
        h(:, t)         = gate_o(:, t) .* fun_c(:, t);
        
        % output layer
        net_o(:, t)     = W_ho' * h(:, t) + b_o';
        o(:, t)         = FUNC_ACTIVATION(net_o(:, t), activation_o);
    end
    
    % [step 2] objective function - cross entrophy
    E(iData, 1)     = FUNC_OBJECTIVE( o, Y, objective_func);
    txto{iData, 1}  = FUNC_ONE_HOT_DECODING_TOY( o );
    
    
    if mod(iData, 100) == 0
        fprintf('TRUE: %s,    ESTIMAT: %s,     (%.3f)\n',...
            txtTrain{iData}(2:end), txto{iData, 1}, E (iData));
    end
    
%     keyboard;
    
    % [step 3] backpropagation through time (BPTT)
    % [step 3-1] initialize
    dE_dW       = zeros(size(ccn_W));
    dE_db       = zeros(size(ccn_b));
    
    dE_dWho     = zeros(size(W_ho));
    dE_dbo      = zeros(size(b_o));
    
    delta_h     = zeros(szHidden, sequence_length);
    delta_c     = zeros(szHidden, sequence_length);
    
    for t = sequence_length:-1:1
        if t == 1
            c_prev  = c_init;
        else
            c_prev  = c(:, t-1);
        end
        
        % output -> hidden
        dE_do       = FUNC_OBJECTIVE_PRIME( o(:, t), Y(:, t), objective_func );
        do_dneto    = FUNC_ACTIVATION_PRIME( '', o(:, t), activation_o);
        
        delta_o     = dE_do .* do_dneto;
        
        % output layer bias (b_o)
        dE_dbo      = dE_dbo + delta_o';
        
        % hidden->output weight (W_ho)
        dneto_dw    = h(:, t);
        dE_dWho     = dE_dWho + dneto_dw * delta_o';
        
        % [Hidden]
        % delta_h
        dneto_dh    = W_ho;
        delta_h(:, t)   = delta_h(:, t) + dneto_dh * delta_o;
        
        % delta_c
        dh_dfunc    = gate_o(:, t);
        dfunc_dc    = FUNC_ACTIVATION_PRIME( '', fun_c(:, t), activation_h );
        
        delta_c(:, t)   = delta_c(:, t) + delta_h(:, t) .* dfunc_dc;
        
        % z gate
        dc_dgz      = gate_i(:, t);
        dgz_dnetgz  = FUNC_ACTIVATION_PRIME( '', gate_z(:, t), activation_z );
                
        % i gate
        dc_dgi      = gate_z(:, t);
        dgi_dnetgi  = FUNC_ACTIVATION_PRIME( '', gate_i(:, t), activation_gate );
        
        % f gate
        dc_dgf      = c_prev;
        dgf_dnetgf  = FUNC_ACTIVATION_PRIME( '', gate_f(:, t), activation_gate );
        
        % o gate
        dh_dgo      = fun_c(:, t);
        dgo_dnetgo  = FUNC_ACTIVATION_PRIME( '', gate_o(:, t), activation_gate );
        
        
        ccn_dE_dnetg    = [ [delta_c(:, t) .* dc_dgz .* dgz_dnetgz];
                            [delta_c(:, t) .* dc_dgi .* dgi_dnetgi];
                            [delta_c(:, t) .* dc_dgf .* dgf_dnetgf];
                            [delta_h(:, t) .* dh_dgo .* dgo_dnetgo]];
                        
        % hidden layer bias (b_o)
        dE_db       = dE_db + ccn_dE_dnetg';
        
        % hidden->output weight (W_ho)
        dnetg_ddat  = ccn_dat(:, t);
        dE_dW       = dE_dW + dnetg_ddat * ccn_dE_dnetg';
        
        % t-1 시점에 미리 넣어두기
        if t > 1
            dc_dc       = gate_f(:, t);
            delta_c(:, t-1)     = dc_dc .* delta_c(:, t);

            dnetg_dh    = ccn_W(1*szHidden+1:2*szHidden, :);
            delta_h(:, t-1)     = dnetg_dh * ccn_dE_dnetg;
        end
    end
    
    % [step 4] weight, bias update
    
    ccn_W   = ccn_W - eta * dE_dW;
    ccn_b   = ccn_b - eta * dE_db;
    
    W_ho    = W_ho - eta * dE_dWho;
    b_o     = b_o - eta * dE_dbo;
end

plot(E)
line([1 nData], [threshold threshold], 'color', 'r');

accu_train   = sum(E < threshold) / nData * 100;


%% 

nData       = size(txtTest, 1);

CORR        = zeros(nData, 1);


for iData = 1:nData
    if mod(iData, 100) == 0
        fprintf('[Testing...] %d / %d\n', iData, nData);
    end
        
    X           = datTest_seq{iData}';
    Y           = datTest_tgt{iData}';
    
    sequence_length     = size(X, 2); % #데이터길이 (hihell (ihello) -> 6)
    
    % [step 0] initialize
    % gate
    ccn_dat     = zeros(szHidden+szInput, sequence_length);
    ccn_net     = zeros(szHidden*4, sequence_length);
    
    gate_z      = zeros(szHidden, sequence_length);
    gate_i      = zeros(szHidden, sequence_length);
    gate_f      = zeros(szHidden, sequence_length);
    gate_o      = zeros(szHidden, sequence_length);    
    
    c           = zeros(szHidden, sequence_length);
    
    fun_c       = zeros(szHidden, sequence_length);
    h           = zeros(szHidden, sequence_length);

    net_o       = zeros(szOutput, sequence_length);
    o           = zeros(szOutput, sequence_length);
    
    
    % [step 1] feed forward
    for t = 1:sequence_length
        if t == 1       % t = 1 (w/ initial cell state)
            c_prev      = c_init;
            h_prev      = o_init;
        else            % t > 1 (w/ previous cell state)  
            c_prev      = c(:, t-1);
            h_prev      = h(:, t-1);
        end
        
        % concatenate data vector (4hX1)
        ccn_dat(:, t)   = [X(:, t); h_prev];
        
        % calculate net 
        ccn_net(:, t)   = ccn_W' * ccn_dat(:, t) + ccn_b';
        
        % split concatenated vector (z-i-f-o)
        gate_z(:, t)    = FUNC_ACTIVATION(ccn_net(0*szHidden+1 : 1*szHidden, t), activation_z);
        gate_i(:, t)    = FUNC_ACTIVATION(ccn_net(1*szHidden+1 : 2*szHidden, t), activation_gate);
        gate_f(:, t)    = FUNC_ACTIVATION(ccn_net(2*szHidden+1 : 3*szHidden, t), activation_gate);
        gate_o(:, t)    = FUNC_ACTIVATION(ccn_net(3*szHidden+1 : 4*szHidden, t), activation_gate);
 
        % cell state, LSTM output (h)
        c(:, t)         = gate_f(:, t) .* c_prev + gate_i(:, t) .* gate_z(:, t);
        
        fun_c(:, t)     = FUNC_ACTIVATION(c(:, t), activation_h);
        h(:, t)         = gate_o(:, t) .* fun_c(:, t);
        
        % output layer
        net_o(:, t)     = W_ho' * h(:, t) + b_o';
        o(:, t)         = FUNC_ACTIVATION(net_o(:, t), activation_o);
    end
    
    E_te(iData, 1)     = FUNC_OBJECTIVE( o, Y, objective_func);
    txto_te{iData, 1}  = FUNC_ONE_HOT_DECODING_TOY( o );
    
    if mod(iData, 100) == 0
        fprintf('INPUT: %s,    TRUE: %s,    ESTIMAT: %s,     (%.3f)\n',...
            txtTest{iData}(1:end), txtTest{iData}(2:end), txto_te{iData, 1}, E_te(iData));
    end
%     
%     fprintf('%s   %s   %s\n',...
%         txtTest{iData}(1:end-1), txtTest{iData}(2:end), txto{iData, 1});
end

figure(2);
plot(E_te)
line([1 nData], [threshold threshold], 'color', 'r');

accu_test   = sum(E_te < threshold) / nData * 100