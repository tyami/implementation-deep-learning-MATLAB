function conditional_prob = FUNC_CALCULATE_CONDITIONAL_PROB( X, W, B, isEnd )
%conditional_prob = FUNC_CALCULATE_CONDITIONAL_PROB( X, W B, isEnd )
%   Coded by T. Yang

if nargin < 4
    isEnd   = 0;
end

if isEnd
    conditional_prob    = W*X + B;
else
    conditional_prob    = 1 ./ (1 + exp(-W*X - B));
end

end