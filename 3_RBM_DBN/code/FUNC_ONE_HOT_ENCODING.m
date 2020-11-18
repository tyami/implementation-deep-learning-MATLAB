function [ trueY_onehot ] = FUNC_ONE_HOT_ENCODING( nClass, trueY )
%[ trueY_onehot ] = FUNC_ONE_HOT_ENCODING( trueY )

dummy           = eye(nClass);
trueY_onehot    = dummy(:, trueY);

end

