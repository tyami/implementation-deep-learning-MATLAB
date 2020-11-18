function [ data_onehot ] = FUNC_ONE_HOT_ENCODING_HIHELLO( data )
%UNTITLED3 이 함수의 요약 설명 위치
%   자세한 설명 위치
    sz = length(data);
    data_onehot = zeros(sz, 5);
%     data_onehot = zeros(sz, 4);
    
    for i = 1:sz
        switch data(i)
            case 'h'
                data_onehot(i, :)  = [1 0 0 0 0];
%                 data_onehot(i, :)  = [1 0 0 0];
            case 'i'
                data_onehot(i, :)  = [0 1 0 0 0];
            case 'e'
                data_onehot(i, :)  = [0 0 1 0 0];
%                 data_onehot(i, :)  = [0 1 0 0];
            case 'l'
                data_onehot(i, :)  = [0 0 0 1 0];
%                 data_onehot(i, :)  = [0 0 1 0];
            case 'o'
                data_onehot(i, :)  = [0 0 0 0 1];
%                 data_onehot(i, :)  = [0 0 0 1];
        end
    end
data_onehot = data_onehot';

end

