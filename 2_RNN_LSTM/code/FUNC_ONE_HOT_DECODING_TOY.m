function [ data ] = FUNC_ONE_HOT_DECODING_TOY( data_onehot )
%UNTITLED3 이 함수의 요약 설명 위치
%   자세한 설명 위치
    character = {'T', 'P', 'X', 'S', 'V', 'B', 'E'};
    data_onehot = data_onehot';
    
    sz = size(data_onehot, 1);
    data = '';
    
    for i = 1:sz
        [~,idx] = max(data_onehot(i, :));
        
        switch idx
            case 1
                data(i)  = 'T';
            case 2
                data(i)  = 'P';
            case 3
                data(i)  = 'X';
            case 4
                data(i)  = 'S';
            case 5
                data(i)  = 'V';
            case 6
                data(i)  = 'B';
            case 7
                data(i)  = 'E';
        end
    end

end

