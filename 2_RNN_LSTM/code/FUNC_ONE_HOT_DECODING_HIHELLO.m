function [ data ] = FUNC_ONE_HOT_DECODING_HIHELLO( data_onehot )
%UNTITLED3 �� �Լ��� ��� ���� ��ġ
%   �ڼ��� ���� ��ġ
    data_onehot = data_onehot';
    
    sz = size(data_onehot, 1);
    data = '';
    
    for i = 1:sz
        [~,idx] = max(data_onehot(i, :));
        switch idx
            case 1
                data(i)  = 'h';
            case 2
                data(i)  = 'i';
%                 data(i)  = 'e';
            case 3
                data(i)  = 'e';
%                 data(i)  = 'l';
            case 4
                data(i)  = 'l';
%                 data(i)  = 'o';
            case 5
                data(i)  = 'o';
        end
    end

end

