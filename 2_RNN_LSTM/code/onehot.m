function out = onehot(data,character)

    % data : nsample*1 ,cell 
    % character : 1*ncharacter ,cell
    nsample = size(data,1);
    ncharacter = size(character,2);    
    out = cell(nsample,1);
    
    for i = 1:nsample
        in = data{i};
        tmpout = zeros(length(in),ncharacter);
        for j = 1:length(in)
            tmp = {in(j)};
            tmptmp = repmat(tmp,7,1);
            tmpout(j,cellfun(@strcmp,tmptmp,character')) = 1;
        end
%         if istest
           tmpout = tmpout(1:end-1,:); 
%         end
        out(i) = {tmpout};
    end


end