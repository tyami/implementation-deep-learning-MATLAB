function [TXT, TARGET_TWOHOT] = make_rebber()

TRANSIT_STR = { 'T','P';
                'X','S';
                'V','T'; 
                'X','S';
                'P','V';
                'E','E'  };  % Q is dummy character
TRANSIT_NUM = [ 2 3;
                4 2;
                5 3;
                3 6;
                4 6;
                0 0  ];
TRANSIT_TARGET_TWOHOT   = { [1 1 0 0 0 0 0];          % TP
                            [0 0 1 1 0 0 0];          % XS
                            [1 0 0 0 1 0 0];          % VT
                            [0 0 1 1 0 0 0];          % XS
                            [0 1 0 0 1 0 0];          % PV
                            [0 0 0 0 0 0 1]}';        % E
% order: 'T', 'P', 'X', 'S', 'V', 'B', 'E'

TXT         = 'B';
idx         = 1;
TARGET_TWOHOT        = [TRANSIT_TARGET_TWOHOT{:, idx}];

while 1
    if unifrnd(0,1)>0.5
        TXT = [TXT TRANSIT_STR{idx,1}];
        idx = TRANSIT_NUM(idx,1);
    else
        TXT = [TXT TRANSIT_STR{idx,2}];
        idx = TRANSIT_NUM(idx,2);
    end
    
    if idx == 0
        break;
    end
    
    TARGET_TWOHOT = [TARGET_TWOHOT; TRANSIT_TARGET_TWOHOT{:, idx}];
end

end