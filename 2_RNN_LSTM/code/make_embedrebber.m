function [TXT, TARGET_TWOHOT] = make_embedrebber()
% order: 'T', 'P', 'X', 'S', 'V', 'B', 'E'

TRANSIT_TWOHOT  = { [1 1 0 0 0 0 0];   % TP
                    [0 0 0 0 0 1 0];   % B
                    [1 0 0 0 0 0 0];   % T
                    [0 0 0 0 0 0 1];   % E
                    [0 1 0 0 0 0 0]}'; % P
                
[TXT, TARGET_TWOHOT]   = make_rebber();

if unifrnd(0,1)>0.5
    TXT = ['BT' TXT 'TE']; % target: TP-B, T-E
    TARGET_TWOHOT   = [ TRANSIT_TWOHOT{:, 1};
                        TRANSIT_TWOHOT{:, 2};
                        TARGET_TWOHOT;
                        TRANSIT_TWOHOT{:, 3};
                        TRANSIT_TWOHOT{:, 4}  ];
else
    TXT = ['BP' TXT 'PE']; % target: TP-B, P-E
    TARGET_TWOHOT   = [ TRANSIT_TWOHOT{:, 1};
                        TRANSIT_TWOHOT{:, 2};
                        TARGET_TWOHOT;
                        TRANSIT_TWOHOT{:, 5};
                        TRANSIT_TWOHOT{:, 4}  ];
end


end