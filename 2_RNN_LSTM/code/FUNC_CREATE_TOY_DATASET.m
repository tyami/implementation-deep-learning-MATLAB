function [TXT, TXT_ONEHOT, TARGET_TWOHOT] = FUNC_CREATE_TOY_DATASET(nsample)
% parameters
character = {'T', 'P', 'X', 'S', 'V', 'B', 'E'};

TXT = cell(nsample, 1);
TARGET_TWOHOT = cell(nsample, 1);

for i = 1:nsample
    [dummy_TXT, dummy_TARGET_TWOHOT] = make_embedrebber();
    TXT(i)  = {dummy_TXT};
    TARGET_TWOHOT(i)    = {dummy_TARGET_TWOHOT};
end

TXT_ONEHOT  = onehot(TXT,character);

end

% problem: ������ sequence �� ���� sequence �����ϱ�
% ex: input : HELLO , output: ELLO
% 0. train data, test data �����
% 1. LSTM �� �����ؼ� train ��, test
% 2. error �� sum of (true - predict)^2
% 3. internal weight value ��ȭ �׷��� ����
% 4. error trial ����
