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

% problem: 들어오는 sequence 의 다음 sequence 예측하기
% ex: input : HELLO , output: ELLO
% 0. train data, test data 만들기
% 1. LSTM 을 구현해서 train 후, test
% 2. error 는 sum of (true - predict)^2
% 3. internal weight value 변화 그래프 관찰
% 4. error trial 보기
