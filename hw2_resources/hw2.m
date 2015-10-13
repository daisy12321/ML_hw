addpath('srcs', 'hw2_resources', 'hw2_resources/data')
%%% 2.1 & 1.2 %%%%
% L2 regularized LR using gradient descent
% with lambda = 0
[w_reg1 valid_1] = lr_test('stdev1', 0, false);
[w_reg2 valid_2] = lr_test('stdev2', 0, false);
[w_reg3 valid_3] = lr_test('stdev4', 0, false);
[w_reg4 valid_4] = lr_test('nonsep', 0, false);
%%% 2.3 %%%%
% with regularized lambda
LAMBDA_RANGE = (-5:1:15);
valid_score_matrix = zeros(length(LAMBDA_RANGE), 4);

for i = 1:length(LAMBDA_RANGE)
    lambda_i = 10^(LAMBDA_RANGE(i));
    [w_reg1_1 valid_score_matrix(i, 1)]= lr_test('stdev1', lambda_i, false);
    [w_reg2_1 valid_score_matrix(i, 2)]= lr_test('stdev2', lambda_i, false);
    [w_reg3_1 valid_score_matrix(i, 3)]= lr_test('stdev4', lambda_i, false);
    [w_reg4_1 valid_score_matrix(i, 4)]= lr_test('nonsep', lambda_i, false);
end
valid_score_matrix
