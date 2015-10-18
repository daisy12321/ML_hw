addpath('srcs', 'hw2_resources', 'hw2_resources/data')
%%% 2.1 & 1.2 %%%%
% L2 regularized LR using gradient descent
% with lambda = 0
[w_reg1, valid_1] = lr_test('stdev1', 0, true);
[w_reg2, valid_2] = lr_test('stdev2', 0, true);
[w_reg3, valid_3] = lr_test('stdev4', 0, true);
[w_reg4, valid_4] = lr_test('nonsep', 0, true);

%%% 2.3 %%%%
% with regularized lambda
LAMBDA_RANGE = (-5:1:5);
valid_score_matrix = zeros(length(LAMBDA_RANGE), 4);

for i = 1:length(LAMBDA_RANGE)
    lambda_i = 10^(LAMBDA_RANGE(i));
    [w_reg1_1, valid_score_matrix(i, 1)]= lr_test('stdev1', lambda_i, false);
    [w_reg2_1, valid_score_matrix(i, 2)]= lr_test('stdev2', lambda_i, false);
    [w_reg3_1, valid_score_matrix(i, 3)]= lr_test('stdev4', lambda_i, false);
    [w_reg4_1, valid_score_matrix(i, 4)]= lr_test('nonsep', lambda_i, false);
end
valid_score_matrix

%%
optim_ver = ver('optim');
optim_ver = str2double(optim_ver.Version);
if optim_ver >= 6
    opts = optimset('Algorithm', 'interior-point-convex');
else
    opts = optimset('Algorithm', 'interior-point', 'LargeScale', 'off', 'MaxIter', 2000);
end
%%

%%% 2.2.1 %%%
%%% Sample SVM problem 
X = [[1,2];[2,2];[0,0];[-2,3]];
Y = [1,1,-1,-1]';
C = 1;
[w, w_0, H] = svm(X, Y, C, 'dot', 0)
% Benchmark against native Matlab function (obtain same alphas)
% cl = fitcsvm(X, Y);
% cl.Alpha

%%
%%% 2.2.2 %%%
C = 1;
kernel = 'dot';
toPlot = 0;

names=cellstr(['stdev1';'stdev2';'stdev4';'nonsep'])
for i=1:4
    svm_test(names{i}, C, kernel, sigma2, toPlot);
end

%%
%%% 2.2.3 %%%
X = [[1,2];[2,2];[0,0];[-2,3]];
Y = [1,1,-1,-1]';
C = 1;
sigma2 = 1;
[w, w_0, H, alpha] = svm(X, Y, C, 'rbf', sigma2)


kernel = 'dot';
toPlot = 0;
for i=1:4
    for C=[0.01,0.1,1,10,100]
        svm_test(names{i}, C, kernel, sigma2, toPlot);
    end
end

kernel = 'dot';
toPlot = 0;
for i=1:4
    for C=[0.01,0.1,1,10,100]
        svm_test(names{i}, C, kernel, sigma2, toPlot);
    end
end
