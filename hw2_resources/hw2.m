addpath('srcs', 'hw2_resources', 'hw2_resources/data')
%%% 2.1 & 1.2 %%%%
% L2 regularized LR using gradient descent
% with lambda = 0
[w_reg1, train_1, valid_1, test_1] = lr_test('stdev1', 0, true);
[w_reg2, train_2, valid_2, test_2] = lr_test('stdev2', 0, true);
[w_reg3, train_3, valid_3, test_3] = lr_test('stdev4', 0, true);
[w_reg4, train_4, valid_4, test_4] = lr_test('nonsep', 0, true);
[train_1, valid_1, train_2, valid_2, train_3, valid_3, train_4, valid_4]
%%% 2.3 %%%%
% with regularized lambda
LAMBDA_RANGE = (-3:1:6);
valid_scores = zeros(length(LAMBDA_RANGE), 4);
test_scores = zeros(length(LAMBDA_RANGE), 4);

for i = 1:length(LAMBDA_RANGE)
    lambda_i = 10^(LAMBDA_RANGE(i));
    [w_reg1_1, tmp, valid_scores(i, 1), test_scores(i, 1)]= lr_test('stdev1', lambda_i, false);
    [w_reg2_1, tmp, valid_scores(i, 2), test_scores(i, 2)]= lr_test('stdev2', lambda_i, false);
    [w_reg3_1, tmp, valid_scores(i, 3), test_scores(i, 3)]= lr_test('stdev4', lambda_i, false);
    [w_reg4_1, tmp, valid_scores(i, 4), test_scores(i, 4)]= lr_test('nonsep', lambda_i, false);
end
valid_scores
test_scores

fig = figure;
hold on;
plot(LAMBDA_RANGE, valid_scores(:,1))
plot(LAMBDA_RANGE, valid_scores(:,2))
plot(LAMBDA_RANGE, valid_scores(:,3))
plot(LAMBDA_RANGE, valid_scores(:,4))
title('Validation accuracy versus \lambda')
legend('stdev1', 'stdev2', 'stdev4', 'nonsep')
xlabel('\lambda in Log')
ylabel('Validation accuracy')
set(fig,'Units','Inches');
pos = get(fig,'Position');
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig, 'hw2_writeup/hw2_1_cv.pdf', '-dpdf', '-r0')



%% just some plots with non-zero lambda
lr_test('stdev1', 100, true)
lr_test('nonsep', 1000, true)
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
sigma2 = 1;

names={'stdev1','stdev2','stdev4','nonsep'};
w_matrix=zeros(4,3);
train_array = [1:4];
valid_array = [1:4];
test_array = [1:4];
for i=1:4
    [w_matrix(i,:), train_array(i), valid_array(i), test_array(i)] = svm_test(names{i}, C, kernel, sigma2, toPlot);
end
[w_matrix, train_array', valid_array', test_array']

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

kernel = 'rbf';
toPlot = 0;
sigma2 = 1;
for i=1:4
    for C=[0.01,0.1,1,10,100]
        svm_test(names{i}, C, kernel, sigma2, toPlot);
    end
end

kernel = 'rbf';
toPlot = 1;
name = 'nonsep';
for sigma2=[0.01,0.1,10,100]
    for C=[0.01,0.1,1,10,100]
        svm_test(name, C, kernel, sigma2, toPlot);
    end
end
