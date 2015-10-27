addpath('srcs', 'hw2_resources', 'hw2_resources/data')
set(gcf,'Visible','off')              % turns current figure "off"
set(0,'DefaultFigureVisible','off');  % all subsequent figures "off"

%%% 2.1 & 1.2 %%%%
% L2 regularized LR using gradient descent
% with lambda = 0
train = zeros(1,4); valid = zeros(1,4); test = zeros(1,4);
str = {'stdev1','stdev2','stdev4','nonsep'};
for j = 1:4
  [w_reg, train(j), valid(j), test(j)] = lr_test(str{j}, 0, true);
end
vertcat(train, valid, test)

%%% 2.3 %%%%
% with regularized lambda
LAMBDA_RANGE = (-3:1:6);
valid_scores = zeros(length(LAMBDA_RANGE), 4);
test_scores = zeros(length(LAMBDA_RANGE), 4);

for j = 1:4
    for i = 1:length(LAMBDA_RANGE)
        lambda_i = 10^(LAMBDA_RANGE(i));
        [w_reg, tmp, valid_scores(i, j), test_scores(i, j)] = lr_test(str{j}, lambda_i, false);
    end
end
valid_scores
test_scores

fig = figure;
hold on;
plot(LAMBDA_RANGE, valid_scores(:,1))
plot(LAMBDA_RANGE, valid_scores(:,2))
plot(LAMBDA_RANGE, valid_scores(:,3))
plot(LAMBDA_RANGE, valid_scores(:,4))
%title('Validation accuracy versus \lambda')
h_legend = legend('stdev1', 'stdev2', 'stdev4', 'nonsep')
set(h_legend,'FontSize',14);
h_xlab = xlabel('\lambda in Log base 10');
h_ylab = ylabel('Validation accuracy');
set(h_xlab, 'FontSize',14);
set(h_ylab, 'FontSize',14);
set(fig,'Units','Inches');
pos = get(fig,'Position');
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig, 'hw2_writeup/hw2_1_cv.pdf', '-dpdf', '-r0')



%% just some plots with non-zero lambda
lr_test('stdev1', 10000, true)
lr_test('stdev2', 10000, true)
lr_test('stdev4', 10000, true)
lr_test('nonsep', 10000, true)
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

names=cellstr(['stdev1';'stdev2';'stdev4';'nonsep'])
train_array = [1:4];
valid_array = [1:4];
test_array = [1:4];
for i=1:4
    [train_array(i), valid_array(i), test_array(i)] = svm_test(names{i}, C, kernel, sigma2, toPlot);
end
train_array
valid_array
test_array

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
