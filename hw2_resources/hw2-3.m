addpath('srcs', 'hw2_resources', 'hw2_resources/data')
%%% 3.1 %%%%
name = 'titanic';

%[X, Y] = readdata(name, 'train', true);
data = importdata(strcat('data/data_',name,'_train.csv'));

C = 1;
X = data(:,1:11); 
Y = data(:,12);
[n, p] = size(X);
[X_scaled_1, X_mean, X_sd] = scale_std(X);
[X_scaled_2, X_min, denom] = scale_minmax(X);
[z_1, f] = lr_run(X_scaled_1, Y, 0, true)
[z_2, f] = lr_run(X_scaled_2, Y, 0, true)
[w1, w1_0] = svm(X_scaled_1, Y, C, 'dot', 1)
[w2, w2_0] = svm(X_scaled_2, Y, C, 'dot', 1)

%%
disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X_val = validate(:,1:11);
Y_val = validate(:,12);
for j = 1:p
    X_val_scaled_1(:, j) = (X_val(:,j) - X_mean(j))/X_sd(j);
    X_val_scaled_2(:, j) = (X_val(:,j) - X_min(j))/denom(j);
end
valid_score = get_accu(z_1, X_val_scaled_1, Y_val)
valid_score = get_accu(z_2, X_val_scaled_2, Y_val)
valid_score_svm = get_accu_svm(w1, w1_0, X_val_scaled_1, Y_val)
valid_score_svm = get_accu_svm(w2, w2_0, X_val_scaled_2, Y_val)
%%
disp('=====Testing======');
% load data from csv files
test = importdata(strcat('data/data_',name,'_test.csv'));
X_test = test(:,1:11);
Y_test = test(:,12);
for j = 1:p
    X_test_scaled_1(:, j) = (X_test(:,j) - X_mean(j))/X_sd(j);
    X_test_scaled_2(:, j) = (X_test(:,j) - X_min(j))/denom(j);
end
test_score = get_accu(z_1, X_test_scaled_1, Y_test)
test_score = get_accu(z_2, X_test_scaled_2, Y_test)
test_score_svm = get_accu_svm(w1, w1_0, X_test_scaled_1, Y_test)
test_score_svm = get_accu_svm(w2, w2_0, X_test_scaled_2, Y_test)

% find best lambda
LAMBDA_RANGE = (-5:1:6);
valid_scores_1 = zeros(length(LAMBDA_RANGE), 1);
valid_scores_2 = zeros(length(LAMBDA_RANGE), 1);
test_scores_1 = zeros(length(LAMBDA_RANGE), 1);
test_scores_2 = zeros(length(LAMBDA_RANGE), 1);

for i = 1:length(LAMBDA_RANGE)
    lambda_i = 10^(LAMBDA_RANGE(i));
    [z_1, f] = lr_run(X_scaled_1, Y, lambda_i, true);
    [z_2, f] = lr_run(X_scaled_2, Y, lambda_i, true);
    valid_scores_1(i) = get_accu(z_1, X_val_scaled_1, Y_val);
    valid_scores_2(i) = get_accu(z_2, X_val_scaled_2, Y_val);
    
    test_scores_1(i) = get_accu(z_1, X_test_scaled_1, Y_test);
    test_scores_2(i) = get_accu(z_2, X_test_scaled_2, Y_test);
end
[valid_scores_1, valid_scores_2]
[test_scores_1, test_scores_2]

%%
% find best C
C_RANGE = (-3:1:6);
valid_scores_svm_1 = zeros(length(C_RANGE), 1);
valid_scores_svm_2 = zeros(length(C_RANGE), 1);
test_scores_svm_1 = zeros(length(C_RANGE), 1);
test_scores_svm_2 = zeros(length(C_RANGE), 1);

%%
for i = 1:length(C_RANGE)
    C = 10^(C_RANGE(i));
    [w1, w1_0] = svm(X_scaled_1, Y, C, 'dot', 1);
    [w2, w2_0] = svm(X_scaled_2, Y, C, 'dot', 1);
    valid_scores_svm_1(i) = get_accu_svm(w1, w1_0, X_val_scaled_1, Y_val);
    valid_scores_svm_2(i) = get_accu_svm(w2, w2_0, X_val_scaled_2, Y_val);
    
    test_scores_svm_1(i) = get_accu_svm(w1, w1_0, X_test_scaled_1, Y_test);
    test_scores_svm_2(i) = get_accu_svm(w2, w2_0, X_test_scaled_2, Y_test);
end
[C_RANGE', valid_scores_svm_1, valid_scores_svm_2]
[C_RANGE', test_scores_svm_1, test_scores_svm_2]


%%
fig = figure;
hold on;
plot(LAMBDA_RANGE, valid_scores_1)
plot(LAMBDA_RANGE, valid_scores_2)
title('Validation accuracy versus \lambda')
legend('Stdev scaling', 'Min-max scaling')
xlabel('\lambda in Log')
ylabel('Validation accuracy')
set(fig,'Units','Inches');
pos = get(fig,'Position');
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig, 'hw2_writeup/hw2_3_cv.pdf', '-dpdf', '-r0')

%%
fig = figure;
hold on;
plot(C_RANGE, valid_scores_svm_1)
plot(C_RANGE, valid_scores_svm_2)
title('Validation accuracy versus C')
legend('Stdev scaling', 'Min-max scaling')
xlabel('C in Log')
ylabel('Validation accuracy')
set(fig,'Units','Inches');
pos = get(fig,'Position');
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig, 'hw2_writeup/hw2_3_svm_cv.pdf', '-dpdf', '-r0')
%%
%all = vertcat(data, validate, test) 
%tabulate(all(:,4))


[z_1, f] = lr_run(X_scaled_1, Y, 10, true);
test_score = get_accu(z_1, X_test_scaled_1, Y_test)
[w1, w1_0] = svm(X_scaled_1, Y, 0.01, 'dot', 1);
test_score_svm = get_accu_svm(w1, w1_0, X_test_scaled_1, Y_test)
[w2, w2_0] = svm(X_scaled_2, Y, 0.1, 'dot', 1);
test_score_svm_2 = get_accu_svm(w2, w2_0, X_test_scaled_2, Y_test)