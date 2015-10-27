addpath('srcs', 'hw2_resources', 'hw2_resources/data')
%%% 3.1 %%%%
name = 'titanic';

%[X, Y] = readdata(name, 'train', true);
data = importdata(strcat('data/data_',name,'_train.csv'));

X = data(:,1:11); 
[n, p] = size(X);
X_scaled = scale(X);
Y = data(:,12);

[z, f] = lr_run(X_scaled, Y, 0, true)
predictLR = @(x) sigmoid(z(2:p+1) * x + z(1));

disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X_val = validate(:,1:11);
X_val_scaled = scale(X_val);
Y_val = validate(:,12);
valid_score = get_accu(z, X_val_scaled, Y_val)



disp('=====Testing======');
% load data from csv files
test = importdata(strcat('data/data_',name,'_test.csv'));
X_test = test(:,1:11);
X_test_scaled = scale(X_test);
Y_test = test(:,12);
test_score = get_accu(z, X_test_scaled, Y_test)

% find best lambda
LAMBDA_RANGE = (-3:1:6);
valid_scores = zeros(length(LAMBDA_RANGE), 1);
test_scores = zeros(length(LAMBDA_RANGE), 1);

for i = 1:length(LAMBDA_RANGE)
    lambda_i = 10^(LAMBDA_RANGE(i));
    [z, f] = lr_run(X_scaled, Y, lambda_i, true)
    valid_scores(i) = get_accu(z, X_val_scaled, Y_val)
    test_scores(i) = get_accu(z, X_test_scaled, Y_test)
end
valid_scores
test_scores

fig = figure;
hold on;
plot(LAMBDA_RANGE, valid_scores)
title('Validation accuracy versus \lambda')
xlabel('\lambda in Log')
ylabel('Validation accuracy')
set(fig,'Units','Inches');
pos = get(fig,'Position');
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig, 'hw2_writeup/hw2_3_cv.pdf', '-dpdf', '-r0')


[z, f] = lr_run(X_scaled, Y, 0.1, true);
test_score = get_accu(z, X_test_scaled, Y_test)
