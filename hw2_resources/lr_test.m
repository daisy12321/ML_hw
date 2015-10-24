function [z, train_score, valid_score, test_score] = lr_test(name, lambda, toPlot)
disp('======Training======');
% load data from csv files
data = importdata(strcat('data/data_',name,'_train.csv'));
X = data(:,1:2);
Y = data(:,3);

% Carry out training.
[z, f] = lr_run(X, Y, lambda, true);


% Define the predictLR(x) function, which uses trained parameters
predictLR = @(x) sigmoid(z(2:3) * x + z(1));

%Y_train_pred(predictLR(X')>.5) = 1;
%Y_train_pred(predictLR(X')<=.5) = -1;
train_score = get_accu(z, X, Y);


disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X_val = validate(:,1:2);
Y_val = validate(:,3);

%Y_pred(predictLR(X_val')>.5) = 1;
%Y_pred(predictLR(X_val')<=.5) = -1;

%train_score = sum(Y_train_pred == Y')/length(Y);
%valid_score = sum(Y_pred == Y_val')/length(Y_val);
disp('======Accuracy======');
valid_score = get_accu(z, X_val, Y_val);
disp(valid_score);

disp('======Testing======');
test = importdata(strcat('data/data_',name,'_test.csv'));
X_test = test(:,1:2);
Y_test = test(:,3);
test_score = get_accu(z, X_test, Y_test);
disp(test_score);

if toPlot
    hold on;

    % plot training results
    plotDecisionBoundary(X, Y, predictLR, [0.7, 0.5, 0.3], ...
        strcat('LR Training, lambda = ', num2str(lambda)), ...
        strcat('hw2_writeup/hw2_1_',name,'_a_',num2str(lambda),'.pdf'));


    % plot validation results
    plotDecisionBoundary(X_val, Y_val, predictLR, [0.7, 0.5, 0.3], ...
        strcat('LR Validate, lambda = ', num2str(lambda)), ...
        strcat('hw2_writeup/hw2_1_',name,'_b_',num2str(lambda),'.pdf'));
end