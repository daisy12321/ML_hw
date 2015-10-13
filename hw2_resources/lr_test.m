function [z, valid_score] = lr_test(name, lambda, toPlot)
disp('======Training======');
% load data from csv files

data = importdata(strcat('data/data_',name,'_train.csv'));

X = data(:,1:2);
Y = data(:,3);

% Carry out training.
%%% TODO %%%
X_train = [ones(size(Y)), data(:,1:2)];
Y_train = Y;
p = size(X_train,2);

LR_loss = @(w) lambda * norm(w(2:p), 2)^2 + sum(-log(sigmoid(Y_train .* (X_train * w'))));
% LR_grad = @(w) 2 * lambda * w  + sum((-repmat(Y_train, 1, p) .* X_train) ...
%               .* repmat(1 - sigmoid(Y_train .* (X_train * w')), 1, p));

% too slow below
% [x, f, x_hist, f_hist] = grad_desc(LR_loss, LR_grad, [0 0 0], 0.005, 1e-6)
%[x, f] = grad_desc_2(LR_loss, [1  1 1], 3, 1e-6)
% plot3(x_hist(:,1), x_hist(:,2), x_hist(:,3))
% plot(f_hist)
%%% compare to Matlab native function:
options = optimoptions(@fminunc,'Display','iter');
[z, f] = fminunc(LR_loss, [0, 0, 0], options)

% Define the predictLR(x) function, which uses trained parameters
predictLR = @(x) sigmoid(z(2:3) * x + z(1));



disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X_val = validate(:,1:2);
Y_val = validate(:,3);

Y_pred(predictLR(X_val')>.5) = 1;
Y_pred(predictLR(X_val')<=.5) = -1;
valid_score = sum(Y_pred == Y_val')/length(Y_val);
disp('======Accuracy======');
disp(valid_score);

if toPlot
    hold on;

    % plot training results
    plotDecisionBoundary(X, Y, predictLR, [0.7, 0.5, 0.3], ...
        strcat('LR Validate, lambda = ', num2str(lambda)), ...
        strcat('hw2_writeup/hw2_1_',name,'_a_',num2str(lambda),'.pdf'));


    % plot validation results
    plotDecisionBoundary(X_val, Y_val, predictLR, [0.7, 0.5, 0.3], ...
        strcat('LR Validate, lambda = ', num2str(lambda)), ...
        strcat('hw2_writeup/hw2_1_',name,'_b_',num2str(lambda),'.pdf'));
end