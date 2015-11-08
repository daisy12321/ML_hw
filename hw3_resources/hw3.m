addpath('srcs', 'hw3_resources')
%set(gcf,'Visible','off')              % turns current figure "off"
%set(0,'DefaultFigureVisible','off');  % all subsequent figures "off"

%%% 3.1 Neuro networks %%%%
X = [1, 2; 4, 5; 6, 9; 8, 11; 11, 10; 15, 14];
X = [ones(6, 1) X];
Y_lab = [1; 1; 2; 2; 3; 3];
Y = dummyvar(Y_lab);
[N, D] = size(X);
[N, K] = size(Y);

M = 2;

w1_0 = rand(M,D);
w2_0 = rand(K,M+1);

[w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X, Y, 0.1, 1, 1e-5, 3000);
[w1_est, w2_est] = grad_desc_3(@ANN_loss, w1_0, w2_0, X, Y, 0.1, 1, 1e-5, 3000);

predict_ANN = @(x) predict_multi_class(x, w1_est, w2_est);
[predict_all predict_class] = predict_ANN(X);
accu = sum(predict_class' == Y_lab)/length(Y_lab)

plotDecisionBoundary3Class(X(:,2:3), Y, predict_ANN, [-0.05, 0.0, 0.05], '', strcat('hw3_writeup/plot_test.pdf'))



%% 3.2.4 Toy Problem
name = 'toy_multiclass_2';
[X_train, Y_train_lab, Y_train, X_valid, Y_valid_lab, Y_valid, X_test, Y_test_lab, Y_test] = read_data(name);
[N, D] = size(X_train);
K = size(Y_train, 2);

% size of hidden units
M = 3;
w1_0 = 10*rand(M,D)-5;
w2_0 = 10*rand(K,M+1)-5;

% batch gradient descent
[w1_est, w2_est] = grad_desc_3(@ANN_loss, w1_0, w2_0, X_train, Y_train, 0.001, 1 ,1e-5, 3000);

% stochastic gradient descent
[w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X_train, Y_train, 0.001, 25, 1e-5, 3000);
predict_ANN = @(x) predict_multi_class(x, w1_est, w2_est);
% calculate accuracy in training set
[predict_all, predict_class] = predict_ANN(X_train);
train_accu = sum(predict_class' == Y_train_lab)/length(Y_train_lab)
% calculate accuracy in test set
[predict_all, predict_class] = predict_ANN(X_test);
test_accu = sum(predict_class' == Y_test_lab)/length(Y_test_lab)

plotDecisionBoundary3Class(X_train(:, 2:3), Y_train, predict_ANN, [0.0, 0.0], '', strcat('hw3_writeup/plot_',name,'.pdf'))



%% 3.2.5 MNIST Data
name = 'mnist';
[X_train, Y_train_lab, Y_train, X_valid, Y_valid_lab, Y_valid, X_test, Y_test_lab, Y_test] = read_data(name);
%X_train = X_train(:, sum(X_train) ~= 0);

[N, D] = size(X_train);
K = size(Y_train, 2);

% size of hidden units
M = 4;
w1_0 = 10*rand(M,D)-5;
w2_0 = 10*rand(K,M+1)-5;

% stochastic gradient descent
[w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X_train, Y_train, 100, 0.1, 1e-5, 6000);
predict_ANN = @(x) predict_multi_class(x, w1_est, w2_est);
% calculate accuracy in training set
[predict_all, predict_class] = predict_ANN(X_train);
train_accu = sum(predict_class' == Y_train_lab)/length(Y_train_lab)
% calculate accuracy in test set
[predict_all, predict_class] = predict_ANN(X_test);
test_accu = sum(predict_class' == Y_test_lab)/length(Y_test_lab)
%disp(strcat('Accuracy is: ', num2str(accu)));
