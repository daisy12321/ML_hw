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
[w1_est, w2_est] = grad_desc_3(@ANN_loss, w1_0, w2_0, X, Y, 0.001, 1, 1e-5, 3000);

predict_ANN = @(x) predict_multi_class(x, w1_est, w2_est);
[predict_all predict_class] = predict_ANN(X);
accu = sum(predict_class' == Y_lab)/length(Y_lab)

plotDecisionBoundary3Class(X(:,2:3), Y, predict_ANN, [0.0, 0.0], '', strcat('hw3_writeup/plot_test.pdf'))



%% 3.2.4 Toy Problem
name = 'toy_multiclass_1';
[X_train, Y_train_lab, Y_train, X_valid, Y_valid_lab, Y_valid, X_test, Y_test_lab, Y_test] = read_data(name);
[N, D] = size(X_train);
K = size(Y_train, 2);

% size of hidden units
M = 1;
% set initial weights
rng(10);
w1_0 = 0.1*rand(M,D)-0.5;
w2_0 = 0.1*rand(K,M+1)-0.5;

% batch gradient descent
[w1_est, w2_est] = grad_desc_3(@ANN_loss, w1_0, w2_0, X_train, Y_train, 0.001, 1 ,1e-5, 3000);
test_accu  = get_accu_ANN(w1_est, w2_est, X_test, Y_test_lab)

% stochastic gradient descent
[w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X_train, Y_train, 0.001, 25, 1e-5, 3000);
valid_accu  = get_accu_ANN(w1_est, w2_est, X_valid, Y_valid_lab)
test_accu  = get_accu_ANN(w1_est, w2_est, X_test, Y_test_lab)

% cross validation
LAMBDA_RANGE = [0, 0.001, 0.01, 0.1];
M_RANGE = 6;
valid_accu = zeros(size(LAMBDA_RANGE, 2), M_RANGE);
% what to do when not converge in iteration limits?

for M = 2:6
    for i = 1:size(LAMBDA_RANGE,2)
        rng(10);
        w1_0 = 0.1*rand(M,D)-0.5;
        w2_0 = 0.1*rand(K,M+1)-0.5;

        lambda = LAMBDA_RANGE(i);
        [w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X_train, Y_train, lambda, 25, 1e-5, 3000);
        valid_accu(i, M) = get_accu_ANN(w1_est, w2_est, X_valid, Y_valid_lab);
    end
end
valid_accu
[w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X_train, Y_train, 0.001, 25, 1e-5, 3000);
test_accu  = get_accu_ANN(w1_est, w2_est, X_test, Y_test_lab)

plotDecisionBoundary3Class(X_train(:, 2:3), Y_train, @(x) predict_multi_class(x, w1_est, w2_est), [0.0, 0.0], '', strcat('hw3_writeup/plot_',name,'.pdf'))



%% 3.2.5 MNIST Data
name = 'mnist';
[X_train, Y_train_lab, Y_train, X_valid, Y_valid_lab, Y_valid, X_test, Y_test_lab, Y_test] = read_data(name);
%X_train = X_train(:, sum(X_train) ~= 0);

[N, D] = size(X_train);
K = size(Y_train, 2);

% size of hidden units
M = 40;
w1_0 = 10*rand(M,D)-5;
w2_0 = 10*rand(K,M+1)-5;

% stochastic gradient descent
[w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X_train, Y_train, 0.1, 20, 1e-5, 6000);
test_accu  = get_accu_ANN(w1_est, w2_est, X_test, Y_test_lab)
train_accu  = get_accu_ANN(w1_est, w2_est, X_train, Y_train_lab)


