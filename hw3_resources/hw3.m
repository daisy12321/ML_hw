addpath('srcs', 'hw3_resources', 'hw2_resources')
%set(gcf,'Visible','off')              % turns current figure "off"
%set(0,'DefaultFigureVisible','off');  % all subsequent figures "off"

%%% 3.1 Neuro networks %%%%
X = [1, 2; 4, 5; 6, 9; 8, 11; 11, 14; 15, 20];
Y = [1, 0; 0, 1; 0, 1; 1, 0; 1, 0; 1,0];

[N, D] = size(X);
[N, K] = size(Y);

M = 3;
w1_0 = ones(M,D);
w2_0 = ones(K,M);

[w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X, Y, 1, 0.001);

%% 3.2.4 Toy Problem
[X_train, Y_train, X_valid, Y_valid, X_test, Y_test] = read_data('toy_multiclass_1');
[N, D] = size(X_train);
K = size(Y_train, 2);

% size of hidden units
M = 3;
w1_0 = rand(M,D);
w2_0 = rand(K,M);

[w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X_train, Y_train, 1, 0.01)
[w1_est, w2_est] = grad_desc_3(@ANN_loss, w1_0, w2_0, X_train, Y_train, 1, 0.01)
predictANN = @(x) sigmoid(fwd_prop(x, w1_est, w2_est));


% calculate accuracy in training set
predict_all = zeros(K, N);
predict_class = zeros(1, N);
for i = 1:size(test,1)
    predict_all(:, i) = predictANN(X_train(i, :));
    [tmp, predict_class(i)] = max(predict_all(:,i));
end
predict_class


% calculate accuracy in test set
predict_all = zeros(K, N);
predict_class = zeros(1, N);
for i = 1:size(test,1)
    predict_all(:, i) = predictANN(X_test(i, :));
    [tmp, predict_class(i)] = max(predict_all(:,i));
end
predict_class

%countsum = 0;
% for i = 1:size(test,1)
%     countsum = countsum + ((predictANN(test(i,1:2)) >= 0.5) == (test(i,3) == Yindex));
% end
%accu = countsum/size(test,1);
%disp(strcat('Accuracy is: ', num2str(accu)));


plotDecisionBoundary(X, Y, predictANN, [0.3, 0.5, 0.7], '', strcat('hw3_writeup/toy_1_class_3.pdf'))

train = importdata('hw3_resources/data/toy_multiclass_2_train.csv');
valid = importdata('hw3_resources/data/toy_multiclass_2_validate.csv');
test = importdata('hw3_resources/data/toy_multiclass_2_test.csv');

%% 3.2.5 MNIST Data
train = importdata('hw3_resources/data/mnist_train.csv');
valid = importdata('hw3_resources/data/mnist_validate.csv');
test = importdata('hw3_resources/data/mnist_test.csv');