addpath('srcs', 'hw3_resources')
%set(gcf,'Visible','off')              % turns current figure "off"
%set(0,'DefaultFigureVisible','off');  % all subsequent figures "off"

%%% 3.1 Neuro networks %%%%
X = [1, 2; 4, 5; 6, 9; 8, 11; 11, 14; 15, 20];
Y = [1, 0; 0, 1; 0, 1; 1, 0; 1, 0; 1,0];

[N, D] = size(X);
[N, K] = size(Y);

M = 3;
w1_0 = rand(M,D);
w2_0 = rand(K,M);

[w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X, Y, 1, 0.001);
[w1_est, w2_est] = grad_desc_3(@ANN_loss, w1_0, w2_0, X, Y, 1, 0.001);

predict_ANN = @(x) predict_multi_class(x, w1_est, w2_est);
[predict_all predict_class] = predict_ANN(X);
accu = sum(predict_class' == Y(:,1))/length(Y(:,1))

%% 3.2.4 Toy Problem
name = 'toy_multiclass_1';
[X_train, Y_train_lab, Y_train, X_valid, Y_valid_lab, Y_valid, X_test, Y_test_lab, Y_test] = read_data(name);
[N, D] = size(X_train);
K = size(Y_train, 2);

% size of hidden units
M = 3;
w1_0 = 5*rand(M,D);
w2_0 = 5*rand(K,M);

[w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X_train, Y_train, 1, 0.001)
[w1_est, w2_est] = grad_desc_3(@ANN_loss, w1_0, w2_0, X_train, Y_train, 1, 0.01)

predict_ANN = @(x) predict_multi_class(x, w1_est, w2_est);
% calculate accuracy in training set
[predict_all predict_class] = predict_ANN(X_train);
accu = sum(predict_class' == Y_train_lab)/length(Y_train_lab)
% calculate accuracy in test set
[predict_all predict_class] = predict_ANN(X_test);
accu = sum(predict_class' == Y_test_lab)/length(Y_test_lab)

%countsum = 0;
% for i = 1:size(test,1)
%     countsum = countsum + ((predictANN(test(i,1:2)) >= 0.5) == (test(i,3) == Yindex));
% end
%accu = countsum/size(test,1);
%disp(strcat('Accuracy is: ', num2str(accu)));


plotDecisionBoundary3Class(X_train(:, 2:3), Y_train, predict_ANN, [-0.05, 0.0, 0.05], '', strcat('hw3_writeup/plot_',name,'.pdf'))



%% 3.2.5 MNIST Data
train = importdata('hw3_resources/data/mnist_train.csv');
valid = importdata('hw3_resources/data/mnist_validate.csv');
test = importdata('hw3_resources/data/mnist_test.csv');