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
train = importdata('hw3_resources/data/toy_multiclass_1_train.csv');
valid = importdata('hw3_resources/data/toy_multiclass_1_validate.csv');
test = importdata('hw3_resources/data/toy_multiclass_1_test.csv');

X = train(:,1:2);
Y_init = train(:,3);
[N, D] = size(X);
K = max(Y_init);
Y = zeros(N,K);
Y(:,1) = [Y_init == ones(N,1)]
Y(:,2) = [Y_init == 2*ones(N,1)]
Y(:,3) = [Y_init == 3*ones(N,1)]
[N, D] = size(X);
M = 3;
K=1;
w1_0 = ones(M,D);
w2_0 = ones(K,M);
[w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X, Y(:,1), 1, 0.001)

predictANN = @(x) sigmoid(fwd_prop(x, w1_est, w2_est));
%predictANN(X(2,:))

plotDecisionBoundary(X, Y(:,1), predictANN, [0.3, 0.5, 0.7], '', strcat('hw3_writeup/toy_1_class_3.pdf'))

train = importdata('hw3_resources/data/toy_multiclass_2_train.csv');
valid = importdata('hw3_resources/data/toy_multiclass_2_validate.csv');
test = importdata('hw3_resources/data/toy_multiclass_2_test.csv');

%% 3.2.5 MNIST Data
train = importdata('hw3_resources/data/mnist_train.csv');
valid = importdata('hw3_resources/data/mnist_validate.csv');
test = importdata('hw3_resources/data/mnist_test.csv');