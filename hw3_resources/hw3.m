addpath('srcs', 'hw3_resources')
%set(gcf,'Visible','off')              % turns current figure "off"
%set(0,'DefaultFigureVisible','off');  % all subsequent figures "off"

%%% 3.1 Neuro networks %%%%
% X = [1, 2; 4, 5; 6, 9; 8, 11; 11, 10; 15, 14];
% X = [ones(6, 1) X];
% Y_lab = [1; 1; 2; 2; 3; 3];
% Y = dummyvar(Y_lab);
% [N, D] = size(X);
% [N, K] = size(Y);
% 
% M = 2;
% 
% w1_0 = rand(M,D);
% w2_0 = rand(K,M+1);
% 
% [w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X, Y, 0.1, 1, 1e-5, 3000);
% [w1_est, w2_est] = grad_desc_3(@ANN_loss, w1_0, w2_0, X, Y, 0.001, 1, 1e-5, 3000);
% 
% predict_ANN = @(x) predict_multi_class(x, w1_est, w2_est);
% [predict_all predict_class] = predict_ANN(X);
% accu = sum(predict_class' == Y_lab)/length(Y_lab)
% 
% plotDecisionBoundary3Class(X(:,2:3), Y, predict_ANN, [0.0, 0.0], '', strcat('hw3_writeup/plot_test.pdf'))



%% 3.2.4 Toy Problem
name = 'toy_multiclass_2';
[X_train, Y_train_lab, Y_train, X_valid, Y_valid_lab, Y_valid, X_test, Y_test_lab, Y_test] = read_data(name);
[N, D] = size(X_train);
K = size(Y_train, 2);

% size of hidden units
M = 4;
% set initial weights
rng(10);
w1_0 = 0.1*rand(M,D)-0.05;
w2_0 = 0.1*rand(K,M+1)-0.05;

% batch gradient descent
[w1_est, w2_est] = grad_desc_3(@ANN_loss, w1_0, w2_0, X_train, Y_train, 0.001, 1 ,1e-5, 3000);
test_accu  = get_accu_ANN(w1_est, w2_est, X_test, Y_test_lab)

% stochastic gradient descent
[w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X_train, Y_train, 0.01,1, 1e-6, 5000);
valid_accu  = get_accu_ANN(w1_est, w2_est, X_valid, Y_valid_lab)
test_accu  = get_accu_ANN(w1_est, w2_est, X_test, Y_test_lab)

% cross validation
LAMBDA_RANGE = [0.005];
M_RANGE = 6;
valid_accu = zeros(size(LAMBDA_RANGE, 2), M_RANGE);
% what to do when not converge in iteration limits?

for M = 2:6
    for i = 1:size(LAMBDA_RANGE,2)
        rng(10);
        w1_0 = 0.1*rand(M,D)-0.05;
        w2_0 = 0.1*rand(K,M+1)-0.05;

        lambda = LAMBDA_RANGE(i);
        [w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X_train, Y_train, lambda, 25, 1e-6, 6000);
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
M = 20;
rng(0);
w1_0 = .1*rand(M,D)-.05;
w2_0 = .1*rand(K,M+1)-.05;

% stochastic gradient descent
[w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X_train, Y_train, 1e-6, 10, 1e-6, 6000);
test_accu  = get_accu_ANN(w1_est, w2_est, X_test, Y_test_lab)
train_accu  = get_accu_ANN(w1_est, w2_est, X_train, Y_train_lab)

%% MNIST cross validation
name = 'mnist';
[X_train, Y_train_lab, Y_train, X_valid, Y_valid_lab, Y_valid, X_test, Y_test_lab, Y_test] = read_data(name);
[N, D] = size(X_train);
K = size(Y_train, 2);

% t = classregtree(X_train,Y_train_lab)
% view(t)
% Yfit = eval(t,X_valid)
% sum(Yfit == Y_valid_lab)/length(Y_valid_lab)

% cross validation
LAMBDA_RANGE = [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2];
M_RANGE = [5:5:30];
valid_accu = zeros(size(LAMBDA_RANGE, 2), length(M_RANGE));

%%
for m = 1:length(M_RANGE) % columns of matrix have constant M
    M = M_RANGE(m);
    disp(M)
    for i = 1:size(LAMBDA_RANGE,2) % rows of matrix have constant lambda
        rng(0);
        w1_0 = .1*rand(M,D)-.05;
        w2_0 = .1*rand(K,M+1)-.05;

        lambda = LAMBDA_RANGE(i);
        [w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X_train, Y_train, lambda, 10, 1e-6, 6000);
        valid_accu(i, m) = get_accu_ANN(w1_est, w2_est, X_valid, Y_valid_lab)
    end
end
valid_accu
% csvwrite('hw3_resources/valid_accu',valid_accu)
% for i= 1:size(LAMBDA_RANGE,2)
%     fprintf('%.4f & ',valid_accu(i,:))
%     fprintf('\\\\\n')
% end
%%

%%
% heatmap of neural network accuracy on the validation set
h = figure;
colormap('jet')
x = [5 40];
y = 2*[-5 -1];
imagesc(x, y, cv_matrix(2:end,:));
z = colorbar
ylabel(z, 'Lambda');
xlabel('M')
ylabel('Log_{10}(\lambda)') 
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h, 'hw3_writeup/mnist_cv_2.pdf', '-dpdf', '-r0')

%% New cross-validation: M = 35, 40
name = 'mnist';
[X_train, Y_train_lab, Y_train, X_valid, Y_valid_lab, Y_valid, X_test, Y_test_lab, Y_test] = read_data(name);
[N, D] = size(X_train);
K = size(Y_train, 2);
% cross validation
LAMBDA_RANGE = [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2];
M_RANGE = [35 40];
valid_accu_2 = zeros(size(LAMBDA_RANGE, 2), length(M_RANGE));

for m = 1:length(M_RANGE) % columns of matrix have constant M
    M = M_RANGE(m);
    disp(M)
    for i = 1:size(LAMBDA_RANGE,2) % rows of matrix have constant lambda
        rng(0);
        w1_0 = .1*rand(M,D)-.05;
        w2_0 = .1*rand(K,M+1)-.05;

        lambda = LAMBDA_RANGE(i);
        [w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X_train, Y_train, lambda, 10, 1e-6, 6000);
        valid_accu_2(i, m) = get_accu_ANN(w1_est, w2_est, X_valid, Y_valid_lab)
    end
end
valid_accu_2
% csvwrite('hw3_resources/valid_accu_2',valid_accu_2)
%%
cv_matrix_1 = csvread('hw3_resources/valid_accu');
cv_matrix_2 = csvread('hw3_resources/valid_accu_2');
cv_matrix = [cv_matrix_1, cv_matrix_2];
%%
% Find optimal pair lambda = 10^-4, M = 35
M = 35;
rng(0);
w1_0 = .1*rand(M,D)-.05;
w2_0 = .1*rand(K,M+1)-.05;
[w1_est, w2_est] = grad_desc_stoch(@ANN_loss, w1_0, w2_0, X_train, Y_train, 1e-4, 10, 1e-10, 6000);
test_accu  = get_accu_ANN(w1_est, w2_est, X_test, Y_test_lab)
train_accu  = get_accu_ANN(w1_est, w2_est, X_train, Y_train_lab)
valid_accu_1  = get_accu_ANN(w1_est, w2_est, X_valid, Y_valid_lab)