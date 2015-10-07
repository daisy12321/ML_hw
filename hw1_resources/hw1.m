addpath('srcs', 'hw2_resources', 'hw2_resources/data')
%%% 1.1 & 1.2 %%%%
% self-implement function on gradient descend
% constant step size
% convex quadratic function
fun = @(x) 3*x(1)^2 + 2*x(1)*x(2) + x(2)^2 - 4*x(1) + 5*x(2)
grad = @(x)[6*x(1) + 2*x(2) - 4, 2*x(1) + 2*x(2) + 5]
[x, f, x_hist] = grad_desc(fun, grad, [-10.0 0.0], 0.1, 1e-10)
% plot the contour and the optimum point
h = figure;
ezcontour('3*x^2 + 2*x*y + y^2 - 4*x + 5*y', [-10, 10], [-10, 10])
hold on;
plot(x_hist(:,1), x_hist(:,2), 'b--o');
plot(x(1), x(2), '+','color','r');
hold off;
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h, 'hw1_1.pdf', '-dpdf', '-r0')

% negative of guassian pdf
sigma = 1
mu = 0
fun2 = @(x) -1/(2*sigma*pi)*exp(-(x-mu)^2/(2*sigma^2))
grad2 = @(x) (x-mu)/(sqrt(2*pi)*sigma^3)*exp(-(x-mu)^2/(2*sigma^2))
[x, f] = grad_desc(fun2, grad2, 0.1, 0.01, 1e-10) % if convergence size small, incorrect results
ezplot(fun2)

% non-convex function
fun3 = @(x) x^3 - 10* x 
grad3 = @(x) 3*x^2 - 10
% local minimum
[x, f] = grad_desc(fun3, grad3, 0.1, 0.01, 1e-10)
% true global minimum 
[x, f] = grad_desc(fun3, grad3, -13, 0.01, 1e-10)
ezplot(fun3)


%%% 1.3 %%%%
%%% implement finite differencing to evaluate gradient
grad_numeric = fin_diff(fun, [1, 2]', 0.001)
grad_analytic = grad([1, 2]) %compare with analytic 

% test on the previous examples
[x, f] = grad_desc(fun, grad, [1.0 -2.0], 0.1, 1e-10)
[x, f] = grad_desc_2(fun, [1.0 -2.0]', 0.1, 1e-10)

[x, f] = grad_desc(fun2, grad2, 0.1, 0.1, 1e-10)
[x, f] = grad_desc_2(fun2, 0.1, 0.1, 1e-10)

[x, f] = grad_desc(fun3, grad3, 0.1, 0.01, 1e-10)
[x, f] = grad_desc_2(fun3, 0.1, 0.01, 1e-10)
% all match!


%%% 1.4 %%%%
%%% compare to Matlab native function:
options = optimoptions(@fminunc,'Display','iter');
[x, f] = fminunc(fun, [1,1], options)
% 8 iterations, versus ours 95 iterations

[x, f] = fminunc(fun2, 1, options)
% 3 iterations, versus ours 165 iterations


%%%%%%%%%%%%
%%% 2.1 %%%%
global X X_full Y M;
[X, Y] = bishopCurveData();
M = 3;
X_full = bishopXPoly(X, M);
% Using our function
w = bishopCurveFit(X_full, Y);
% using polyfit to verify
w_polyfit = fliplr(polyfit(X, Y, M));
% Same as w!

% replicate Bishop 1.4 plots
X_full = bishopXPoly(X, 0);
w_0 = bishopCurveFit(X_full, Y);
X_full = bishopXPoly(X, 1);
w_1 = bishopCurveFit(X_full, Y);
X_full = bishopXPoly(X, 3);
w_3 = bishopCurveFit(X_full, Y);
X_full = bishopXPoly(X, 9);
w_9 = bishopCurveFit(X_full, Y);

%%%%%% Plot on various M's %%%%%%%
h2 = figure;
hold on;
plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hw1_plot(w_0, 0, @bishopXPoly);
hw1_plot(w_1, 1, @bishopXPoly);
hw1_plot(w_3, 3, @bishopXPoly);
hw1_plot(w_9, 9, @bishopXPoly);
hold off;
h_legend = legend('Points', 'M=0','M=1', 'M=3', 'M=9')
set(h_legend,'FontSize',14);
set(h2,'Units','Inches');
pos = get(h2,'Position');
set(h2,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h2, 'hw1_2.pdf', '-dpdf', '-r0')




%%%% 2.2 %%%%
M = 9; X_full = bishopXPoly(X, M);
sse_calc = SSE(w_9);
sse_derivative = SSE_derivative(w_9);
fin_diff(@SSE, w_9, 0.001)
% verified: at optimal solution w, SSE derivative is basically 0

w_init = ones(size(w_9));
w_converge = grad_desc_2(@SSE, w_9+norm(w_9,2)*0.05*rand(size(w_9)), 0.1, 1e-6);
[w_9 w_converge]
% compare the formula fitted w vs. the GD algorithm 
w_nice = fminunc(@SSE, w_init);

h3 = figure;
hold on;
plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hw1_plot(w_9, 9, @bishopXPoly);
hw1_plot(w_nice,9, @bishopXPoly);
hold off;
h_legend = legend('Points', 'M=3, use formula','M=3, using SSE GD')
set(h_legend,'FontSize',14);
set(h3,'Units','Inches');
pos = get(h3,'Position');
set(h3,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h3, 'hw1_2_2.pdf', '-dpdf', '-r0')



%%%% 2.4 %%%% 
%%% using the sin basis function
X_full = bishopXSin(X, 1);
w_sin_1 = bishopCurveFit(X_full, Y);
X_full = bishopXSin(X, 4);
w_sin_4 = bishopCurveFit(X_full, Y);


h4 = figure;
hold on;
plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hw1_plot(w_sin_1, 1, @bishopXSin);
hw1_plot(w_sin_4, 4, @bishopXSin);
hold off;
h_legend = legend('Points', 'M=1','M=4')
set(h_legend,'FontSize',14);
set(h4,'Units','Inches');
pos = get(h4,'Position');
set(h4,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h4, 'hw1_2_4.pdf', '-dpdf', '-r0')


%%%% 3.1 %%%%
% M = 3 fixed, vary lambda
M = 3;
X_full = bishopXPoly(X, M);
w = bishopCurveFit(X_full, Y);
h5 = figure;
hold on;
plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hw1_plot(w, M, @bishopXPoly);
for lambda=[0.0001,0.05,100]
    w_ridge = ridge_reg(X_full, Y, lambda);
    hw1_plot(w_ridge, M, @bishopXPoly);
end
hold off;
h_legend = legend('Points', 'OLS', 'lambda=0.0001',...
        'lambda=0.05','lambda=100')
    set(h_legend,'FontSize',14);
set(h5,'Units','Inches');
pos = get(h5,'Position');
set(h5,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h5, 'hw1_3_1a.pdf', '-dpdf', '-r0')

% lambda = 0.05 fixed, vary M
h6 = figure;
hold on;
plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
for M=[1,3,5,7]
    X_full = bishopXPoly(X, M);
    w_ridge = ridge_reg(X_full, Y, 0.05);
    hw1_plot(w_ridge, M, @bishopXPoly);
end
hold off;
h_legend = legend('Points', 'M=1','M=3','M=5','M=7')
set(h_legend,'FontSize',14);
set(h6,'Units','Inches');
pos = get(h6,'Position');
set(h6,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h6, 'hw1_3_1b.pdf', '-dpdf', '-r0')   

%%
%%%% 3.2 %%%%
train = load('regress_train.txt');
valid = load('regress_validate.txt');
test = load('regress_test.txt');

x_train = train(1,:);
x_valid = valid(1,:);
x_test = test(1,:);
y_train = train(2,:);
y_valid = valid(2,:);
y_test = test(2,:);

min_sse = 10^10;
lambda_opt = 999;
M_opt = 999;
sse_all = zeros(21, 10);
sse_test_all = zeros(21, 10);
sse_test = 0;
for l=-10:10
    lambda = 10^l;
    for m=0:9
        % build model with training set
        X_full = bishopXPoly(x_train, m);
        Y = y_train;
        w = ridge_reg(X_full, Y, lambda);
        % compute SSE for validation set
        X_full = bishopXPoly(x_valid, m);
        Y = y_valid;
        sse_ridge = SSE(w);
        X_test = bishopXPoly(x_test, m);
        sse_test_all(l+11,m+1) = SSE_2(X_test, y_test, w);
        sse_all(l+11,m+1) = sse_ridge;
        if min_sse > sse_ridge
            min_sse = sse_ridge;
            sse_test = sse_test_all(l+11,m+1);
            lambda_opt = lambda;
            M_opt = m;
        end
    end
end

% w = ridge_reg(bishopXPoly(x_train, M_opt), y_train, lambda_opt);
% SSE_2(bishopXPoly(x_test, M_opt), y_test, w)
% plot log of MSE on the validation set %
h = figure;
colormap('parula')
x = [0 9];
y = [-10 10];
imagesc(x, y, log(sse_all/10))
z = colorbar
ylabel(z, 'Log of MSE');
xlabel('M')
ylabel('Log_{10}(\lambda)') 
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h, 'hw1_3_2.pdf', '-dpdf', '-r0')
% plot log of MSE on the test set %
h = figure;
colormap('parula')
x = [0 9];
y = [-10 10];
imagesc(x, y, log(sse_test_all/10))
z = colorbar
ylabel(z, 'Log of MSE');
xlabel('M')
ylabel('Log_{10}(\lambda)') 
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h, 'hw1_3_2b.pdf', '-dpdf', '-r0')

%%
%%%% 3.3 %%%%
x_test = load('BlogFeedback_data/x_test.csv');
x_train = load('BlogFeedback_data/x_train.csv');
x_val = load('BlogFeedback_data/x_val.csv');
y_test = load('BlogFeedback_data/y_test.csv');
y_train = load('BlogFeedback_data/y_train.csv');
y_val = load('BlogFeedback_data/y_val.csv');

%%
min_sse_blog = 10^10;
sse_blog_test = 0;
lambda_blog = 999;
M = 280; % simple linear model
X_train = ones(length(x_train), M+1);
X_train(:,2:(M+1)) = x_train;
Y_train = y_train';
X_val = ones(length(x_val), M+1);
X_val(:,2:(M+1)) = x_val;
Y_val = y_val';
X_test = ones(length(x_test), M+1);
X_test(:,2:(M+1)) = x_test;
Y_test = y_test';

for l=-4:10
    lambda = 10^l;
    % build model with training set
    w = ridge_reg(X_train, Y_train, lambda);
    % compute SSE for validation set
    sse_ridge = SSE_2(X_val, Y_val, w);
    if min_sse_blog > sse_ridge
        min_sse_blog = sse_ridge;
        lambda_blog = lambda;
        sse_blog_test = SSE_2(X_test,Y_test,w);
    end
end
%%
min_mse_blog = min_sse_blog/length(y_val)
mse_blog_test = sse_blog_test/length(y_test)
%%
%%%% 4.2 %%%%
data = load('regress-highdim.mat');
w_true = data.W_true;
data.X_train = (data.X_train)';
data.X_test = (data.X_test)';
true_fun = @(x) w_true(3)*sin(0.4*pi*x*2) + w_true(9)*sin(0.4*pi*x*8);

M = 12;
X_train =data.X_train;
X_test = data.X_test;
Y_train = data.Y_train;
Y_test = data.Y_test;
%%
% Compute Ridge Regression using Gradient Descent
lambda_2 = 0.1;
X_full = X_train;
n = size(X_full, 1);
Y = Y_train;
fun_ridge = @(w) (1/n)*SSE(w) + (lambda_2)*norm(w,2);
w_ridge = grad_desc_2(fun_ridge, zeros(M, 1), .01, 1e-8);
ridge_2D = @(x) sine_curve(w_ridge, x);

% Compute Ridge Regression with lambda_2 = 0
fun_ridge_0 = @(w) (1/n)*SSE(w);
w_ridge_0 = grad_desc_2(fun_ridge_0, zeros(M, 1), .1, 1e-8);
% w_ridge_0 = bishopCurveFit(X_full, Y);
ridge_0_2D = @(x) sine_curve(w_ridge_0, x);

% Compute LASSO using Gradient Descent
lambda_1 = 0.1;
X_full = X_train;
Y = Y_train;
fun_lasso = @(w) (1/n)*SSE(w) + (lambda_1)*norm(w,1);
w_lasso = grad_desc_2(fun_lasso, zeros(M, 1), .01, 1e-8);
lasso_2D = @(x) sine_curve(w_lasso, x);

% Plot curves
h4_1 = figure;
hold on;
plot(X_train(:,1), Y_train, 'p', 'MarkerSize', 10,'color','b');
ezplot(true_fun,[-1.2,1.2, -3, 3])
ezplot(ridge_0_2D,[-1.2,1.2, -3, 3])
ezplot(ridge_2D,[-1.2,1.2, -3, 3])
ezplot(lasso_2D,[-1.2,1.2, -3, 3])
% set(gcf,'')
hold off;
h_legend = legend('Points', 'W True', 'OLS','Ridge','LASSO',...
'Location','NorthWest')
set(h_legend,'FontSize',14);
set(h4_1,'Units','Inches');
title('');
pos = get(h4_1,'Position');
set(h4_1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h4_1, 'hw1_4_1.pdf', '-dpdf', '-r0')

%%
% Compute MSE on test data
n_test = size(X_test, 1)
w_ridge
sprintf('%.2f,',w_ridge)
w_ridge_0
sprintf('%.2f,',w_ridge_0)
w_lasso
sprintf('%.2f,',w_lasso)
MSE_ridge = SSE_2(X_test, Y_test, w_ridge)/n_test
MSE_ridge_0 = SSE_2(X_test, Y_test, w_ridge_0)/n_test
MSE_lasso = SSE_2(X_test, Y_test, w_lasso)/n_test
MSE_true = SSE_2(X_test, Y_test, w_true')/n_test 
