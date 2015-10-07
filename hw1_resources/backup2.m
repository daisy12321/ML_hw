%%
%%%% 3.1 %%%%
[X, Y] = bishopCurveData();
% M = 3 fixed, vary lambda
M = 3;
X_full = bishopXPoly(X, M);
w = bishopCurveFit(X_full, Y, M);
h5 = figure;
hold on;
plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hw1_plot(w, M, @bishopXPoly);
for lambda=[0.0001,0.05,100]
    X_full = bishopXPoly(X, M);
    w_ridge = ridge_reg(X_full, Y, M, lambda);
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
    w_ridge = ridge_reg(X_full, Y, M, 0.05);
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
for l=-10:10
    lambda = 10^l;
    for M=0:9
        % build model with training set
        X_full = bishopXPoly(x_train, M);
        Y = y_train;
        w = ridge_reg(X_full, Y, M, lambda);
        % compute SSE for validation set
        X_full = bishopXPoly(x_valid, M);
        Y = y_valid;
        sse_ridge = SSE(w); 
        if min_sse > sse_ridge
            min_sse = sse_ridge;
            lambda_opt = lambda;
            M_opt = M;
        end
    end
end


%%
%%%% 3.3 %%%%
x_test = load('BlogFeedback_data/x_test.csv');
x_train = load('BlogFeedback_data/x_train.csv');
x_val = load('BlogFeedback_data/x_val.csv');
y_test = load('BlogFeedback_data/y_test.csv');
y_train = load('BlogFeedback_data/y_train.csv');
y_val = load('BlogFeedback_data/y_val.csv');

min_sse_blog = 10^10;
lambda_blog = 999;
M = 280; % simple linear model
X_train = ones(length(x_train), M+1);
X_train(:,2:(M+1)) = x_train;
Y_train = y_train;
X_val = ones(length(x_val), M+1);
X_val(:,2:(M+1)) = x_val;
Y_val = y_val;
%%
for l=-4:2
    lambda = 10^l;
    % build model with training set
    w = ridge_reg(X_train, Y_train, M, lambda);
    % compute SSE for validation set
    sse_ridge = SSE(w);
    if min_sse_blog > sse_ridge
        min_sse_blog = sse_ridge;
        lambda_blog = lambda;
    end
end