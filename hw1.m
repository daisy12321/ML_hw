%%% 1.1 & 1.2 %%%%
% self-implement function on gradient descend
% constant step size
% convex quadratic function
fun = @(x) 3*x(1)^2 + 2*x(1)*x(2) + x(2)^2 - 4*x(1) + 5*x(2)
grad = @(x)[6*x(1) + 2*x(2) - 4, 2*x(1) + 2*x(2) + 5]
[x, f] = grad_desc(fun, grad, [1.0 -2.0], 0.1, 1e-10)
% plot the contour and the optimum point
figure;
ezcontour('3*x^2 + 2*x*y + y^2 - 4*x + 5*y', [-10, 10], [-10, 10])
hold on;
plot(x(1), x(2), '+','color','r');
hold off;

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
grad_numeric = fin_diff(fun, [1, 2], 0.001)
grad_analytic = grad([1, 2]) %compare with analytic 

% test on the previous examples
[x, f] = grad_desc(fun, grad, [1.0 -2.0], 0.1, 1e-10)
[x, f] = grad_desc_2(fun, [1.0 -2.0], 0.1, 1e-10)

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
w = bishopCurveFit(X_full, Y, M);
% using polyfit to verify
w_polyfit = fliplr(polyfit(X, Y, M));
% Same as w!


%%%% 2.2 %%%%
sse_calc = SSE(w)
sse_derivative = SSE_derivative(w)
fin_diff(@SSE, w, 0.001)
% verified: at optimal solution w, SSE derivative is basically 0

w_init = zeros(size(w));
w_converge = grad_desc_2(@SSE, w, 0.01, 1e-8);
% compare the formula fitted w vs. the GD algorithm 

figure;
hold on;
plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hw1_plot(w, @bishopXPoly);
hw1_plot(w_converge,@bishopXPoly);
hold off;
legend('Points', 'Direct solve','Gradient descend')

%%%% 2.3 %%%% 
%%% using the sin basis function
M = 4;
X_full2 = bishopXSin(X, M);
w = bishopCurveFit(X_full2, Y, M);

figure;
hold on;
plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hw1_plot(w, @bishopXSin);
hold off;


%%%% 3.1 %%%%
M = 3;
X_full = bishopXPoly(X, M);
w = bishopCurveFit(X_full, Y, M);
w_ridge = ridge_reg(X_full, Y, M, 0.05);
figure;
hold on;
plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hw1_plot(w, @bishopXPoly);
hw1_plot(w_ridge,@bishopXPoly);
hold off;
legend('Points', 'OLS', 'Ridge regression, lambda = 0.05')

M = 3;
X_full = bishopXPoly(X, M);
w_ridge = ridge_reg(X_full, Y, M, 0.0001);
figure;
hold on;
plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hw1_plot(w, @bishopXPoly);
hw1_plot(w_ridge,@bishopXPoly);
hold off;
legend('Points', 'OLS', 'Ridge regression, lambda = 0.0001')

M = 3;
X_full = bishopXPoly(X, M);
w_ridge = ridge_reg(X_full, Y, M, 100);
figure;
hold on;
plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hw1_plot(w, @bishopXPoly);
hw1_plot(w_ridge,@bishopXPoly);
hold off;
legend('Points', 'OLS', 'Ridge regression, lambda = 100')

M = 1;
X_full = bishopXPoly(X, M);
w_ridge = ridge_reg(X_full, Y, M, 0.05);
figure;
hold on;
plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hw1_plot(w, @bishopXPoly);
hw1_plot(w_ridge,@bishopXPoly);
hold off;
legend('Points', 'OLS', 'Ridge regression, M=1, lambda=0.05')

M = 5;
X_full = bishopXPoly(X, M);
w_ridge = ridge_reg(X_full, Y, M, 0.05);
figure;
hold on;
plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hw1_plot(w, @bishopXPoly);
hw1_plot(w_ridge,@bishopXPoly);
hold off;
legend('Points', 'OLS', 'Ridge regression, M=5, lambda=0.05')
