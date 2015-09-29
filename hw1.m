%%% 1.1 & 1.2 %%%%
% self-implement function on gradient descend
% constant step size
% convex quadratic function
fun = @(x) 3*x(1)^2 + 2*x(1)*x(2) + x(2)^2 - 4*x(1) + 5*x(2)
grad = @(x)[6*x(1) + 2*x(2) - 4, 2*x(1) + 2*x(2) + 5]
[x, f] = grad_desc(fun, grad, [1.0 -2.0], 0.1, 1e-10)
% plot the contour and the optimum point
h = figure;
ezcontour('3*x^2 + 2*x*y + y^2 - 4*x + 5*y', [-10, 10], [-10, 10])
hold on;
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
w = bishopCurveFit(X_full, Y, M);
% using polyfit to verify
w_polyfit = fliplr(polyfit(X, Y, M));
% Same as w!

% replicate Bishop 1.4 plots
X_full = bishopXPoly(X, 0);
w_0 = bishopCurveFit(X_full, Y, 0);
X_full = bishopXPoly(X, 1);
w_1 = bishopCurveFit(X_full, Y, 1);
X_full = bishopXPoly(X, 3);
w_3 = bishopCurveFit(X_full, Y, 3);
X_full = bishopXPoly(X, 9);
w_9 = bishopCurveFit(X_full, Y, 9);

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
M = 3; X_full = bishopXPoly(X, M);
sse_calc = SSE(w);
sse_derivative = SSE_derivative(w);
fin_diff(@SSE, w, 0.001)
% verified: at optimal solution w, SSE derivative is basically 0

w_init = ones(size(w));
w_converge = grad_desc_2(@SSE, w_init, 0.05, 1e-6);
[w w_converge]
% compare the formula fitted w vs. the GD algorithm 

h3 = figure;
hold on;
plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hw1_plot(w_3, 3, @bishopXPoly);
hw1_plot(w_converge,3, @bishopXPoly);
hold off;
h_legend = legend('Points', 'M=3, use formula','M=3, using SSE GD')
set(h_legend,'FontSize',14);
set(h3,'Units','Inches');
pos = get(h3,'Position');
set(h3,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h3, 'hw1_2_2.pdf', '-dpdf', '-r0')



%%%% 2.3 %%%% 
%%% using the sin basis function
X_full = bishopXSin(X, 1);
w_sin_1 = bishopCurveFit(X_full, Y, 1);
X_full = bishopXSin(X, 4);
w_sin_4 = bishopCurveFit(X_full, Y, 4);


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
w = bishopCurveFit(X_full, Y, M);
h5 = figure;
hold on;
plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hw1_plot(w, M, @bishopXPoly);
for lambda=[0.0001,0.05,100]
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
h_legend = legend('Points', 'M=1','M=3','M=5','M=7','M=9')
set(h_legend,'FontSize',14);
set(h6,'Units','Inches');
pos = get(h6,'Position');
set(h6,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h3, 'hw1_3_1b.pdf', '-dpdf', '-r0')
