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

% compare to Matlab native function:

options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton');
[x2, f2] = fminunc(fun, [1,1])
% x3 = fminunc(@(x)norm(x-1)^2,11.0)
% fun = @(x)3*x(1)^2 + 2*x(1)*x(2) + x(2)^2 - 4*x(1) + 5*x(2);


%%% 1.3 %%%%
%%% implement finite differencing to evaluate gradient
grad_numeric = fin_diff(fun, [1, 2], 0.001)
grad_analytic = grad([1, 2]) %compare with analytic 

% test on the previous examples
[x, f] = grad_desc(fun, grad, [1.0 -2.0], 0.1, 1e-10)
[x, f] = grad_desc_2(fun, [1.0 -2.0], 0.1, 1e-10)

[x, f] = grad_desc(fun2, grad2, 0.1, 0.01, 1e-10)
[x, f] = grad_desc_2(fun2, 0.1, 0.01, 1e-10)

[x, f] = grad_desc(fun3, grad3, 0.1, 0.01, 1e-10)
[x, f] = grad_desc_2(fun3, 0.1, 0.01, 1e-10)
% all match!


%%% 1.4 %%%%


%%%%%%%%%%%%
%%% 2.1 %%%%
[X, Y] = bishopCurveData();
M = 3;
format short
% Using our function
[X_full, w0, w_other] = bishopCurveFit(X, Y, M);
w = [w0 w_other']
% using polyfit to verify
w_polyfit = fliplr(polyfit(X, Y, M));
% Same as [w0, w_other]!

% generate points for plotting
x_1 = (0:0.01:1.0);
y_1 = zeros(1,length(x_1)) + w0;
for i = 1:M
    y_1 = y_1 + w_other(i) * (x_1 .^ i);
end


figure;

plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hold on;

plot(x_1,y_1);
xlabel('x');
ylabel('y');
grid on


%%%% 2.2 %%%%
sse = SSE(X, Y, M, w)
