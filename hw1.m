%%% 1.1 & 1.2 %%%%
% self-implement function on gradient descend
% constant step size
% convex quadratic function
fun = @(x) 3*x(1)^2 + 2*x(1)*x(2) + x(2)^2 - 4*x(1) + 5*x(2)
grad = @(x)[6*x(1) + 2*x(2) - 4, 2*x(1) + 2*x(2) + 5]
[x, f] = graddesc(fun, grad, [1.0 -2.0], 0.1, 0.0001)

% plot the contour and the optimum point
figure;
ezcontour('3*x^2 + 2*x*y + y^2 - 4*x + 5*y', [-10, 10], [-10, 10])
hold on;
plot(x(1), x(2), '+','color','r');

% compare to Matlab native function:
[x2, f2] = fminunc(fun, [1,1])
% x3 = fminunc(@(x)norm(x-1)^2,11.0)
% fun = @(x)3*x(1)^2 + 2*x(1)*x(2) + x(2)^2 - 4*x(1) + 5*x(2);

%%% 1.3 %%%%



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
