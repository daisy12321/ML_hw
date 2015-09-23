%%% 1.1 & 1.2 %%%%
% self-implement function on gradient descend
% constant step size
% convex quadratic function
[x, f] = graddesc(@(x)(x(1)-1)^2 + (x(2)+0.5)^2, @(x)2*(x(1)-1) + 2*(x(2)+0.5), [1.5,0], 0.1, 0.0001)
%fn = @(x)(x(1)+x(2)-0.5)^2;
figure;
ezsurf('(x-1)^2 + (y+0.5)^2')


% compare to Matlab native function:
[x2, f2] = fminunc(@(x)(x(1)+x(2)-0.5)^2, [1,1])
% x3 = fminunc(@(x)norm(x-1)^2,11.0)
% fun = @(x)3*x(1)^2 + 2*x(1)*x(2) + x(2)^2 - 4*x(1) + 5*x(2);


%%% 2.1
[X, Y] = bishopCurveData();
M = 9;
[X_full, w0, w] = bishopCurveFit(X, Y, M);
%y_1 = @(x) w0 + w(1)*x + w(2)*x.^2 + w(3)*x.^3

x_1 = (0:0.01:1.0);
y_1 = zeros(length(x_1), 1);
for i = 0:(M-1)
    y_1 = y_1 + w(i) * x_1 .^ i 
end


figure;

plot(X, Y, 'o', 'MarkerSize', 10);
hold on;

plot(x_1,y_1(x_1));
xlabel('x');
ylabel('y');
