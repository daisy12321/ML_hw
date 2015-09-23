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
format short
% Using our function
[X_full, w0, w_other] = bishopCurveFit(X, Y, M);

% using polyfit to verify
w_polyfit = fliplr(polyfit(X, Y, M-1));
% Same as [w0, w_other]!

% generate points for plotting
x_1 = (0:0.01:1.0);
y_1 = zeros(1,length(x_1)) + w0;
for i = 1:(M-1)
    y_1 = y_1 + w_other(i) * (x_1 .^ i);
    disp(y_1(100));
end


figure;

plot(X, Y, 'o', 'MarkerSize', 10,'color','b');
hold on;

plot(x_1,y_1);
xlabel('x');
ylabel('y');
grid on
