function [x_new f_new] = grad_desc_2(fun, x0, step, eps)
f_old = fun(x0);
grad = fin_diff(fun, x0, 0.001);
disp(grad)
x_new = x0 - step*grad';
f_new = fun(x_new);
%disp(f_new)
while abs(f_new - f_old) > eps
%    disp(x_new);
    f_old = f_new;
    grad = fin_diff(fun, x_new, 0.001);
    x_new = x_new-step*grad';
    f_new = fun(x_new);
end
