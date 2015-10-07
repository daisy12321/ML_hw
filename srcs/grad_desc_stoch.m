function [x_new f_new] = grad_desc_stoch(fun, grad, x0, step, eps)
Global X Y;
f_old = fun(X_full, Y, x0);
%disp(grad)
x_new = x0 - step*grad;
f_new = fun(x_new);
counter = 1;
%disp(f_new)
while abs(f_new - f_old) > eps
    counter = counter + 1;
    %disp(x_new);
    f_old = f_new;
    grad = fin_diff(fun, x_new, 0.001);
    x_new = x_new-step*grad;
    f_new = fun(x_new);
end
disp(counter)

