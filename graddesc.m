function [x_new f_new] = graddesc(fun, grad, x0, step, eps)
f_old = fun(x0);
step = 
x_new = x0-step*grad(x0);
f_new = fun(x_new);
while abs(f_new - f_old) > eps
    disp(x_new);
    f_old = f_new;
    x_new = x_new-step*grad(x_new);
    f_new = fun(x_new);
end
