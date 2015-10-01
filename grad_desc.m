function [x_new f_new x_history] = grad_desc(fun, grad, x0, step, eps)
f_old = fun(x0);
x_new = x0 - step*grad(x0);
x_history = zeros(1000, size(x0,2));
x_history(1, :) = x_new;
f_new = fun(x_new);
counter = 1;
%disp(f_old)
%disp(f_new)
while abs(f_new - f_old) > eps
    counter = counter + 1;
%    disp(x_new);
    f_old = f_new;
    x_new = x_new-step*grad(x_new);
    f_new = fun(x_new);
    x_history(counter, :) = x_new;
end
idx_nz = find(x_history(:, 1))
x_history = x_history(idx_nz, :)
disp(counter)

