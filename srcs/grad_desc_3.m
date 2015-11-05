function [w1, w2] = grad_desc_3(fun, w1_0, w2_0, X, Y, step, eps)
    f_old = fun(w1_0, w2_0);
    x_new = w1_0 - step*ANN_grad(w1_0, w2_0, X, Y);
    x_history = zeros(1000, size(x0,2));
    x_history(1, :) = x_new;
    f_new = fun(x_new);
    f_hist = zeros(1000,1);
    counter = 1;
    while abs(f_new - f_old) > eps && counter < 1000
        disp(x_new)
        counter = counter + 1;
        f_old = f_new;
        x_new = x_new-step/counter*grad(x_new);
        f_new = fun(x_new);
        f_hist(counter) = f_new;
        x_history(counter, :) = x_new;
    end
    idx_nz = find(x_history(:, 1));
    x_history = x_history(idx_nz, :);
    f_hist = f_hist(idx_nz);
    disp(counter)

