function hw1_plot(w, M, basis_fun)
    % generate points for plotting
    x_1 = (0:0.01:1.0);
    x_full = basis_fun(x_1, M);
    y_1 = w' * x_full';

    plot(x_1,y_1);
    xlabel('x'); ylabel('y');