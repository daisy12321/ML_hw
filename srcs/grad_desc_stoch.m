function [w1_new, w2_new] = grad_desc_stoch(fun, w1_0, w2_0, X, Y, step, eps)
    
    N = size(X, 1);
    
    f_old = 1;
    f_new = 0;
    
    counter = 1;
    
    w1_old = w1_0; 
    w2_old = w2_0;
    
    i = 0;
    
    while abs(f_new - f_old) > eps && counter < 3000
        i = mod(i, N) + 1;
        
        if i == 1
            disp('Looping over data again')
            ordering = randperm(N);
            X = X(ordering,:);
            Y = Y(ordering,:);
        end
        
        f_old = fun(w1_old, w2_old, X, Y);
        [grad1, grad2] = ANN_grad(w1_old, w2_old, X(i, :), Y(i, :));
        
        step_size = 25/(counter + 50)^0.55;
        
        w1_new = w1_old - step_size*grad1;
        w2_new = w2_old - step_size*grad2;    
        f_new = fun(w1_new, w2_new, X, Y);
        
        % disp('Difference between objective values is '); 
        % disp(abs(f_old - f_new));
        
        w1_old = w1_new;
        w2_old = w2_new;
        counter = counter + 1;
        
    end
    
    if counter >= 2999
        disp('Not converge in 3000 iterations')
    end
    