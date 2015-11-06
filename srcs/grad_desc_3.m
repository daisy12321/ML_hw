function [w1_new, w2_new] = grad_desc_3(fun, w1_0, w2_0, X, Y, step, eps)
    
    N = size(X, 1);
    
    f_old = 1;
    f_new = 0;
    
    counter = 1;
    
    w1_old = w1_0; 
    w2_old = w2_0;
    
    while abs(f_new - f_old) > eps && counter < 1000
        
        f_old = fun(w1_old, w2_old, X, Y);
        grad1_sum = zeros(size(w1_0));
        grad2_sum = zeros(size(w2_0));
    
        for i = 1:N
            [grad1, grad2] = ANN_grad(w1_old, w2_old, X(i, :), Y(i, :));
            grad1_sum = grad1_sum + grad1;
            grad2_sum = grad2_sum + grad2;
        end
        
        w1_new = w1_old - step/counter*grad1_sum;
        w2_new = w2_old - step/counter*grad2_sum;   
        f_new = fun(w1_new, w2_new, X, Y);
        
        %disp('Difference between objective values is '); 
        %disp(abs(f_old - f_new));
        
        w1_old = w1_new;
        w2_old = w2_new;
        counter = counter + 1;
        
    end
    
