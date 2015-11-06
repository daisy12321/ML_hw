function [F,a1, Z, a2] = fwd_prop(x, W1, W2)
    if size(x, 1) == 1
        x = x';
    end
    
    % level 1
    [M, d] = size(W1);
    a1 = zeros(M, 1);
    for j = 1:M
         a1(j) = W1(j, :)*x;
    end
    Z = sigmoid(a1);
    
    % level 2
    K = size(W2, 1);
    a2 = zeros(K, 1);
    for j = 1:K
        a2(j) = W2(j, :)*Z;
    end
    F = sigmoid(a2);