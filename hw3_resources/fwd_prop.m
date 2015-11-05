function [a1, Z,a2, F] = fwd_prop(X, W1, W2)
    [M, d] = size(W1);
    a1 = zeros(M, 1);
    for j = 1:M
         a1(j) = W1(j, :)*X';
    end
    Z = sigmoid(a1);
    %disp(a1);
    %disp(Z);
    
    K = size(W2, 1);
    a2 = zeros(K, 1);
    for j = 1:K
        a2(j) = W2(j, :)*Z;
    end
    F = sigmoid(a2);
    %disp(a2);
    disp(F);