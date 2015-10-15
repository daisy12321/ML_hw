function [w, w_0] = svm(X, Y, C)
    n = length(Y);
    f = ones(n, 1);
    A = [];
    b = [];
    Aeq = Y';
    beq = 0;
    lb = zeros(n, 1);
    ub = C*ones(n, 1);
    
    H = zeros(n);
    for i=1:n
        for j=1:n
            H(i,j) = Y(i)*Y(j)*X(i,:)*X(j,:)';
        end
    end
    
    % Solve quadratic program
    alpha = quadprog(H,-f,A,b,Aeq,beq,lb,ub);
    
    % Solve for w, w_0 parameters
    sum = alpha(1)*Y(1)*X(1,:);
    for i=2:n
        sum = sum + alpha(i)*Y(i)*X(i);
    end
    w = sum;
    % S = set of support vectors, M = {i : 0 < alpha_i < C}
    S = alpha > zeros(n,1)
    
end