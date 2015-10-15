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
    
    alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub);
end