function [w, w_0, H, alpha] = svm(X, Y, C, kernel, sigma2)
    n = length(Y);
    f = ones(n, 1);
    A = [];
    b = [];
    Aeq = Y';
    beq = 0;
    lb = zeros(n, 1);
    ub = C*ones(n, 1);
    
    gaussian = @(x,z) rbf(x,z,sigma2);
    
    % Compute kernel matrix
    K = zeros(n);
    if kernel=='rbf'
        for i=1:n
            for j=1:n
                K(i,j) = rbf(X(i,:),X(j,:),sigma2);
            end
        end
    else
        for i=1:n
            for j=1:n
                K(i,j) = dot(X(i,:),X(j,:));
            end
        end
    end
        
    H = zeros(n);
    for i=1:n
        for j=1:n
            H(i,j) = Y(i)*Y(j)*K(i,j);
        end
    end
    
    % Solve quadratic program
    alpha = quadprog(H,-f,A,b,Aeq,beq,lb,ub);
    alpha = round(alpha, 4);
    
    % Solve for w, w_0 parameters
    w = alpha(1)*Y(1)*X(1,:);
    for i=2:n
        w = w + alpha(i)*Y(i)*X(i,:);
    end
    
    % S = set of support vectors, M = {i : 0 < alpha_i < C}
    S = alpha > zeros(n,1);
    M = (zeros(n,1) < alpha).*(alpha < C*ones(n,1));
    
    w_0 = 0;
    for j=1:n
        if M(j)
            w_0 = w_0 + Y(j);
            for i=1:n
                if S(i)
                    w_0 = w_0 - alpha(i)*Y(i)*K(i,j);
                end
            end
        end
    end
    w_0 = w_0/sum(M);
end

