function lsum = ANN_loss(W1, W2, X, Y, lambda)
    
    N = size(X, 1);
    [M, D] = size(W1);
    K = size(W2, 1);
    h = zeros(N, K);
    l = zeros(N, K);
    
    for i = 1:N
         h(i, :) = fwd_prop(X(i, :), W1, W2)';

         for k = 1:K
             l(i, k) = - Y(i,k)*log(h(i,k)) - (1-Y(i,k))*log(1-h(i,k));
         end
    end
    
    lsum = sum(l(:))/N + lambda*sum(sum(W1(:, 2:D) .^ 2)) + lambda*sum(sum(W2(:, 2:M+1) .^ 2));
end