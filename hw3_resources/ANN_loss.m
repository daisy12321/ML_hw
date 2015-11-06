function lsum = ANN_loss(W1, W2, X, Y)
    
    N = size(X, 1);
    K = size(W2, 1);
    h = zeros(N, K);
    l = zeros(N, K);
    
    for i = 1:N
         h(i, :) = fwd_prop(X(i, :), W1, W2)';

         for k = 1:K
             l(i, k) = - Y(i,k)*log(h(i,k)) - (1-Y(i,k))*(log(1-h(i,k)));
         end
    end
    
    lsum = sum(l(:));
end