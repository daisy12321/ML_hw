function lsum2 = ANN_loss(W1, W2, X, Y)
    N = size(X, 1);
    [K, M] = size(W2);
    h = zeros(N, K);
    l = zeros(N, K);
    for i = 1:N
        [tmp1 tmp2 tmp3 tmp4] = fwd_prop(X(i, :), W1, W2);
        h(i, :) = tmp4';
         for k = 1:K
             l(i, k) = - Y(i,k)*log(h(i,k)) - (1-Y(i,k))*(log(1-h(i,k)));
         end
    end
    disp(l)
    lsum = sum(l)
    lsum2 = sum(lsum)
end