function X_full = bishopXPoly(X, M)
    X_full = ones(length(X), M+1);
    for i = 1:(M+1)
        X_full(:, i) = X .^ (i-1);
    end
    