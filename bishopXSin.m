function X_full = bishopXSin(X, M)
    X_full = ones(length(X), M+1);
    for i = 1:M
        X_full(:, i+1) = sin(2*pi*i*X);
    end
    