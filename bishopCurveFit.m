function [X_full, w0, w_other] = bishopCurveFit(X, Y, M)
    X_full = ones(length(X), M+1);
    for i = 1:(M+1)
        X_full(:, i) = X .^ (i-1);
    end
   
    MP_inv = eye(M+1)/(X_full' * X_full) * X_full';
    w = MP_inv * Y';
    
    w0 = w(1);
    w_other = w(2:length(w));

end
