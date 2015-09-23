function [X_full, w0, w_other] = bishopCurveFit(X, Y, M)
    X_full = ones(length(X), M);
    for i = 1:M
        X_full(:, i) = X .^ (i-1);
    end
   
    MP_inv = inv(X_full' * X_full) * X_full';
    w = MP_inv * Y';
    disp(w);
    
    w0 = w(1);
    w_other = w(2:length(w));

end
