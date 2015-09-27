function w = ridge_reg(X_full, Y, M, lambda)
    MP_inv = eye(M+1)/(X_full' * X_full + lambda * eye(M+1)) * X_full';
    w = MP_inv * Y';