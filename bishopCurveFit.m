function [w] = bishopCurveFit(X_full, Y, M)
    MP_inv = eye(M+1)/(X_full' * X_full) * X_full';
    w = MP_inv * Y';
end
