function [w] = bishopCurveFit(X_full, Y)
    d = size(X_full, 2);
    MP_inv = eye(d)/(X_full' * X_full) * X_full';
    w = MP_inv * Y';
end
