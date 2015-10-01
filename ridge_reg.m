%%%% Problem 3.1
% Implement Ridge Regression
% (Least squares with quadratic regularizer)
% Optimal (analytic) solution is: w 

function w = ridge_reg(X_full, Y, lambda)
    d = size(X_full, 2);
    MP_inv = eye(d)/(X_full' * X_full + lambda * eye(d)) * X_full';
    w = MP_inv * Y';
