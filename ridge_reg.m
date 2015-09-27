<<<<<<< HEAD
%%%% Problem 3.1
% Implement Ridge Regression
% (Least squares with quadratic regularizer)
% Optimal (analytic) solution is: w 
=======
function w = ridge_reg(X_full, Y, M, lambda)
    MP_inv = eye(M+1)/(X_full' * X_full + lambda * eye(M+1)) * X_full';
    w = MP_inv * Y';
>>>>>>> ff76b2a1a7aa86bf037f672e029e21a90f52a334
