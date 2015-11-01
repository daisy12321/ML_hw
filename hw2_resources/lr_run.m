function [z, f] = lr_run(X,Y,lambda,addbias)
    [n, p] = size(X);
    if addbias == true
        X_train = [ones(n,1) X];
    else
        X_train = X;
    end
    Y_train = Y;
    
    LR_loss = @(w) lambda * norm(w(2:end), 2)^2 + sum(-log(sigmoid(Y_train .* (X_train * w'))));
    options = optimoptions(@fminunc,'Display','iter');
    [z, f] = fminunc(LR_loss, zeros(1,p+1));%, options);
    
    % LR_grad = @(w) 2 * lambda * w  + sum((-repmat(Y_train, 1, p) .* X_train) ...
    %               .* repmat(1 - sigmoid(Y_train .* (X_train * w')), 1, p));

    % [x, f, x_hist, f_hist] = grad_desc(LR_loss, LR_grad, [0 0 0], 0.005, 1e-6)
    %[x, f] = grad_desc_2(LR_loss, [1  1 1], 3, 1e-6)

end
