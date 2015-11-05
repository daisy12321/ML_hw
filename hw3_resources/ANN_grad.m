function [dl_w1, dl_w2] = ANN_grad(W1, W2, X, Y)
    [K, M] = size(W2);
    [M, d] = size(W1);
    
    [a1, Z, a2, h] = fwd_prop(X, W1, W2);%compute estimated y_k from x and w

    dl_w2 = zeros(1, M);
    dl_w1 = zeros(K, M);

    dSig2 = zeros(1, K);

    for k = 1:K
        % Layer 2
        dSig2(k) = sigmoid(a2(k))*(1-sigmoid(a2(k)));
        dE2Log_tmp = 1/h(k)*dSig2(k)*Z';
        dE2Log_sub_tmp = -1/(1-h(k))*dSig2(k)*Z';
        dl_w2 = dl_w2 + -Y(k).* dE2Log_tmp - (1-Y(k)) .* dE2Log_sub_tmp;
    end
    disp(dl_w2);

    dSig1 = sigmoid(a1) .* (1-sigmoid(a1)); % in Mx1
    dSig1Matrix = repmat(dSig1', K, 1);
    for k = 1:K
        % layer 1
        dE1Log_tmp = 1/h(k)*dSig2(k)* W2 .* dSig1Matrix;
        dE1Log_sub_tmp = -1/(1-h(k))*dSig2(k)* W2 .* dSig1Matrix;
        dl_w1 = dl_w1 + -Y(k).* dE1Log_tmp - (1-Y(k)) .* dE1Log_sub_tmp;
    end
    disp(dl_w1);