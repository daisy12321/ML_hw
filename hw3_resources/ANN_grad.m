function [dl_w1, dl_w2] = ANN_grad(W1, W2, x, y)
    [K, M] = size(W2);
    [M, d] = size(W1);
    
    [a1, Z, a2, h] = fwd_prop(x, W1, W2);%compute estimated y_k from x and w

    dl_w2 = zeros(1, M);
    dl_w1 = zeros(M, d);

    dSig2 = zeros(1, K);

    for k = 1:K
        % Layer 2
        dSig2(k) = sigmoid(a2(k))*(1-sigmoid(a2(k)));
        dE2Log_tmp = 1/h(k)*dSig2(k)*Z';
        dE2Log_sub_tmp = -1/(1-h(k))*dSig2(k)*Z';
        dl_w2 = dl_w2 + -y(k).* dE2Log_tmp - (1-y(k)) .* dE2Log_sub_tmp;
    end
    disp(dl_w2);

    dSig1 = sigmoid(a1) .* (1-sigmoid(a1)); % Mx1

    for k = 1:K
        % layer 1
        dE1Log_tmp = 1/h(k)*dSig2(k)*(W2(k, :)' .* dSig1) * x; % W2: Mx1   x: 1xD
        dE1Log_sub_tmp = -1/(1-h(k))*dSig2(k)*(W2(k, :)' .* dSig1) * x;
        dl_w1 = dl_w1 + -y(k).* dE1Log_tmp - (1-y(k)) .* dE1Log_sub_tmp;
    end
    disp(dl_w1);