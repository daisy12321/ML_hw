function [dl_w1, dl_w2] = ANN_grad(W1, W2, x, y)
    if size(W2, 2) ~= size(W1, 1)
        error('Dimension incompatible')
    else
        
        [K, M] = size(W2);
        [M, d] = size(W1);

        [h, a1, Z, a2] = fwd_prop(x, W1, W2);   %compute estimated y_k from x and w

        dl_w2 = zeros(K, M);
        dl_w1 = zeros(M, d);

        %%%%% Layer 2 %%%%%
        dSig2 = sigmoid(a2) .* (1-sigmoid(a2));
        
        for k = 1:K

            dE2Log_tmp = 1/h(k)*dSig2(k)*Z';
            dE2Log_sub_tmp = -1/(1-h(k))*dSig2(k)*Z';
            dl_w2(k, :) = -y(k) .* dE2Log_tmp - (1-y(k)) .* dE2Log_sub_tmp;

        end

        %%%%% Layer 1 %%%%%
        dSig1 = sigmoid(a1) .* (1-sigmoid(a1)); % Mx1

        for k = 1:K

            dE1Log_tmp = 1/h(k)*dSig2(k)*(W2(k, :)' .* dSig1) * x; % W2': Mx1   x: 1xD
            dE1Log_sub_tmp = -1/(1-h(k))*dSig2(k)*(W2(k, :)' .* dSig1) * x;
            dl_w1 = dl_w1 + -y(k).* dE1Log_tmp - (1-y(k)) .* dE1Log_sub_tmp;

        end
    end