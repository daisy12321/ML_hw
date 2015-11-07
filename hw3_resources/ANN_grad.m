function [dl_w1, dl_w2] = ANN_grad(W1, W2, x, y, lambda)
    if size(W2, 2) ~= size(W1, 1)+1
        error('Dimension incompatible')
    else
        
        K = size(W2, 1);
        [M, D] = size(W1);

        [h, a1, Z, a2] = fwd_prop(x, W1, W2);   %compute estimated y_k from x and w

        dl_w2 = zeros(K, M+1);
        dl_w1 = zeros(M, D);

        %%%%% Layer 2 %%%%%
        % dSig2 = sigmoid(a2) .* (1-sigmoid(a2));
        
        for k = 1:K

            dE2Log_tmp = (1-sigmoid(a2(k)))*Z';  %dE2Log_tmp = 1/h(k)*dSig2(k)*Z';
            dE2Log_sub_tmp = -sigmoid(a2(k))*Z';
            dl_w2(k, :) = -y(k) .* dE2Log_tmp - (1-y(k)) .* dE2Log_sub_tmp;

        end
        
        W2_no_intercept = W2;
        W2_no_intercept(:, 1) = zeros(K, 1);
        dl_w2 = dl_w2 + 2*lambda*W2_no_intercept;
        
        
        %%%%% Layer 1 %%%%%
        dSig1 = sigmoid(a1) .* (1-sigmoid(a1)); % Mx1

        for k = 1:K

            dE1Log_tmp = (1-sigmoid(a2(k)))*(W2(k, 2:M+1)' .* dSig1) * x; % W2': Mx1   x: 1xD
            dE1Log_sub_tmp = -sigmoid(a2(k))*(W2(k, 2:M+1)' .* dSig1) * x;
            
            dl_w1 = dl_w1 + -y(k).* dE1Log_tmp - (1-y(k)) .* dE1Log_sub_tmp;

        end
        
        W1_no_intercept = W1;
        W1_no_intercept(:, 1) = zeros(M, 1);
        dl_w1 = dl_w1 + 2*lambda*W1_no_intercept;
   
    end