function [predict_all, predict_class] = predict_multi_class(X, W1, W2)
    N = size(X, 1);
    K = size(W2, 1);
    
    predict_all = zeros(K, N);
    predict_class = zeros(1, N);
    
    for i = 1:N
        predict_all(:, i) = sigmoid(fwd_prop(X(i,:), W1, W2));
        [tmp, predict_class(i)] = max(predict_all(:,i));
    end

    %accu = sum(predict_class' == Y_train_lab)/length(Y_train_lab)
